/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/* Contact ckubisch@nvidia.com (Christoph Kubisch) for feedback */


#include <algorithm>
#include <assert.h>

#include "renderer.hpp"
#include "resources_vk.hpp"

#include <nvh/nvprint.hpp>
#include <nvh/misc.hpp>
#include <nvmath/nvmath_glsltypes.h>

#include "common.h"


namespace idraster {

//////////////////////////////////////////////////////////////////////////


class RendererVK : public Renderer
{
public:
  enum Mode
  {
    MODE_PER_DRAW_BASEINST,
    MODE_PER_TRI_ID_GS,
    MODE_PER_TRI_PART_SEARCH_GS,
    MODE_PER_TRI_ID_FS,
    MODE_PER_TRI_PART_SEARCH_FS,
  };


  class TypeInstance : public Renderer::Type
  {
    bool isAvailable(const nvvk::Context& context) const { return true; }

    const char* name() const { return "per-draw part index"; }
    Renderer*   create() const
    {
      RendererVK* renderer = new RendererVK();
      renderer->m_mode     = MODE_PER_DRAW_BASEINST;
      return renderer;
    }
    unsigned int priority() const { return 8; }

    Resources* resources() { return ResourcesVK::get(); }
  };

  class TypePrimGS : public Renderer::Type
  {
    bool isAvailable(const nvvk::Context& context) const { return true; }

    const char* name() const { return "per-tri part index gs"; }
    Renderer*   create() const
    {
      RendererVK* renderer = new RendererVK();
      renderer->m_mode     = MODE_PER_TRI_ID_GS;
      return renderer;
    }
    unsigned int priority() const { return 0; }

    Resources* resources() { return ResourcesVK::get(); }
  };

  class TypePrimSearchGS : public Renderer::Type
  {
    bool isAvailable(const nvvk::Context& context) const { return true; }

    const char* name() const { return "per-tri search part index gs"; }
    Renderer*   create() const
    {
      RendererVK* renderer = new RendererVK();
      renderer->m_mode     = MODE_PER_TRI_PART_SEARCH_GS;
      return renderer;
    }
    unsigned int priority() const { return 1; }

    Resources* resources() { return ResourcesVK::get(); }
  };

  class TypePrim : public Renderer::Type
  {
    bool isAvailable(const nvvk::Context& context) const { return true; }

    const char* name() const { return "per-tri part index fs"; }
    Renderer*   create() const
    {
      RendererVK* renderer = new RendererVK();
      renderer->m_mode     = MODE_PER_TRI_ID_FS;
      return renderer;
    }
    unsigned int priority() const { return 0; }

    Resources* resources() { return ResourcesVK::get(); }
  };

  class TypePrimSearch : public Renderer::Type
  {
    bool isAvailable(const nvvk::Context& context) const { return true; }

    const char* name() const { return "per-tri search part index fs"; }
    Renderer*   create() const
    {
      RendererVK* renderer = new RendererVK();
      renderer->m_mode     = MODE_PER_TRI_PART_SEARCH_FS;
      return renderer;
    }
    unsigned int priority() const { return 1; }

    Resources* resources() { return ResourcesVK::get(); }
  };

public:
  bool init(const CadScene* NV_RESTRICT scene, Resources* resources, const Config& config, Stats& stats) override;
  void deinit() override;
  void draw(const Resources::Global& global, Stats& stats) override;


  Mode m_mode;

  RendererVK() {}

private:
  struct StateSetup
  {
    nvvk::ShaderModuleID vertexShader;
    nvvk::ShaderModuleID geometryShader;
    nvvk::ShaderModuleID fragmentShader;

    VkPipeline                   pipeline = VK_NULL_HANDLE;
    nvvk::DescriptorSetContainer container;
  };

  struct DrawSetup
  {
    VkCommandBuffer cmdBuffer = VK_NULL_HANDLE;

    size_t fboChangeID;
    size_t pipeChangeID;
  };

  std::vector<DrawItem> m_drawItems;
  std::vector<uint32_t> m_seqIndices;
  VkCommandPool         m_cmdPool;
  DrawSetup             m_draw;
  StateSetup            m_setup;

  ResourcesVK* NV_RESTRICT m_resources;

  void fillCmdBuffer(VkCommandBuffer cmd, const DrawItem* NV_RESTRICT drawItems, size_t drawCount)
  {
    const ResourcesVK* res   = m_resources;
    const CadSceneVK&  scene = res->m_scene;

    int      lastMaterial     = -1;
    int      lastGeometry     = -1;
    int      lastMatrix       = -1;
    uint32_t lastUniqueOffset = ~0;

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_setup.container.getPipeLayout(), 0, 1,
                            m_setup.container.getSets(), 0, NULL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_setup.pipeline);

    for(size_t idx = 0; idx < drawCount; idx++)
    {
      const DrawItem&             di  = drawItems[idx];
      const CadSceneVK::Geometry& geo = scene.m_geometry[di.geometryIndex];

      if(lastGeometry != di.geometryIndex)
      {
        vkCmdBindVertexBuffers(cmd, 0, 1, &geo.vbo.buffer, &geo.vbo.offset);
        vkCmdBindIndexBuffer(cmd, geo.ibo.buffer, geo.ibo.offset, VK_INDEX_TYPE_UINT32);

        lastGeometry = di.geometryIndex;

        if(m_mode == MODE_PER_TRI_ID_GS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(), VK_SHADER_STAGE_GEOMETRY_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.triangleIdsAddr);
        }
        else if(m_mode == MODE_PER_TRI_PART_SEARCH_GS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(), VK_SHADER_STAGE_GEOMETRY_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.partIdsAddr);
        }
        else if(m_mode == MODE_PER_TRI_ID_FS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(), VK_SHADER_STAGE_FRAGMENT_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.triangleIdsAddr);
        }
        else if(m_mode == MODE_PER_TRI_PART_SEARCH_FS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(), VK_SHADER_STAGE_FRAGMENT_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.partIdsAddr);
        }
      }

      if(lastMatrix != di.matrixIndex)
      {
        vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(), VK_SHADER_STAGE_VERTEX_BIT,
                           offsetof(DrawPushData, matrixIndex), sizeof(uint32_t), &di.matrixIndex);

        lastMatrix = di.matrixIndex;
      }

      int materialIndex = di.materialIndex;
      if (m_config.colorizeDraws)
      {
        materialIndex = int(idx);
      }

      if(lastMaterial != materialIndex)
      {
        vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(), VK_SHADER_STAGE_FRAGMENT_BIT,
                           offsetof(DrawPushData, materialIndex), sizeof(uint32_t), &materialIndex);

        lastMaterial = materialIndex;
      }

      if (di.objectOffset != lastUniqueOffset)
      {
        vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(), VK_SHADER_STAGE_FRAGMENT_BIT,
                           offsetof(DrawPushData, uniquePartOffset), sizeof(uint32_t), &di.objectOffset);

        lastUniqueOffset = di.objectOffset;
      }

      // drawcall

      uint32_t instanceIndex;
      switch(m_mode)
      {
        case RendererVK::MODE_PER_DRAW_BASEINST:
          // directly pass the partIndex as instanceIndex
          instanceIndex = di.partIndex;
          break;
        case RendererVK::MODE_PER_TRI_ID_GS:
        case RendererVK::MODE_PER_TRI_ID_FS:
          // the partIndex is encoded per triangle in the idsBuffer
          // instanceIndex will encode the offset into the per-triangle idsBuffer, / 3 because 3 indices per triangle.
          instanceIndex = uint32_t(di.range.offset / sizeof(uint32_t)) / 3;
          break;
        case RendererVK::MODE_PER_TRI_PART_SEARCH_GS:
        case RendererVK::MODE_PER_TRI_PART_SEARCH_FS:
          // the partIndex is encoded per triangle in the idsBuffer
          // instanceIndex will encode the offset into the per-triangle idsBuffer, / 3 because 3 indices per triangle.
          instanceIndex = di.partCount | (di.partIndex << 8);
          break;
        default:
          instanceIndex = 0;
          break;
      }
      vkCmdDrawIndexed(cmd, di.range.count, 1, uint32_t(di.range.offset / sizeof(uint32_t)), 0, instanceIndex);
    }
  }

  void setupCmdBuffer(const DrawItem* NV_RESTRICT drawItems, size_t drawCount)
  {
    const ResourcesVK* res = m_resources;

    VkCommandBuffer cmd = res->createCmdBuffer(m_cmdPool, false, false, true);
    res->cmdDynamicState(cmd);

    fillCmdBuffer(cmd, drawItems, drawCount);

    vkEndCommandBuffer(cmd);
    m_draw.cmdBuffer = cmd;
  }

  void deleteCmdBuffer() { vkFreeCommandBuffers(m_resources->m_device, m_cmdPool, 1, &m_draw.cmdBuffer); }

  void setupPipeline()
  {
    ResourcesVK* res    = m_resources;
    VkDevice     device = res->m_device;

    vkDestroyPipeline(device, m_setup.pipeline, nullptr);

    {
      nvvk::GraphicsPipelineState     state = res->m_gfxState;
      nvvk::GraphicsPipelineGenerator gen(state);
      gen.setRenderPass(res->m_framebuffer.passPreserve);
      gen.setDevice(device);
      // pipelines
      gen.setLayout(m_setup.container.getPipeLayout());
      state.depthStencilState.depthCompareOp      = VK_COMPARE_OP_LESS_OR_EQUAL;
      state.rasterizationState.cullMode           = VK_CULL_MODE_BACK_BIT;
      state.multisampleState.rasterizationSamples = res->m_framebuffer.samplesUsed;

      gen.addShader(res->m_shaderManager.get(m_setup.vertexShader), VK_SHADER_STAGE_VERTEX_BIT);
      gen.addShader(res->m_shaderManager.get(m_setup.fragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
      if(m_mode == RendererVK::MODE_PER_TRI_ID_GS || m_mode == RendererVK::MODE_PER_TRI_PART_SEARCH_GS)
      {
        gen.addShader(res->m_shaderManager.get(m_setup.geometryShader), VK_SHADER_STAGE_GEOMETRY_BIT);
      }
      m_setup.pipeline = gen.createPipeline();
    }
  }
};


static RendererVK::TypeInstance     s_type_instance_vk;
static RendererVK::TypePrim         s_type_prim_vk;
static RendererVK::TypePrimSearch   s_type_prim_search_vk;
static RendererVK::TypePrimGS       s_type_prim_gs_vk;
static RendererVK::TypePrimSearchGS s_type_prim_search_gs_vk;

bool RendererVK::init(const CadScene* NV_RESTRICT scene, Resources* resources, const Config& config, Stats& stats)
{
  ResourcesVK* NV_RESTRICT res    = (ResourcesVK*)resources;
  VkDevice                 device = res->m_device;
  m_resources                     = res;
  m_scene                         = scene;
  m_config                        = config;

  {
    std::string prepend = nvh::stringFormat("#define COLORIZE_DRAWS %d\n", config.colorizeDraws ? 1 : 0);

    // init shaders
    switch(m_mode)
    {
      case RendererVK::MODE_PER_DRAW_BASEINST:
        m_setup.fragmentShader =
            res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "drawid_instanceid.frag.glsl", prepend);
        m_setup.vertexShader = res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "drawid_instanceid.vert.glsl");
        break;
      case RendererVK::MODE_PER_TRI_ID_FS:
      case RendererVK::MODE_PER_TRI_PART_SEARCH_FS:
        m_setup.fragmentShader = res->m_shaderManager.createShaderModule(
            VK_SHADER_STAGE_FRAGMENT_BIT, "drawid_primid.frag.glsl",
            prepend + nvh::stringFormat("#define SEARCH_COUNT %d\n", m_mode == MODE_PER_TRI_PART_SEARCH_FS ? config.searchBatch : 0));
        m_setup.vertexShader = res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "drawid_primid.vert.glsl");
        break;
      case RendererVK::MODE_PER_TRI_ID_GS:
      case RendererVK::MODE_PER_TRI_PART_SEARCH_GS:
        m_setup.fragmentShader =
            res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "drawid_primid_gs.frag.glsl", prepend);
        m_setup.geometryShader = res->m_shaderManager.createShaderModule(
            VK_SHADER_STAGE_GEOMETRY_BIT, "drawid_primid_gs.geo.glsl",
            nvh::stringFormat("#define USE_GEOMETRY_SHADER_PASSTHROUGH %d\n", config.passthrough ? 1 : 0)
                + nvh::stringFormat("#define SEARCH_COUNT %d\n", m_mode == MODE_PER_TRI_PART_SEARCH_GS ? config.searchBatch : 0));
        m_setup.vertexShader = res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "drawid_primid_gs.vert.glsl");
        break;
      default:
        break;
    }

    if(!res->m_shaderManager.areShaderModulesValid())
    {
      return false;
    }

    // init container
    m_setup.container.init(device);
    m_setup.container.addBinding(DRAW_UBO_SCENE, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1,
                                 VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_VERTEX_BIT);
    m_setup.container.addBinding(DRAW_SSBO_MATRIX, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT);
    m_setup.container.addBinding(DRAW_SSBO_MATERIAL, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_setup.container.addBinding(DRAW_SSBO_RAY, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT);
    m_setup.container.initLayout();

    VkPushConstantRange ranges[3];
    uint32_t            rangeCount = 2;

    ranges[0].offset     = offsetof(DrawPushData, matrixIndex);
    ranges[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ranges[0].size       = sizeof(uint32_t);
    ranges[1].offset     = offsetof(DrawPushData, materialIndex);
    ranges[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    ranges[1].size       = sizeof(uint32_t) + sizeof(uint32_t);

    if(m_mode == RendererVK::MODE_PER_TRI_ID_FS || m_mode == RendererVK::MODE_PER_TRI_PART_SEARCH_FS)
    {
      ranges[1].size = sizeof(uint32_t) + sizeof(uint32_t) + sizeof(uint64_t);
    }
    else if(m_mode == RendererVK::MODE_PER_TRI_ID_GS || m_mode == RendererVK::MODE_PER_TRI_PART_SEARCH_GS)
    {
      ranges[2].offset     = offsetof(DrawPushData, idsAddr);
      ranges[2].stageFlags = VK_SHADER_STAGE_GEOMETRY_BIT;
      ranges[2].size       = sizeof(uint64_t);
      rangeCount           = 3;
    }

    m_setup.container.initPipeLayout(rangeCount, ranges);
    m_setup.container.initPool(1);

    {
      std::vector<VkWriteDescriptorSet> updateDescriptors;

      updateDescriptors.push_back(m_setup.container.makeWrite(0, DRAW_UBO_SCENE, &res->m_common.view.info));
      updateDescriptors.push_back(m_setup.container.makeWrite(0, DRAW_SSBO_MATRIX, &res->m_scene.m_buffers.matrices.info));
      updateDescriptors.push_back(m_setup.container.makeWrite(0, DRAW_SSBO_MATERIAL, &res->m_scene.m_buffers.materials.info));
      updateDescriptors.push_back(m_setup.container.makeWrite(0, DRAW_SSBO_RAY, &res->m_common.ray.info));

      vkUpdateDescriptorSets(device, updateDescriptors.size(), updateDescriptors.data(), 0, 0);
    }

    setupPipeline();
  }
  {
    VkResult                result;
    VkCommandPoolCreateInfo cmdPoolInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cmdPoolInfo.queueFamilyIndex        = 0;
    result                              = vkCreateCommandPool(device, &cmdPoolInfo, NULL, &m_cmdPool);
    assert(result == VK_SUCCESS);

    uint32_t maxCombine;

    switch(m_mode)
    {
      case RendererVK::MODE_PER_DRAW_BASEINST:
        maxCombine = 0;
        break;
      case RendererVK::MODE_PER_TRI_ID_GS:
      case RendererVK::MODE_PER_TRI_ID_FS:
        maxCombine = ~0;
        break;
      case RendererVK::MODE_PER_TRI_PART_SEARCH_GS:
      case RendererVK::MODE_PER_TRI_PART_SEARCH_FS:
        maxCombine = config.searchBatch;
        break;
      default:
        maxCombine = 0;
        break;
    }

    fillDrawItems(m_drawItems, scene, config, maxCombine, stats);

    setupCmdBuffer(m_drawItems.data(), m_drawItems.size());
  }

  m_draw.fboChangeID  = res->m_fboChangeID;
  m_draw.pipeChangeID = res->m_pipeChangeID;

  return true;
}

void RendererVK::deinit()
{
  deleteCmdBuffer();
  vkDestroyCommandPool(m_resources->m_device, m_cmdPool, nullptr);

  m_setup.container.deinit();
  vkDestroyPipeline(m_resources->m_device, m_setup.pipeline, nullptr);

  m_resources->m_shaderManager.destroyShaderModule(m_setup.geometryShader);
  m_resources->m_shaderManager.destroyShaderModule(m_setup.fragmentShader);
  m_resources->m_shaderManager.destroyShaderModule(m_setup.vertexShader);
}

void RendererVK::draw(const Resources::Global& global, Stats& stats)
{
  ResourcesVK* NV_RESTRICT res = m_resources;

  if(m_draw.pipeChangeID != res->m_pipeChangeID || m_draw.fboChangeID != res->m_fboChangeID)
  {
    setupPipeline();
    deleteCmdBuffer();
    setupCmdBuffer(m_drawItems.data(), m_drawItems.size());

    m_draw.fboChangeID  = res->m_fboChangeID;
    m_draw.pipeChangeID = res->m_pipeChangeID;
  }


  VkCommandBuffer primary = res->createTempCmdBuffer();
  {
    nvvk::ProfilerVK::Section profile(res->m_profilerVK, "Render", primary);
    {
      nvvk::ProfilerVK::Section profile(res->m_profilerVK, "Draw", primary);
      // upload scene data
      vkCmdUpdateBuffer(primary, res->m_common.view.buffer, 0, sizeof(SceneData), (const uint32_t*)&global.sceneUbo);
      // reset the buffer used for picking so that atomicMin would give us the lowest value
      vkCmdFillBuffer(primary, res->m_common.ray.buffer, 0, sizeof(RayData), ~0);
      
      res->cmdPipelineBarrier(primary);

      // render scene
      res->cmdBeginRenderPass(primary, true, true);
      vkCmdExecuteCommands(primary, 1, &m_draw.cmdBuffer);
      vkCmdEndRenderPass(primary);

      // copy the mouse-picking hit result from this frame
      // into the main ubo, so that we can use the result
      // for the next frame
      
      VkMemoryBarrier memBarrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      memBarrier.srcAccessMask   = VK_ACCESS_SHADER_WRITE_BIT;
      memBarrier.dstAccessMask   = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(primary, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1,
                           &memBarrier, 0, nullptr, 0, nullptr);

      VkBufferCopy cpy;
      cpy.dstOffset = sizeof(SceneData);
      cpy.srcOffset = 0;
      cpy.size      = sizeof(RayData);
      vkCmdCopyBuffer(primary, res->m_common.ray.buffer, res->m_common.view.buffer, 1, &cpy);
    }
  }
  vkEndCommandBuffer(primary);
  res->submissionEnqueue(primary);
}

}  // namespace idraster
