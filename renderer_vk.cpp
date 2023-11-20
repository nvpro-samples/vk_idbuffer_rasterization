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
#include <nvvk/debug_util_vk.hpp>

#include "common.h"

#include <numeric> // std::iota

namespace idraster {

//////////////////////////////////////////////////////////////////////////


class RendererVK : public Renderer
{
public:
  enum PartIdMode
  {
    MODE_PER_DRAW_BASEINST,
    MODE_PER_TRI_ID_GS,
    MODE_PER_TRI_BATCH_PART_SEARCH_GS,
    MODE_PER_TRI_ID_FS,
    MODE_PER_TRI_BATCH_PART_SEARCH_FS,
    MODE_PER_TRI_GLOBAL_PART_SEARCH_FS,
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
      renderer->m_mode     = MODE_PER_TRI_BATCH_PART_SEARCH_GS;
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
      renderer->m_mode     = MODE_PER_TRI_BATCH_PART_SEARCH_FS;
      return renderer;
    }
    unsigned int priority() const { return 1; }

    Resources* resources() { return ResourcesVK::get(); }
  };

  class TypeGlobalPrimSearch : public Renderer::Type
  {
    bool isAvailable(const nvvk::Context& context) const { return true; }

    const char* name() const { return "per-tri global search part index fs"; }
    Renderer*   create() const
    {
      RendererVK* renderer = new RendererVK();
      renderer->m_mode     = MODE_PER_TRI_GLOBAL_PART_SEARCH_FS;
      return renderer;
    }
    unsigned int priority() const { return 1; }

    Resources* resources() { return ResourcesVK::get(); }
  };

public:
  bool init(const CadScene* NV_RESTRICT scene, Resources* resources, const Config& config, Stats& stats) override;
  void deinit() override;
  void draw(const Resources::Global& global, Stats& stats) override;


  PartIdMode m_mode;

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
  ResBuffer             m_perDrawDataBuffer;
  ResBuffer             m_perDrawIndexBuffer;
  ResBuffer             m_indirectDrawBuffer;

  ResourcesVK* NV_RESTRICT m_resources;

  void fillCmdBuffer(VkCommandBuffer cmd, const DrawItem* NV_RESTRICT drawItems, size_t drawCount)
  {
    const ResourcesVK* res   = m_resources;
    const CadSceneVK&  scene = res->m_scene;

    int      lastMaterial     = -1;
    int      lastGeometry     = -1;
    int      lastMatrix       = -1;
    uint32_t lastUniqueOffset = ~0;
    VkBuffer lastVbo          = VK_NULL_HANDLE;
    VkBuffer lastIbo          = VK_NULL_HANDLE;

    uint32_t numBufferBinds         = 0;
    uint32_t numDrawCalls           = 0;
    uint32_t numPushConstantUpdates = 0;
    uint32_t numPushConstantBytes   = 0;

    nvvk::DebugUtil::ScopedCmdLabel dbgLabel(cmd, "fillCmdBuffer");

    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_setup.container.getPipeLayout(), 0, 1,
                            m_setup.container.getSets(), 0, NULL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_setup.pipeline);

    for(size_t idx = 0; idx < drawCount; idx++)
    {
      const DrawItem&             di  = drawItems[idx];
      const CadSceneVK::Geometry& geo = scene.m_geometry[di.geometryIndex];
      assert(geo.ibo.offset % sizeof(uint32_t) == 0);
      uint32_t drawIndicesOffset = uint32_t(geo.ibo.offset + di.range.offset) / sizeof(uint32_t);
      uint32_t drawIndicesCount  = di.range.count;

      if(lastGeometry != di.geometryIndex)
      {
        if(geo.vbo.buffer != lastVbo)
        {
          lastVbo             = geo.vbo.buffer;
          VkDeviceSize offset = 0;
          vkCmdBindVertexBuffers(cmd, BINDING_PER_VERTEX, 1, &geo.vbo.buffer, &offset);
          ++numBufferBinds;
        }
        if(geo.ibo.buffer != lastIbo)
        {
          lastIbo = geo.ibo.buffer;
          vkCmdBindIndexBuffer(cmd, geo.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
          ++numBufferBinds;
        }

        lastGeometry = di.geometryIndex;

        if(m_mode == MODE_PER_TRI_ID_GS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(),
                             VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.trianglePartIdsAddr);
          numPushConstantBytes += sizeof(uint64_t);
          ++numPushConstantUpdates;
        }
        else if(m_mode == MODE_PER_TRI_BATCH_PART_SEARCH_GS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(),
                             VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.partTriCountsAddr);
          numPushConstantBytes += sizeof(uint64_t);
          ++numPushConstantUpdates;
        }
        else if(m_mode == MODE_PER_TRI_ID_FS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(),
                             VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.trianglePartIdsAddr);
          numPushConstantBytes += sizeof(uint64_t);
          ++numPushConstantUpdates;
        }
        else if(m_mode == MODE_PER_TRI_BATCH_PART_SEARCH_FS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(),
                             VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.partTriCountsAddr);
          numPushConstantBytes += sizeof(uint64_t);
          ++numPushConstantUpdates;
        }
        else if(m_mode == MODE_PER_TRI_GLOBAL_PART_SEARCH_FS)
        {
          vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(),
                             VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                             offsetof(DrawPushData, idsAddr), sizeof(uint64_t), &geo.partTriOffsetsAddr);
          numPushConstantBytes += sizeof(uint64_t);
          ++numPushConstantUpdates;
        }
      }

      if(lastMatrix != di.matrixIndex)
      {
        vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(),
                           VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           offsetof(DrawPushData, matrixIndex), sizeof(uint32_t), &di.matrixIndex);
        numPushConstantBytes += sizeof(uint32_t);
        ++numPushConstantUpdates;

        lastMatrix = di.matrixIndex;
      }

      int materialIndex = di.materialIndex;
      if(m_config.colorizeDraws)
      {
        materialIndex = int(idx);
      }

      if(lastMaterial != materialIndex)
      {
        vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(),
                           VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           offsetof(DrawPushData, materialIndex), sizeof(uint32_t), &materialIndex);
        numPushConstantBytes += sizeof(uint32_t);
        ++numPushConstantUpdates;

        lastMaterial = materialIndex;
      }

      if(di.objectOffset != lastUniqueOffset)
      {
        vkCmdPushConstants(cmd, m_setup.container.getPipeLayout(),
                           VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
                           offsetof(DrawPushData, uniquePartOffset), sizeof(uint32_t), &di.objectOffset);
        numPushConstantBytes += sizeof(uint32_t);
        ++numPushConstantUpdates;

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
          instanceIndex = uint32_t(di.range.offset) / sizeof(uint32_t) / 3;
          break;
        case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_GS:
        case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_FS:
        case RendererVK::MODE_PER_TRI_GLOBAL_PART_SEARCH_FS:
          // the partIndex is encoded per triangle in the idsBuffer
          // instanceIndex will encode the offset into the per-triangle idsBuffer, / 3 because 3 indices per triangle.
          assert(di.partIndex < (1 << 16));
          assert(di.partCount < (1 << 16));
          instanceIndex = uint32_t(di.partCount) | (uint32_t(di.partIndex) << 16);
          break;
        default:
          instanceIndex = 0;
          break;
      }
      assert(geo.vbo.offset % sizeof(CadScene::Vertex) == 0);
      vkCmdDrawIndexed(cmd, drawIndicesCount, 1, drawIndicesOffset, geo.vbo.offset / sizeof(CadScene::Vertex), instanceIndex);
      ++numDrawCalls;
    }

    LOGSTATS("buffer binds: %u, push constant updates: %u (%u byte), drawcalls: %u \n", numBufferBinds,
             numPushConstantUpdates, numPushConstantBytes, numDrawCalls);
  }

  void fillCmdBufferPerDrawBuffer(VkCommandBuffer cmd, const DrawItem* NV_RESTRICT drawItems, size_t drawCount)
  {
    const ResourcesVK* res   = m_resources;
    const CadSceneVK&  scene = res->m_scene;

    int      lastMaterial     = -1;
    int      lastGeometry     = -1;
    int      lastMatrix       = -1;
    uint32_t lastUniqueOffset = ~0;
    VkBuffer lastVbo          = VK_NULL_HANDLE;
    VkBuffer lastIbo          = VK_NULL_HANDLE;

    uint32_t numBufferBinds = 0;
    uint32_t numMDIDraws   = 0;
    VkDeviceSize mdiBufferOffset = 0;
    VkDeviceSize startMdiBufferOffset = 0;

    nvvk::DebugUtil::ScopedCmdLabel dbgLabel(cmd, "fillCmdBufferPerDrawBuffer");

    // Here we store per-Draw data into a buffer
    m_perDrawDataBuffer = m_resources->createBuffer(sizeof(DrawPushData) * drawCount,
                                                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    assert(m_perDrawDataBuffer.buffer);

    {
      VkDevice                          device = res->m_device;
      std::vector<VkWriteDescriptorSet> updateDescriptors;

      updateDescriptors.push_back(m_setup.container.makeWrite(0, DRAW_SSBO_PER_DRAW, &m_perDrawDataBuffer.info));

      vkUpdateDescriptorSets(device, updateDescriptors.size(), updateDescriptors.data(), 0, 0);
    }

    m_indirectDrawBuffer = m_resources->createBuffer(sizeof(VkDrawIndexedIndirectCommand) * drawCount,
                                                     VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    assert(m_indirectDrawBuffer.buffer);


    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_setup.container.getPipeLayout(), 0, 1,
                            m_setup.container.getSets(), 0, NULL);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_setup.pipeline);

    {
      VkDeviceSize offset = 0;
      vkCmdBindVertexBuffers(cmd, BINDING_PER_INSTANCE, 1, &m_perDrawIndexBuffer.buffer, &offset);
    }

    auto flushMDIDraws = [&]()
    {
        if (numMDIDraws)
        {
            vkCmdDrawIndexedIndirect(cmd, m_indirectDrawBuffer.buffer, startMdiBufferOffset, numMDIDraws, sizeof(VkDrawIndexedIndirectCommand));
            startMdiBufferOffset = mdiBufferOffset;
            numMDIDraws = 0;
        }
    };


    std::vector<DrawPushData> perDrawData(drawCount);
    std::vector<VkDrawIndexedIndirectCommand> indirectDraws;

    for(size_t idx = 0; idx < drawCount; idx++)
    {
      const DrawItem&             di       = drawItems[idx];
      const CadSceneVK::Geometry& geo      = scene.m_geometry[di.geometryIndex];
      DrawPushData&               drawData = perDrawData[idx];
      drawData.flexible                        = idx;

      assert(geo.ibo.offset % sizeof(uint32_t) == 0);
      uint32_t drawIndicesOffset = uint32_t(geo.ibo.offset + di.range.offset) / sizeof(uint32_t);
      uint32_t drawIndicesCount  = di.range.count;

      if(lastGeometry != di.geometryIndex)
      {
        flushMDIDraws();

        if(geo.vbo.buffer != lastVbo)
        {
          lastVbo             = geo.vbo.buffer;
          VkDeviceSize offset = 0;
          vkCmdBindVertexBuffers(cmd, BINDING_PER_VERTEX, 1, &geo.vbo.buffer, &offset);
          ++numBufferBinds;
        }
        if(geo.ibo.buffer != lastIbo)
        {
          lastIbo = geo.ibo.buffer;
          vkCmdBindIndexBuffer(cmd, geo.ibo.buffer, 0, VK_INDEX_TYPE_UINT32);
          ++numBufferBinds;
        }

        lastGeometry = di.geometryIndex;
      }
      if(m_mode == MODE_PER_TRI_ID_GS)
      {
        drawData.idsAddr = uint64_t(geo.trianglePartIdsAddr);
      }
      else if(m_mode == MODE_PER_TRI_BATCH_PART_SEARCH_GS)
      {
        drawData.idsAddr = uint64_t(geo.partTriCountsAddr);
      }
      else if(m_mode == MODE_PER_TRI_ID_FS)
      {
        drawData.idsAddr = uint64_t(geo.trianglePartIdsAddr);
      }
      else if(m_mode == MODE_PER_TRI_BATCH_PART_SEARCH_FS)
      {
        drawData.idsAddr = uint64_t(geo.partTriCountsAddr);
      }
      else if(m_mode == MODE_PER_TRI_GLOBAL_PART_SEARCH_FS)
      {
        drawData.idsAddr = uint64_t(geo.partTriOffsetsAddr);
      }

      drawData.matrixIndex = di.matrixIndex;

      int materialIndex = di.materialIndex;
      if(m_config.colorizeDraws)
      {
        materialIndex = int(idx);
      }
      drawData.materialIndex = materialIndex;

      drawData.uniquePartOffset = di.objectOffset;

      switch(m_mode)
      {
        case RendererVK::MODE_PER_DRAW_BASEINST:
          // directly pass the partIndex as instanceIndex
          drawData.flexible = di.partIndex;
          break;
        case RendererVK::MODE_PER_TRI_ID_GS:
        case RendererVK::MODE_PER_TRI_ID_FS:
          // the partIndex is encoded per triangle in the idsBuffer
          // instanceIndex will encode the offset into the per-triangle idsBuffer, / 3 because 3 indices per triangle.
          drawData.flexible = (uint32_t(di.range.offset) / sizeof(uint32_t)) / 3;
          break;
        case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_GS:
        case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_FS:
        case RendererVK::MODE_PER_TRI_GLOBAL_PART_SEARCH_FS:
          // the partIndex is encoded per triangle in the idsBuffer
          // instanceIndex will encode the offset into the per-triangle idsBuffer, / 3 because 3 indices per triangle.
          assert(di.partIndex < (1 << 16));
          assert(di.partCount < (1 << 16));
          drawData.flexible = uint32_t(di.partCount) | (uint32_t(di.partIndex) << 16);
          break;
        default:
          break;
      }


      assert(geo.vbo.offset % sizeof(CadScene::Vertex) == 0);
      uint vertexOffset = geo.vbo.offset / sizeof(CadScene::Vertex);

      {
          VkDrawIndexedIndirectCommand cmd{ drawIndicesCount , 1, drawIndicesOffset , vertexOffset, idx};
          indirectDraws.push_back(cmd);
          mdiBufferOffset += sizeof(VkDrawIndexedIndirectCommand);
      }

      ++numMDIDraws;
    }

    flushMDIDraws();

    // now that we know how many and in what order the drawcalls happen, set up the per-drawcall buffer
    {
        ResourcesVK* res = m_resources;
        ScopeStaging staging(res->m_allocator, res->m_queue, res->m_queueFamily);
        staging.upload({ m_perDrawDataBuffer.buffer, 0, drawCount * sizeof(DrawPushData) }, perDrawData.data());
        staging.upload({ m_indirectDrawBuffer.buffer, 0, drawCount * sizeof(VkDrawIndexedIndirectCommand) }, indirectDraws.data());
        staging.submit();
    }


    LOGSTATS("buffer binds: %u, drawcalls: %u \n", numBufferBinds, numMDIDraws);
  }


  void setupCmdBuffer(const DrawItem* NV_RESTRICT drawItems, size_t drawCount)
  { 
    ResourcesVK* res = m_resources;

    VkCommandBuffer cmd = res->createCmdBuffer(m_cmdPool, false, false, true);
    res->cmdDynamicState(cmd);

    if (m_config.perDrawParameterMode == Renderer::PER_DRAW_PUSHCONSTANTS)
    {
        fillCmdBuffer(cmd, drawItems, drawCount);
    }
    else
    {
        fillCmdBufferPerDrawBuffer(cmd, drawItems, drawCount);
    }

    vkEndCommandBuffer(cmd);
    m_draw.cmdBuffer = cmd;
  }

  void deleteCmdBuffer() { vkFreeCommandBuffers(m_resources->m_device, m_cmdPool, 1, &m_draw.cmdBuffer); }

  void setupPipeline(bool needsBaseInstanceBuffer)
  {
    ResourcesVK* res    = m_resources;
    VkDevice     device = res->m_device;

    vkDestroyPipeline(device, m_setup.pipeline, nullptr);

    {
      nvvk::GraphicsPipelineState     state = res->m_gfxState;
      nvvk::GraphicsPipelineGenerator gen(state);

      if (needsBaseInstanceBuffer)
      {
          state.addAttributeDescription(nvvk::GraphicsPipelineState::makeVertexInputAttribute(ATTRIB_BASEINSTANCE, BINDING_PER_INSTANCE,
              VK_FORMAT_R32_UINT, 0));
          state.addBindingDescription(nvvk::GraphicsPipelineState::makeVertexInputBinding(BINDING_PER_INSTANCE, sizeof(uint32_t),
              VK_VERTEX_INPUT_RATE_INSTANCE));
      }

      gen.setRenderPass(res->m_framebuffer.passPreserve);
      gen.setDevice(device);
      // pipelines
      gen.setLayout(m_setup.container.getPipeLayout());
      state.depthStencilState.depthCompareOp      = VK_COMPARE_OP_LESS_OR_EQUAL;
      state.rasterizationState.cullMode           = VK_CULL_MODE_BACK_BIT;
      state.multisampleState.rasterizationSamples = res->m_framebuffer.samplesUsed;

      gen.addShader(res->m_shaderManager.get(m_setup.vertexShader), VK_SHADER_STAGE_VERTEX_BIT);
      gen.addShader(res->m_shaderManager.get(m_setup.fragmentShader), VK_SHADER_STAGE_FRAGMENT_BIT);
      if(m_mode == RendererVK::MODE_PER_TRI_ID_GS || m_mode == RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_GS)
      {
        gen.addShader(res->m_shaderManager.get(m_setup.geometryShader), VK_SHADER_STAGE_GEOMETRY_BIT);
      }
      m_setup.pipeline = gen.createPipeline();
    }
  }
};


static RendererVK::TypeInstance         s_type_instance_vk;
static RendererVK::TypePrim             s_type_prim_vk;
static RendererVK::TypePrimSearch       s_type_prim_search_vk;
static RendererVK::TypeGlobalPrimSearch s_type_global_prim_search_vk;
static RendererVK::TypePrimGS           s_type_prim_gs_vk;
static RendererVK::TypePrimSearchGS     s_type_prim_search_gs_vk;

bool RendererVK::init(const CadScene* NV_RESTRICT scene, Resources* resources, const Config& config, Stats& stats)
{
  ResourcesVK* NV_RESTRICT res    = (ResourcesVK*)resources;
  VkDevice                 device = res->m_device;
  m_resources                     = res;
  m_scene                         = scene;
  m_config                        = config;

  {
    std::string prepend;
    prepend += nvh::stringFormat("#define IGNORE_MATERIALS %d\n", config.ignoreMaterials ? 1 : 0);
    prepend += nvh::stringFormat("#define COLORIZE_DRAWS %d\n", config.colorizeDraws ? 1 : 0);
    prepend += nvh::stringFormat("#define GLOBAL_GUESS %d\n", config.globalSearchGuess ? 1 : 0);
    prepend += nvh::stringFormat("#define GLOBAL_NARY_N %d\n", config.globalNaryN);
    prepend += nvh::stringFormat("#define GLOBAL_NARY_MIN %d\n", config.globalNaryMin);
    prepend += nvh::stringFormat("#define GLOBAL_NARY_ITERATIONS_MAX %d\n", config.globalNaryMaxIter);
    switch(config.perDrawParameterMode)
    {
      case Renderer::PER_DRAW_PUSHCONSTANTS:
        prepend += nvh::stringFormat("#define USE_PUSHCONSTANTS\n");
        break;
      case Renderer::PER_DRAW_INDEX_ATTRIBUTE:
        prepend += nvh::stringFormat("#define USE_ATTRIB_BASEINSTANCE\n");
        break;
    }

    // init shaders
    switch(m_mode)
    {
      case RendererVK::MODE_PER_DRAW_BASEINST:
        m_setup.fragmentShader =
            res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "drawid_instanceid.frag.glsl", prepend);
        m_setup.vertexShader =
            res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "drawid_instanceid.vert.glsl", prepend);
        break;
      case RendererVK::MODE_PER_TRI_ID_FS:
      case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_FS:
      case RendererVK::MODE_PER_TRI_GLOBAL_PART_SEARCH_FS:
        prepend += nvh::stringFormat("#define SEARCH_COUNT %d\n",
                                     m_mode == MODE_PER_TRI_BATCH_PART_SEARCH_FS ? config.searchBatch : 0);
        prepend += nvh::stringFormat("#define MODE_PER_TRI_GLOBAL_PART_SEARCH_FS %d\n",
                                     m_mode == MODE_PER_TRI_GLOBAL_PART_SEARCH_FS ? 1 : 0);
        m_setup.fragmentShader =
            res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "drawid_primid.frag.glsl", prepend);
        m_setup.vertexShader =
            res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "drawid_primid.vert.glsl", prepend);
        break;
      case RendererVK::MODE_PER_TRI_ID_GS:
      case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_GS:
        m_setup.fragmentShader =
            res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_FRAGMENT_BIT, "drawid_primid_gs.frag.glsl", prepend);
        m_setup.geometryShader = res->m_shaderManager.createShaderModule(
            VK_SHADER_STAGE_GEOMETRY_BIT, "drawid_primid_gs.geo.glsl",
            nvh::stringFormat("#define USE_GEOMETRY_SHADER_PASSTHROUGH %d\n", config.passthrough ? 1 : 0)
                + nvh::stringFormat("#define SEARCH_COUNT %d\n", m_mode == MODE_PER_TRI_BATCH_PART_SEARCH_GS ? config.searchBatch : 0)
                + prepend);
        m_setup.vertexShader =
            res->m_shaderManager.createShaderModule(VK_SHADER_STAGE_VERTEX_BIT, "drawid_primid_gs.vert.glsl", prepend);
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
    m_setup.container.addBinding(DRAW_SSBO_PER_DRAW, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                 VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT | VK_SHADER_STAGE_VERTEX_BIT);
    m_setup.container.initLayout();

    VkPushConstantRange ranges[3];
    uint32_t            rangeCount = 0;

    // Creates a VkPushConstantRange from a range of members in a struct
#define makePushConstantRange(stages, struct, memberFirst, memberLast)                                                                \
  VkPushConstantRange                                                                                                                 \
  {                                                                                                                                   \
    stages, offsetof(struct, memberFirst), offsetof(struct, memberLast) + sizeof(struct ::memberLast) - offsetof(struct, memberFirst) \
  }


    if(config.perDrawParameterMode == Renderer::PER_DRAW_PUSHCONSTANTS)
    {
#if 0
        rangeCount = 2;  // VS + FS ranges

      // Vertex shader push constants - just DrawPushData::matrixIndex
      ranges[0] = makePushConstantRange(VK_SHADER_STAGE_VERTEX_BIT, DrawPushData, matrixIndex, idsAddr);


      // Fragment shader push constants
      switch(m_mode)
      {
        case RendererVK::MODE_PER_DRAW_BASEINST:
        case RendererVK::MODE_PER_TRI_ID_GS:
        case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_GS:
          ranges[1] = makePushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, DrawPushData, matrixIndex, idsAddr);
          break;
        case RendererVK::MODE_PER_TRI_ID_FS:
        case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_FS:
        case RendererVK::MODE_PER_TRI_GLOBAL_PART_SEARCH_FS:
          ranges[1] = makePushConstantRange(VK_SHADER_STAGE_FRAGMENT_BIT, DrawPushData, matrixIndex, idsAddr);
          break;
        default:
          assert(0);
      }

      // Geometry shader variants only need DrawPushData::idsAddr and have their own range
      if(m_mode == RendererVK::MODE_PER_TRI_ID_GS || m_mode == RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_GS)
      {
        ranges[2]  = makePushConstantRange(VK_SHADER_STAGE_GEOMETRY_BIT, DrawPushData, matrixIndex, idsAddr);
        rangeCount = 3;
      }
#endif
      ranges[0] = makePushConstantRange(VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_GEOMETRY_BIT,
                                        DrawPushData, matrixIndex, idsAddr);
      rangeCount = 1;
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

    setupPipeline(m_config.perDrawParameterMode == Renderer::PER_DRAW_INDEX_ATTRIBUTE);
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
      case RendererVK::MODE_PER_TRI_GLOBAL_PART_SEARCH_FS:
        maxCombine = ~0;
        break;
      case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_GS:
      case RendererVK::MODE_PER_TRI_BATCH_PART_SEARCH_FS:
        maxCombine = config.searchBatch;
        break;
      default:
        maxCombine = 0;
        break;
    }

    fillDrawItems(m_drawItems, scene, config, maxCombine, stats);

    // now that we know how many and in what order the drawcalls happen, set up the per-drawcall buffer
    {
      ScopeStaging staging(res->m_allocator, res->m_queue, res->m_queueFamily);

      // This buffer will be essentially indexed by gl_BaseInstance and thus returns
      // gl_BaseInstance to the shader without accessing gl_BaseInstance explicitly
      std::vector<uint32_t> perDrawIndices(m_drawItems.size());
      for(size_t x = 0; x < perDrawIndices.size(); ++x)
      {
        perDrawIndices[x] = x;
      }
      
      m_perDrawIndexBuffer = m_resources->createBufferT(perDrawIndices.data(), perDrawIndices.size(),
                                                        VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                        staging.getCmd());
      staging.submit();
    }


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

  m_resources->destroy(m_perDrawDataBuffer);
  m_resources->destroy(m_indirectDrawBuffer);
  m_resources->destroy(m_perDrawIndexBuffer);
}

void RendererVK::draw(const Resources::Global& global, Stats& stats)
{
  ResourcesVK* NV_RESTRICT res = m_resources;

  if(m_draw.pipeChangeID != res->m_pipeChangeID || m_draw.fboChangeID != res->m_fboChangeID)
  {
    setupPipeline(m_config.perDrawParameterMode == Renderer::PER_DRAW_INDEX_ATTRIBUTE);
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
