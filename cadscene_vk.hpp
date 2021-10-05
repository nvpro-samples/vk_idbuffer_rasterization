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


#pragma once

#include "cadscene.hpp"
#include "resources_base.hpp"

// ScopeStaging handles uploads and other staging operations.
// not efficient because it blocks/syncs operations

struct ScopeStaging
{
  ScopeStaging(nvvk::ResourceAllocator& resAllocator_, VkQueue queue_, uint32_t queueFamily)
      : resAllocator(resAllocator_)
      , cmdPool(resAllocator_.getDevice(), queueFamily)
      , queue(queue_)
      , cmd(VK_NULL_HANDLE)
  {
  }
  nvvk::ResourceAllocator& resAllocator;
  VkCommandBuffer          cmd;
  nvvk::CommandPool        cmdPool;
  VkQueue                  queue;

  VkCommandBuffer getCmd()
  {
    cmd = cmd ? cmd : cmdPool.createCommandBuffer();
    return cmd;
  }

  void submit()
  {
    if(cmd)
    {
      cmdPool.submitAndWait(cmd, queue);
      cmd = VK_NULL_HANDLE;
      resAllocator.getStaging()->releaseResources();
    }
  }

  void upload(const VkDescriptorBufferInfo& binding, const void* data)
  {
    if(cmd && (data == nullptr || !resAllocator.getStaging()->fitsInAllocated(binding.range)))
    {
      submit();
    }
    if(data && binding.range)
    {
      resAllocator.getStaging()->cmdToBuffer(getCmd(), binding.buffer, binding.offset, binding.range, data);
    }
  }
};


// GeometryMemoryVK manages vbo/ibo etc. in chunks
// allows to reduce number of bindings and be more memory efficient

struct GeometryMemoryVK
{
  typedef size_t Index;


  struct Allocation
  {
    Index        chunkIndex;
    VkDeviceSize vboOffset;
    VkDeviceSize iboOffset;
    VkDeviceSize triangleIdsOffset;
    VkDeviceSize partIdsOffset;
  };

  struct Chunk
  {
    ResBuffer vbo;
    ResBuffer ibo;
    ResBuffer triangleIds;
    ResBuffer partIds;
  };

  nvvk::ResourceAllocator* m_resAllocator;
  std::vector<Chunk>       m_chunks;

  void init(nvvk::ResourceAllocator* resAllocator, VkDeviceSize maxChunk);
  void deinit();
  void alloc(VkDeviceSize vboSize, VkDeviceSize iboSize, VkDeviceSize triangleIdsSize, VkDeviceSize partIdsSize, Allocation& allocation);
  void finalize();

  const Chunk& getChunk(const Allocation& allocation) const { return m_chunks[allocation.chunkIndex]; }

  const Chunk& getChunk(Index index) const { return m_chunks[index]; }

  VkDeviceSize getVertexSize() const
  {
    VkDeviceSize size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].vbo.info.range;
    }
    return size;
  }

  VkDeviceSize getIndexSize() const
  {
    VkDeviceSize size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].ibo.info.range;
    }
    return size;
  }

  VkDeviceSize getIdSize() const
  {
    VkDeviceSize size = 0;
    for(size_t i = 0; i < m_chunks.size(); i++)
    {
      size += m_chunks[i].triangleIds.info.range;
      size += m_chunks[i].partIds.info.range;
    }
    return size;
  }

  VkDeviceSize getChunkCount() const { return m_chunks.size(); }

private:
  VkDeviceSize m_alignment;
  VkDeviceSize m_vboAlignment;
  VkDeviceSize m_maxVboChunk;
  VkDeviceSize m_maxIboChunk;
  VkDeviceSize m_maxIdsChunk;

  Index getActiveIndex() { return (m_chunks.size() - 1); }

  Chunk& getActiveChunk()
  {
    assert(!m_chunks.empty());
    return m_chunks[getActiveIndex()];
  }
};


class CadSceneVK
{
public:
  struct Geometry
  {
    GeometryMemoryVK::Allocation allocation;

    VkDescriptorBufferInfo vbo;
    VkDescriptorBufferInfo ibo;
    VkDescriptorBufferInfo triangleIds;
    VkDescriptorBufferInfo partIds;

    VkDeviceAddress vboAddr;
    VkDeviceAddress iboAddr;
    VkDeviceAddress triangleIdsAddr;
    VkDeviceAddress partIdsAddr;
  };

  struct Buffers
  {
    ResBuffer materials;
    ResBuffer matrices;
    ResBuffer matricesOrig;
  };

  nvvk::ResourceAllocator* m_resAllocator = nullptr;

  Buffers m_buffers;

  std::vector<Geometry> m_geometry;
  GeometryMemoryVK      m_geometryMem;

  void init(const CadScene& cadscene, nvvk::ResourceAllocator* resAllocator, VkQueue queue, uint32_t queueFamilyIndex);
  void deinit();
};
