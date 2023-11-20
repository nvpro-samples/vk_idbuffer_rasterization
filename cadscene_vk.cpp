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


#include "cadscene_vk.hpp"

#include <algorithm>
#include <inttypes.h>
#include <nvh/nvprint.hpp>


static inline VkDeviceSize alignedSize(VkDeviceSize sz, VkDeviceSize align)
{
  return ((sz + align - 1) / (align)) * align;
}


void GeometryMemoryVK::init(nvvk::ResourceAllocator* resAllocator, VkDeviceSize maxChunk)
{
  m_resAllocator = resAllocator;
  m_alignment    = 16;
  m_vboAlignment = 16;

  m_maxVboChunk =        maxChunk;
  m_maxIboChunk =        maxChunk;
  m_maxIdsChunk =        maxChunk;
  m_maxTriOffsetsChunk = maxChunk;
}

void GeometryMemoryVK::deinit()
{
  for(size_t i = 0; i < m_chunks.size(); i++)
  {
    Chunk& chunk = m_chunks[i];
    destroyResBuffer(*m_resAllocator, chunk.vbo);
    destroyResBuffer(*m_resAllocator, chunk.ibo);
    destroyResBuffer(*m_resAllocator, chunk.trianglePartIds);
    destroyResBuffer(*m_resAllocator, chunk.partTriCounts);
    destroyResBuffer(*m_resAllocator, chunk.partTriOffsets);
  }
  m_chunks       = std::vector<Chunk>();
  m_resAllocator = nullptr;
}

void GeometryMemoryVK::alloc(VkDeviceSize vboSize,
                             VkDeviceSize iboSize,
                             VkDeviceSize trianglePartIdsSize,
                             VkDeviceSize partTriCountsSize,
                             VkDeviceSize triOffsetsSize,
                             Allocation&  allocation)
{
  vboSize             = alignedSize(vboSize, m_vboAlignment);
  iboSize             = alignedSize(iboSize, m_alignment);
  trianglePartIdsSize = alignedSize(trianglePartIdsSize, m_alignment);
  partTriCountsSize   = alignedSize(partTriCountsSize, m_alignment);
  triOffsetsSize      = alignedSize(triOffsetsSize, m_alignment);

  if(m_chunks.empty() || getActiveChunk().vbo.info.range + vboSize > m_maxVboChunk || getActiveChunk().ibo.info.range + iboSize > m_maxIboChunk
     || getActiveChunk().trianglePartIds.info.range + trianglePartIdsSize > m_maxIdsChunk
     || getActiveChunk().partTriCounts.info.range + partTriCountsSize > m_maxIdsChunk
     || getActiveChunk().partTriOffsets.info.range + triOffsetsSize > m_maxTriOffsetsChunk)
  {
    finalize();
    Chunk chunk = {};
    m_chunks.push_back(chunk);
  }

  Chunk& chunk = getActiveChunk();

  allocation.chunkIndex            = getActiveIndex();
  allocation.vboOffset             = chunk.vbo.info.range;
  allocation.iboOffset             = chunk.ibo.info.range;
  allocation.trianglePartIdsOffset = chunk.trianglePartIds.info.range;
  allocation.partTriCountsOffset   = chunk.partTriCounts.info.range;
  allocation.triOffsetsOffset      = chunk.partTriOffsets.info.range;

  chunk.vbo.info.range += vboSize;
  chunk.ibo.info.range += iboSize;
  chunk.trianglePartIds.info.range += trianglePartIdsSize;
  chunk.partTriCounts.info.range += partTriCountsSize;
  chunk.partTriOffsets.info.range += triOffsetsSize;
}

void GeometryMemoryVK::finalize()
{
  if(m_chunks.empty())
  {
    return;
  }

  Chunk& chunk = getActiveChunk();

  chunk.vbo = createResBuffer(*m_resAllocator, chunk.vbo.info.range, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
  chunk.ibo = createResBuffer(*m_resAllocator, chunk.ibo.info.range, VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
  chunk.trianglePartIds = createResBuffer(*m_resAllocator, chunk.trianglePartIds.info.range, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  // pad to allow graceful out-of-bounds access
  chunk.partTriCounts = createResBuffer(*m_resAllocator, chunk.partTriCounts.info.range + sizeof(uint32_t) * 32,
                                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
  chunk.partTriOffsets = createResBuffer(*m_resAllocator, chunk.partTriOffsets.info.range, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

void CadSceneVK::init(const CadScene& cadscene, nvvk::ResourceAllocator* resAllocator, VkQueue queue, uint32_t queueFamilyIndex)
{
  VkDeviceSize MB = 1024 * 1024;

  m_resAllocator = resAllocator;
  m_geometry.resize(cadscene.m_geometry.size(), {0});

  if(m_geometry.empty())
    return;

  {
    // allocation phase
    m_geometryMem.init(m_resAllocator, 256 * MB);

    for(size_t g = 0; g < cadscene.m_geometry.size(); g++)
    {
      const CadScene::Geometry& cadgeom = cadscene.m_geometry[g];
      Geometry&                 geom    = m_geometry[g];

      m_geometryMem.alloc(cadgeom.vboSize, cadgeom.iboSize, cadgeom.trianglePartIdsSize, cadgeom.partTriCountsSize,
                          cadgeom.partTriOffsetsSize, geom.allocation);
    }

    m_geometryMem.finalize();

    LOGI("Size of vertex data: %11" PRId64 "\n", uint64_t(m_geometryMem.getVertexSize()));
    LOGI("Size of index data:  %11" PRId64 "\n", uint64_t(m_geometryMem.getIndexSize()));
    LOGI("Size of ids data:    %11" PRId64 "\n", uint64_t(m_geometryMem.getIdSize()));
    LOGI("Size of data:        %11" PRId64 "\n",
         uint64_t(m_geometryMem.getVertexSize() + m_geometryMem.getIndexSize() + m_geometryMem.getIdSize()));
    LOGI("Chunks:              %11d\n", uint32_t(m_geometryMem.getChunkCount()));
  }

  ScopeStaging staging(*m_resAllocator, queue, queueFamilyIndex);

  for(size_t g = 0; g < cadscene.m_geometry.size(); g++)
  {
    const CadScene::Geometry&      cadgeom = cadscene.m_geometry[g];
    Geometry&                      geom    = m_geometry[g];
    const GeometryMemoryVK::Chunk& chunk   = m_geometryMem.getChunk(geom.allocation);

    // upload and assignment phase
    geom.vbo.buffer = chunk.vbo.buffer;
    geom.vbo.offset = geom.allocation.vboOffset;
    geom.vbo.range  = cadgeom.vboSize;
    geom.vboAddr    = chunk.vbo.addr + geom.allocation.vboOffset;
    staging.upload(geom.vbo, cadgeom.vboData);

    geom.ibo.buffer = chunk.ibo.buffer;
    geom.ibo.offset = geom.allocation.iboOffset;
    geom.ibo.range  = cadgeom.iboSize;
    geom.iboAddr    = chunk.ibo.addr + geom.allocation.iboOffset;
    staging.upload(geom.ibo, cadgeom.iboData);

    geom.trianglePartIds.buffer = chunk.trianglePartIds.buffer;
    geom.trianglePartIds.offset = geom.allocation.trianglePartIdsOffset;
    geom.trianglePartIds.range  = cadgeom.trianglePartIdsSize;
    geom.trianglePartIdsAddr    = chunk.trianglePartIds.addr + geom.allocation.trianglePartIdsOffset;
    staging.upload(geom.trianglePartIds, cadgeom.trianglePartIdsData);

    geom.partTriCounts.buffer = chunk.partTriCounts.buffer;
    geom.partTriCounts.offset = geom.allocation.partTriCountsOffset;
    geom.partTriCounts.range  = cadgeom.partTriCountsSize;
    geom.partTriCountsAddr    = chunk.partTriCounts.addr + geom.allocation.partTriCountsOffset;
    staging.upload(geom.partTriCounts, cadgeom.partTriCountsData);

    geom.partTriOffsets.buffer = chunk.partTriOffsets.buffer;
    geom.partTriOffsets.offset = geom.allocation.triOffsetsOffset;
    geom.partTriOffsets.range  = cadgeom.partTriOffsetsSize;
    geom.partTriOffsetsAddr    = chunk.partTriOffsets.addr + geom.allocation.triOffsetsOffset;
    staging.upload(geom.partTriOffsets, cadgeom.partTriOffsetsData);
  }

  VkDeviceSize materialsSize = cadscene.m_materials.size() * sizeof(CadScene::Material);
  VkDeviceSize matricesSize  = cadscene.m_matrices.size() * sizeof(CadScene::MatrixNode);

  m_buffers.materials =
      createResBuffer(*resAllocator, materialsSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  m_buffers.matrices =
      createResBuffer(*resAllocator, matricesSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
  m_buffers.matricesOrig =
      createResBuffer(*resAllocator, matricesSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);


  staging.upload(m_buffers.materials.info, cadscene.m_materials.data());
  staging.upload(m_buffers.matrices.info, cadscene.m_matrices.data());
  staging.upload(m_buffers.matricesOrig.info, cadscene.m_matrices.data());

  staging.submit();
}

void CadSceneVK::deinit()
{
  destroyResBuffer(*m_resAllocator, m_buffers.materials);
  destroyResBuffer(*m_resAllocator, m_buffers.matrices);
  destroyResBuffer(*m_resAllocator, m_buffers.matricesOrig);

  m_geometry.clear();
  m_geometryMem.deinit();
}
