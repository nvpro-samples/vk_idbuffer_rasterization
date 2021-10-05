/*
* Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
*
* NVIDIA CORPORATION and its licensors retain all intellectual property
* and proprietary rights in and to this software, related documentation
* and any modifications thereto.  Any use, reproduction, disclosure or
* distribution of this software and related documentation without an express
* license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#pragma once

#include <algorithm>
#include <platform.h>
#include <nvh/nvprint.hpp>
#include <nvvk/buffers_vk.hpp>
#include <nvvk/commands_vk.hpp>
#include <nvvk/debug_util_vk.hpp>
#include <nvvk/descriptorsets_vk.hpp>
#include <nvvk/memorymanagement_vk.hpp>
#include <nvvk/pipeline_vk.hpp>
#include <nvvk/shadermodulemanager_vk.hpp>
#include <nvvk/resourceallocator_vk.hpp>

namespace idraster {
class Resources;
class ResourcesVK;
}  // namespace idraster

struct ResBuffer : nvvk::Buffer
{
  VkDescriptorBufferInfo info = {VK_NULL_HANDLE, 0, 0};
  VkDeviceAddress        addr = 0;
};

inline ResBuffer createResBuffer(nvvk::ResourceAllocator& resAllocator,
                                 VkDeviceSize             size,
                                 VkBufferUsageFlags       flags,
                                 VkMemoryPropertyFlags    memFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
{
  ResBuffer entry = {nullptr};

  if(size)
  {
    ((nvvk::Buffer&)entry) =
        resAllocator.createBuffer(size, flags | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT, memFlags);
    entry.info.buffer = entry.buffer;
    entry.info.offset = 0;
    entry.info.range  = size;
    entry.addr        = nvvk::getBufferDeviceAddress(resAllocator.getDevice(), entry.buffer);
  }

  return entry;
}


template <typename T>
inline ResBuffer createResBufferT(nvvk::ResourceAllocator& resAllocator, const T* obj, size_t count, VkBufferUsageFlags flags, VkCommandBuffer cmd = VK_NULL_HANDLE)
{
  ResBuffer entry = createResBuffer(resAllocator, sizeof(T) * count, flags, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
  if(cmd)
  {
    resAllocator.getStaging()->cmdToBuffer(cmd, entry.buffer, entry.info.offset, entry.info.range, obj);
  }

  return entry;
}

inline void destroyResBuffer(nvvk::ResourceAllocator& resAllocator, ResBuffer& obj)
{
  resAllocator.destroy(obj);
  obj.info = {nullptr};
  obj.addr = 0;
}
