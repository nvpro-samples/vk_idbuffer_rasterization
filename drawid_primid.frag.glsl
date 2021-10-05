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


#version 460 core
/**/

#extension GL_GOOGLE_include_directive : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable
#extension GL_EXT_shader_atomic_int64 : enable
#extension GL_EXT_control_flow_attributes : enable

#ifndef SEARCH_COUNT
#define SEARCH_COUNT 8
#endif

#include "common.h"

///////////////////////////////////////////////////////////
// Bindings

layout(set=0, binding=DRAW_UBO_SCENE, scalar) uniform sceneBuffer {
  SceneData   scene;
  RayData     rayLast;
};

layout(set=0, binding=DRAW_SSBO_MATERIAL, scalar) buffer materialBuffer {
  MaterialData    materials[];
};

layout(set=0, binding=DRAW_SSBO_RAY, scalar) buffer coherent rayBuffer {
  RayData   ray;
};

layout(buffer_reference, buffer_reference_align=4) buffer readonly uints_in {
  uint d[];
};

layout(push_constant, scalar) uniform pushConstants {
  layout(offset=8)
  uint     materialIdx;
  uint     uniquePartOffset;
  uints_in partIds;
} PUSH;

///////////////////////////////////////////////////////////
// Input

layout(location=0) in Interpolants {
  vec3 wPos;
  vec3 wNormal;
} IN;

layout(location=2) in Id {
  flat uint idsOffset;
} IN_ID;


///////////////////////////////////////////////////////////
// Output

layout(location=0,index=0) out vec4 out_Color;

///////////////////////////////////////////////////////////

#include "drawid_shading.glsl"

void main()
{
#if SEARCH_COUNT
  // find which partIndex we are based on the gl_PrimitiveIDIn which spans
  // multiple parts.

  uint partOffset = IN_ID.idsOffset >> 8;
  uint partCount  = IN_ID.idsOffset & 0xFF;
  
  int begin = 0;
  int partIndex = 0;
  
  // unroll this loop so compiler has ability to batch loads

  [[unroll]]
  for (int i = 0; i < SEARCH_COUNT; i++)
  {
    // don't make the load part of any condition that isn't hardcoded.
    int partTriangleCount = int(PUSH.partIds.d[partOffset + i]);
    // We unroll the full loop and test conditionally within, rather than using
    // a them the dynamic partCount in the loop condition, to optimize the code
    // generation again. We optimize for the useage that most drawcalls will max
    // out the SEARCH_COUNT.
    [[flatten]]
    if (i < partCount && gl_PrimitiveID >= begin && gl_PrimitiveID < begin + partTriangleCount) {
      partIndex = i;
    }
    begin += partTriangleCount;
  }
  
  // the primitive id passed to the fragment shader contains the partIndex
  partIndex += int(partOffset);
  
#else

  // lookup each triangle's partId
  // the address of the buffer contains the partIndex per triangle
  int partIndex = int(PUSH.partIds.d[gl_PrimitiveID + int(IN_ID.idsOffset)]);
  
#endif

  out_Color = shading(uint(partIndex));
}
