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
#extension GL_EXT_control_flow_attributes : enable

#ifndef USE_GEOMETRY_SHADER_PASSTHROUGH
#define USE_GEOMETRY_SHADER_PASSTHROUGH 1
#endif

#ifndef SEARCH_COUNT
#define SEARCH_COUNT 8
#endif

#if USE_GEOMETRY_SHADER_PASSTHROUGH
#extension GL_NV_geometry_shader_passthrough : require
#endif

#include "common.h"

///////////////////////////////////////////////////////////
// Bindings

layout(buffer_reference, buffer_reference_align=4) buffer readonly uints_in {
  uint d[];
};

layout(push_constant, scalar) uniform pushConstants 
{
  // This buffer address is different depending on the renderer:
  // USE_SEARCH == 0: array of a per-triangle partIndex
  // USE_SEARCH != 0: array of the number of triangles per part

  layout(offset=16) uints_in partIds;
} PUSH;

///////////////////////////////////////////////////////////
// Input/Output

layout(triangles) in;

#if USE_GEOMETRY_SHADER_PASSTHROUGH

  // Declare "Inputs" with "passthrough" to automatically copy members.
  layout(passthrough) in gl_PerVertex {
    vec4 gl_Position;
  } gl_in[];

  layout(passthrough,location=0) in Inputs {
    vec3 wPos;
    vec3 wNormal;
  } PASSTHROUGH[];

  layout(location=2) in Inputs2 {
    flat uint idsOffset;
  } IN_ID[];
  
#else

  layout(triangle_strip) out;
  layout(max_vertices=3) out;

  layout(location=0) in Inputs {
    vec3 wPos;
    vec3 wNormal;
  } PASSTHROUGH[];
  
  layout(location=2) in InputsID {
    flat uint idsOffset;
  } IN_ID[];

  layout(location=0) out Outputs {
    vec3 wPos;
    vec3 wNormal;
  } OUT;
  
#endif

///////////////////////////////////////////////////////////

void main()
{
#if SEARCH_COUNT
  // find which partIndex we are based on the gl_PrimitiveIDIn which spans
  // multiple parts.
 

  uint partOffset = IN_ID[0].idsOffset >> 8;
  uint partCount  = IN_ID[0].idsOffset & 0xFF;
  
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
    if (i < partCount && gl_PrimitiveIDIn >= begin && gl_PrimitiveIDIn < begin + partTriangleCount) {
      partIndex = i;
    }
    begin += partTriangleCount;
  }
  
  // the primitive id passed to the fragment shader contains the partIndex
  gl_PrimitiveID = int(partOffset) + partIndex;
  
#else

  // lookup each triangle's partId
  // the address of the buffer contains the partIndex per triangle
  gl_PrimitiveID = int(PUSH.partIds.d[gl_PrimitiveIDIn + int(IN_ID[0].idsOffset)]);
  
#endif

#if !USE_GEOMETRY_SHADER_PASSTHROUGH
  [[unroll]]
  for (int i = 0; i < 3; i++) {
    gl_Position = gl_in[i].gl_Position;
    OUT.wPos = PASSTHROUGH[i].wPos;
    OUT.wNormal = PASSTHROUGH[i].wNormal;
    EmitVertex();
  }
#endif
}
