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
#include "per_draw_inputs.glsl"

///////////////////////////////////////////////////////////
// Bindings

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

  #ifdef USE_PUSHCONSTANTS
  layout(location=2) in Inputs2 {
    flat uint idsOffset;
  } IN_ID[];
  #endif
  
#else

  layout(triangle_strip) out;
  layout(max_vertices=3) out;

  layout(location=0) in Inputs {
    vec3 wPos;
    vec3 wNormal;
  } PASSTHROUGH[];
  
  #ifdef USE_PUSHCONSTANTS  
  layout(location=2) in InputsID {
    flat uint idsOffset;
  } IN_ID[];
  #endif

  layout(location=0) out Outputs {
    vec3 wPos;
    vec3 wNormal;
  } OUT;

#endif

uint getIdsOffset()
{
#ifdef USE_PUSHCONSTANTS
  return IN_ID[0].idsOffset;
#else
  return perDrawData[getDrawId()].flexible;
#endif
}

///////////////////////////////////////////////////////////

void main()
{
#if SEARCH_COUNT
  // find which partIndex we are based on the gl_PrimitiveIDIn which spans
  // multiple parts.

  // PUSH.idsAddr points to partTriCounts for MODE_PER_TRI_BATCH_PART_SEARCH_FS. This
  // holds per-part triangle count.
  uints_in partTriCounts = getIdsAddress();

  // The "part" range for this draw call is  encoded in the instance ID
  uint partOffset = getIdsOffset() >> 16;
  uint partCount  = getIdsOffset() & 0xFFFF;

  int begin = 0;
  int partIndex = 0;
  
  // unroll this loop so compiler has ability to batch loads

  [[unroll]]
  for (int i = 0; i < SEARCH_COUNT; i++)
  {
    // don't make the load part of any condition that isn't hardcoded.
    int partTriangleCount = int(partTriCounts.d[partOffset + i]);
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

  // PUSH.idsAddr points to trianglePartIds for MODE_PER_TRI_ID_GS. This holds
  // per-triangle part IDs, which can be memory/bandwidth intensive.
  uints_in triangleIDs = getIdsAddress();

  // lookup each triangle's partId
  // the address of the buffer contains the partIndex per triangle
  gl_PrimitiveID = int(triangleIDs.d[gl_PrimitiveIDIn + int(getIdsOffset())]);
  
#endif

#if !USE_GEOMETRY_SHADER_PASSTHROUGH
#ifndef USE_PUSHCONSTANTS
  OUT_DRAWID.drawId = getDrawId();
#endif
  [[unroll]]
  for (int i = 0; i < 3; i++) {
    gl_Position = gl_in[i].gl_Position;
    OUT.wPos = PASSTHROUGH[i].wPos;
    OUT.wNormal = PASSTHROUGH[i].wNormal;
    EmitVertex();
  }
#endif
}
