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

// 1/0 toggle for a guess plus exponential search
#ifndef GLOBAL_GUESS
#error GLOBAL_GUESS not set
#endif

// N-way search (where binary search is a 2-way split)
#ifndef GLOBAL_NARY_N
#error GLOBAL_NARY_N not set
#endif

// Fall back to binary search when the remaining items is less than MIN
#ifndef GLOBAL_NARY_MIN
#error GLOBAL_NARY_MIN not set
#endif

#ifndef GLOBAL_NARY_ITERATIONS_MAX
#error GLOBAL_NARY_ITERATIONS_MAX not set
#endif

#if GLOBAL_NARY_N >= GLOBAL_NARY_MIN
#error GLOBAL_NARY_N must be less than GLOBAL_NARY_MIN
#endif


///////////////////////////////////////////////////////////
// Common

#include "common.h"
#include "per_draw_inputs.glsl"

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

///////////////////////////////////////////////////////////
// Input

layout(location=0) in Interpolants {
  vec3 wPos;
  vec3 wNormal;
} IN;

#ifdef USE_PUSHCONSTANTS
layout(location=2) in Id {
  flat uint idsOffset;
} IN_ID;
#endif

// we are using an atomic for the raytest, which means earlyZ would be skipped
// but that is not our intent
layout(early_fragment_tests) in;

///////////////////////////////////////////////////////////
// Output

layout(location=0,index=0) out vec4 out_Color;

///////////////////////////////////////////////////////////

#include "drawid_shading.glsl"

uint getIdsOffset()
{
#ifdef USE_PUSHCONSTANTS
    return IN_ID.idsOffset;
#else
    return perDrawData[getDrawId()].flexible;
#endif
}

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
    if (i < partCount && gl_PrimitiveID >= begin && gl_PrimitiveID < begin + partTriangleCount) {
      partIndex = i;
    }
    begin += partTriangleCount;
  }
  
  // the primitive id passed to the fragment shader contains the partIndex
  partIndex += int(partOffset);

#elif MODE_PER_TRI_GLOBAL_PART_SEARCH_FS == 1

  // Find which geometry "part" this triangle, gl_PrimitiveID, belongs to. I.e.
  // find the index in partTriOffsets that bounds the current triangle index.
  //
  // Example:
  // triangles per part:  [150,  10, 290, 123]
  // partTriOffsets:      [  0, 150, 160, 450] <- PUSH.idsAddr
  // partOffset:          2
  // partCount:           2
  // gl_PrimitiveID:      293
  // This shader needs to find which of the 4 parts triangle 293 belongs to.
  // We are drawing part 2 and 3 combined, so gl_PrimitiveID begins at 160.
  // 160 + 293 = 453, which is in the 4th part, or part 3.

  // The "part" range for this draw call is  encoded in the instance ID
  uint partOffset = getIdsOffset() >> 16;
  uint partCount  = getIdsOffset() & 0xFFFF;

  // PUSH.idsAddr points to partTriOffsets for
  // MODE_PER_TRI_GLOBAL_PART_SEARCH_FS. This holds per-part triangle range
  // start, i.e. first triangle index for each part.
  uints_in partTriOffsets = getIdsAddress();

  // Each draw call draws a range of parts. The first triangle index can be
  // found in partTriOffsets.
  int triangleOffset = int(partTriOffsets.d[partOffset]);

  // Part ID is one less than the upper bound, found with a binary search
  int l = int(partOffset);
  int h = int(partOffset) + int(partCount);
  int value = triangleOffset + gl_PrimitiveID;

  // Begin with a guess based on interpolation, then continue with "exponential
  // search".
#if GLOBAL_GUESS
  // TODO: add a final element to partTriOffsets for the total triangle count.
  // Otherwise we can't read the last part size.
  int trianglesSecondLast = int(partTriOffsets.d[partOffset + partCount - 1]);
  float trianglesPerPart = float(trianglesSecondLast - triangleOffset) / max(1.0, float(partCount - 1));
  int guess = clamp(int(value / trianglesPerPart), l, h - 1);
  bool guessIsLow = partTriOffsets.d[guess] <= value;
  int bound = 1;
  // TODO: a tad ugly and could have a few off-by-one errors
  for (int canary = 0; canary < 32; ++canary) {
    int next = guessIsLow ? guess + bound : guess - bound;
    bound *= 2;
    if(next < l || next >= h)
    {
      if(guessIsLow)
      {
        l = guess;
      }
      else
      {
        h = guess + 1;
      }
      break;
    }
    bool nextIsLow = partTriOffsets.d[next] <= value;
    if(guessIsLow != nextIsLow)
    {
      if(guessIsLow)
      {
        l = guess;
        h = next + 1;
      }
      else
      {
        h = guess + 1;
        l = next;
      }
      break;
    }
  }
#endif

  // "N-ary search" for objects with large part counts. Like a binary search,
  // but splitting into more than 2. This is intended to avoid conditional
  // reads.
#if GLOBAL_NARY_ITERATIONS_MAX > 0
  const int steps = GLOBAL_NARY_N;
  const int largePartCount = GLOBAL_NARY_MIN;
  for (int canary = 0; canary < GLOBAL_NARY_ITERATIONS_MAX && l + largePartCount < h; ++canary) {
    // 'steps - 1' to guarantee the whole range is covered. An alternative is a
    // ceiling divide.
    int step = (h - l) / (steps - 1);
    int begin = l;
    [[unroll]]
    for(int i = 0; i < steps; ++i)
    {
      int end = begin + step;
      if(end < h && partTriOffsets.d[end] < value)
      {
        l = end;
      }
      begin = end;
    }
    h = min(h, l + step);
  }
#endif

  // Standard upper bound binary search
  for (int canary = 0; canary < 32 && l < h; ++canary) {
    int mid =  l + (h - l) / 2;
    if (value < partTriOffsets.d[mid]) {
      h = mid;
    } else {
      l = mid + 1;
    }
  }
  int partIndex = l - 1;

#else

  // PUSH.idsAddr points to trianglePartIds for MODE_PER_TRI_ID_FS. This holds
  // per-triangle part IDs, which can be memory/bandwidth intensive.
  uints_in triangleIDs = getIdsAddress();

  // lookup each triangle's partId
  // the address of the buffer contains the partIndex per triangle
  int partIndex = int(triangleIDs.d[gl_PrimitiveID + int(getIdsOffset())]);
  
#endif
  //partIndex = gl_PrimitiveID - partIndex;

  out_Color = shading(uint(partIndex));
}
