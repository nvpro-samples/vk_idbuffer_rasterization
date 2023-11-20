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

#include "common.h"
#include "per_draw_inputs.glsl"

///////////////////////////////////////////////////////////
// Bindings

layout(set=0, binding=DRAW_UBO_SCENE, scalar) uniform sceneBuffer {
  SceneData   scene;
};

layout(set=0, binding=DRAW_SSBO_MATRIX, scalar) buffer matrixBuffer {
  MatrixData    matrices[];
};


///////////////////////////////////////////////////////////
// Input

in layout(location = ATTRIB_VERTEX_POS_OCTNORMAL) vec4 inPosNormal;


///////////////////////////////////////////////////////////
// Output

layout(location=0) out Interpolants {
  vec3 wPos;
  vec3 wNormal;
} OUT;

#ifdef USE_PUSHCONSTANTS  
layout(location=2) out Id {
  flat uint idsOffset;
} OUT_ID;
#endif

///////////////////////////////////////////////////////////

// oct functions from http://jcgt.org/published/0003/02/01/paper.pdf
vec2 oct_signNotZero(vec2 v) {
  return vec2((v.x >= 0.0) ? +1.0 : -1.0, (v.y >= 0.0) ? +1.0 : -1.0);
}
vec3 oct_to_float32x3(vec2 e) {
  vec3 v = vec3(e.xy, 1.0 - abs(e.x) - abs(e.y));
  if (v.z < 0) v.xy = (1.0 - abs(v.yx)) * oct_signNotZero(v.xy);
  return normalize(v);
}
vec2 float32x3_to_oct(in vec3 v) {
  // Project the sphere onto the octahedron, and then onto the xy plane
  vec2 p = v.xy * (1.0 / (abs(v.x) + abs(v.y) + abs(v.z)));
  // Reflect the folds of the lower hemisphere over the diagonals
  return (v.z <= 0.0) ? ((1.0 - abs(p.yx)) * oct_signNotZero(p)) : p;
}

void main()
{
  vec3 inNormal = oct_to_float32x3(unpackSnorm2x16(floatBitsToUint(inPosNormal.w)));
  
  MatrixData matrix = matrices[getMatrixIndex()];

  vec3 wPos     = (matrix.worldMatrix   * vec4(inPosNormal.xyz,1)).xyz;
  vec3 wNormal  = mat3(matrix.worldMatrixIT) * inNormal;

  gl_Position   = scene.viewProjMatrix * vec4(wPos,1);  
  OUT.wPos      = wPos;
  OUT.wNormal   = wNormal;
  
#ifdef USE_PUSHCONSTANTS
  OUT_ID.idsOffset = gl_InstanceIndex;
#else
  OUT_DRAWID.drawId = getDrawId();
#endif

  
}
