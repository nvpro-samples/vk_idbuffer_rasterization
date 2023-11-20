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

// we are using an atomic for the raytest, which means earlyZ would be skipped
// but that is not our intent
layout(early_fragment_tests) in;

///////////////////////////////////////////////////////////
// Output

layout(location=0,index=0) out vec4 out_Color;

///////////////////////////////////////////////////////////

#include "drawid_shading.glsl"

void main()
{
  out_Color = shading(uint(gl_PrimitiveID));
}
