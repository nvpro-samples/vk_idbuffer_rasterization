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

vec4 shading(uint partIndex)
{
  // partIndex is a running index for each geometry.
  // To make it fully unique over all objects in the scene
  // we need to apply this offset
  // (otherwise every first part of any geometry will have the same index)
  partIndex += PUSH.uniquePartOffset;

#if COLORIZE_DRAWS
  MaterialSide side;
  side.diffuse = unpackUnorm4x8(murmurHash(PUSH.materialIdx));
  side.diffuse = (vec4(dot(side.diffuse, vec4(1,1,1,0)) / 3.0) * 1.2  + side.diffuse) / 1.7f;
  side.ambient = vec4(0.05);
  side.emissive = vec4(0);
  side.specular = vec4(0.5);
#else
  MaterialSide side = materials[PUSH.materialIdx].sides[gl_FrontFacing ? 1 : 0];
#endif

  vec4 color = side.ambient + side.emissive;
  vec3 eyePos = vec3(scene.viewMatrixIT[0].w,scene.viewMatrixIT[1].w,scene.viewMatrixIT[2].w);

  vec3 lightDir = normalize( scene.wLightPos.xyz - IN.wPos);
  vec3 viewDir  = normalize( eyePos - IN.wPos);
  vec3 halfDir  = normalize(lightDir + viewDir);
  vec3 normal   = normalize(IN.wNormal) * (gl_FrontFacing ? 1 : -1);

  float ldot = dot(normal,lightDir);
  normal *= sign(ldot);
  ldot   *= sign(ldot);

  // (sin(scene.time * 2) * 0.5 + 0.5) * 0.7 + 0.1
  color += mix(side.diffuse, unpackUnorm4x8(murmurHash(partIndex)), scene.partWeight) * ldot;
  color += side.specular * pow(max(0,dot(normal,halfDir)),16);
  
  
  // simple ray selection highlight:
  
  // if this fragment coordinate matches the mouse cursor
  // we do a 64-bit atomicMin to find the closest surface (lowest depth value)
  // and we store the unique partIndex 
  if (all(equal(ivec2(gl_FragCoord.xy), scene.mousePos))) 
  {
    // pack partIndex in lower  32-bit
    //      depth     in higher 32-bit
    atomicMin(ray.mouseHit, packUint2x32(uvec2(partIndex, floatBitsToUint(gl_FragCoord.z))) );
  }
  
  // rayLast is the result of the above logic from last frame.
  // We cannot use the same frame's result, because as we raster the various triangles
  // the result will change.
  // If the current partIndex matches the one that was the closest in the last
  // frame, then alter the color for the selection highlight.
  // The copying of the result is done after rendering
  // (see the vkCmdCopyBuffer at end of RendererVK::draw)
  if (partIndex == unpackUint2x32(rayLast.mouseHit).x)
  {
    color = mix(color, vec4(1) - color, sin(scene.time * 10) * 0.5 + 0.5);
  }
  
  return color;
}