/*
* Copyright (c) 2023-2025, NVIDIA CORPORATION.  All rights reserved.
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
* SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION
* SPDX-License-Identifier: Apache-2.0
*/

#ifndef USE_PUSHCONSTANTS
// When not using pish constants, we use a single large SSBO to provide the
// per-draw data, indexed by the only per-draw parameter that we can specify with
// each draw call, the 'firstInstance' parameter
layout(set = 0, binding = DRAW_SSBO_PER_DRAW, scalar) buffer perDrawBuffer
{
  DrawPushData perDrawData[];
};
#else
layout(push_constant, scalar) uniform pushConstants
{
  DrawPushData PUSH;
};
#endif // USE_PUSHCONSTANTS

#ifdef _VERTEX_SHADER_
#ifdef USE_ATTRIB_BASEINSTANCE
// Use an intanced vertex attribute to 'emulate' gl_BaseInstance
in   layout(location = ATTRIB_BASEINSTANCE) uint InBaseInstance;
uint getBaseInstance()
{
  return InBaseInstance;
}
#else
// We use  gl_InstanceIndex here instead of gl_BaseInstance as in
// the non-instanced case, gl_InstanceIndex == gl_BaseIndex
uint getBaseInstance()
{
  return gl_InstanceIndex;
}
#endif
#endif // VERTEX_SHADER

#ifdef _VERTEX_SHADER_
#ifndef USE_PUSHCONSTANTS
// When not using push constants, we need to forward the draw ID to the next
// shader stages, so they can access 'perDrawBuffer'. The alternative would be
// to make the vertex shader read out all of the per-draw parameters and let
// it forward all of them to the next stage. But this is very wasteful and
// impacts performance, in particular if the next stage(s) don't make use of all
// parameters.
layout(location = 3) out DrawId
{
  flat uint drawId;
}
OUT_DRAWID;
uint getDrawId()
{
  return getBaseInstance();
}
#endif  // USE_PUSHCONSTANTS
#endif  //_VERTEX_SHADER_

#if _FRAGMENT_SHADER_
#ifndef USE_PUSHCONSTANTS
layout(location = 3) in DrawId
{
  flat uint drawId;
}
IN_DRAWID;

uint getDrawId()
{
  return IN_DRAWID.drawId;
};
#else   // USE_PUSHCONSTANTS
uint getDrawId();
#endif  // USE_PUSHCONSTANTS
#endif  // _FRAGMENT_SHADER

#ifdef _GEOMETRY_SHADER_
#ifndef USE_PUSHCONSTANTS
layout(location = 3) in InDrawId
{
  flat uint drawId;
}
IN_DRAWID[];
layout(location = 3) out OutDrawId
{
  flat uint drawId;
}
OUT_DRAWID;
uint getDrawId()
{
  return IN_DRAWID[0].drawId;
};
#else   // USE_PUSHCONSTANTS
uint getDrawId(); // declared, but not defined
#endif  // USE_PUSHCONSTANTS
#endif  //_GEOMETRY_SHADER_

#ifdef _COMPUTE_SHADER_
uint getDrawId(); // declared, but not defined
#endif  // _COMPUTE_SHADER_


uint getMaterialIndex()
{
#ifdef USE_PUSHCONSTANTS
  return PUSH.materialIndex;
#else   // USE_PUSHCONSTANTS
  return perDrawData[getDrawId()].materialIndex;
#endif  // USE_PUSHCONSTANTS
}

uint getMatrixIndex()
{
#ifdef USE_PUSHCONSTANTS
  return PUSH.matrixIndex;
#else   // USE_PUSHCONSTANTS
  return perDrawData[getDrawId()].matrixIndex;
#endif  // USE_PUSHCONSTANTS
}


uints_in getIdsAddress()
{
#ifdef USE_PUSHCONSTANTS
  return PUSH.idsAddr;
#else   // USE_PUSHCONSTANTS
  return perDrawData[getDrawId()].idsAddr;
#endif  // USE_PUSHCONSTANTS
}

uint getUniquePartOffset()
{
#ifdef USE_PUSHCONSTANTS
  return PUSH.uniquePartOffset;
#else   // USE_PUSHCONSTANTS
  return perDrawData[getDrawId()].uniquePartOffset;
#endif  // USE_PUSHCONSTANTS
}

#ifdef _VERTEX_SHADER_
uint getPartId()
{
#ifdef USE_PUSHCONSTANTS
  return getBaseInstance();
#else   // USE_PUSHCONSTANTS
  return perDrawData[getDrawId()].flexible;
#endif  // USE_PUSHCONSTANTS
}
#endif
