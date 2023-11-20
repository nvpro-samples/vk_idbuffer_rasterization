#ifndef USE_PUSHCONSTANTS
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
in   layout(location = ATTRIB_BASEINSTANCE) uint inBaseInstance;
uint getBaseInstance()
{
  return inBaseInstance;
}
#else
uint getBaseInstance()
{
  return gl_InstanceIndex;
}
#endif
#endif // VERTEX_SHADER

#ifdef _VERTEX_SHADER_
#ifndef USE_PUSHCONSTANTS
layout(location = 3) out DrawId
{
  flat uint drawId;
}
OUT_DRAWID;
#endif  // USE_PUSHCONSTANTS
uint getDrawId()
{
  return getBaseInstance();
}
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
#if USE_GEOMETRY_SHADER_PASSTHROUGH
layout(passthrough, location = 3) in DrawId
{
  flat uint drawId;
}
IN_DRAWID[];
#else // USE_GEOMETRY_SHADER_PASSTHROUGH
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
#endif // USE_GEOMETRY_SHADER_PASSTHROUGH

uint getDrawId()
{
  return IN_DRAWID[0].drawId;
};
#else   // USE_PUSHCONSTANTS
uint getDrawId();
#endif  // USE_PUSHCONSTANTS
#endif  //_GEOMETRY_SHADER_

#ifdef _COMPUTE_SHADER_
uint getDrawId();
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