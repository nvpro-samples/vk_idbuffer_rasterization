# vk_idbuffer_rasterization

In CAD rendering a surface is often created of multiple feature parts, for example from nurbs-surfaces joined together create the solid shape.
When rendering an object with material effects, we normally ignore such parts and attempt to render the shape as a whole. However, under some
circumstances we might want to have the ability to identify each part uniquely (see the individually-tinted parts in the screenshot).

![sample screenshot - CAD model courtesy of PTC](doc/screenshot.png)

(CAD model courtesy of PTC)

In this sample we leverage the unique part ids to implement a very basic mouse selection highlight (animated inverted color effect over the part).
Another use-case for idbuffers / item buffers containing unique part IDs is screenspace-based outlining for feature- or silhouette-edges.

One issue we face is that, while an object might have many triangles in whole, when it's made of many such parts, they tend to have few triangles each. Therefore, rendering the parts individually does not allow GPUs to run at full performance, even when we remove the CPU-bottlenecks from such small drawcalls, the GPU can become front-end bound on the GPU, as the front-end is responsible for creating the drawcalls.

A modern GPU tends to be wide, that means it has lots of execution units that prefer big work loads, so a drawcall with just a few triangles (less than 500) tends to not be able to saturate the hardware quick enough. While the hardware is able to process multiple drawcalls in parallel, there tend to be limits on how much can run in parallel, and also if the work per-draw is small the setup/overhead of each draw just can bite us in the end.

Sometimes it's not avoidable to have objects with few drawcalls, ideally those get not drawn in bulk, but interleaved with objects that have more triangles so we can hide this problem better.

In this sample we showcase a few rendering techniques to get a per-part ID within the fragment shader and still be efficient.

Typical UI operations:

- `renderer` change between different techniques to render the part IDs
- `per-draw parameters` alter the way per-draw parameters are passed and how drawcalls are submitted.
- `search batch` the number of parts per drawcall to batch in the `search` renderers.
- `part color weight` slider allows to blend between the individual part colors and the material color
- `colorize drawcalls` when active overrides the object's material color with a per-draw color (useful to show the batching)
- `model copies` increase the number of instances of the model (recommended for fast GPUs and performance investigation)
- `Render GPU [ms]`: milliseconds it took to render the scene, please disable vsync (press V, see window title) for performance investigation and create meaningful loads.


## CAD Model Setup

Some assumptions about how we organized our CAD model / buffers:

* geometry is a set of vertices and indices
* geometry is made of multiple parts, which each take a range of the geometry's triangle indices.
* geometry can be drawn as single drawcall spanning all triangles / all parts.
* one object references a geometry (so we can instance the same geometry for each wheel)
* each object has a `uniquePartOffset` so that a final part that is rendered can be uniquely identified 

## Techniques to draw per-part IDs

We implemented a few different approaches to get a per-part ID in the fragment-shader stage at the end.
Not all of them are as versatile as others. As always with graphics programming, your mileage will vary as
the outcome greatly depends on your typical input data.

### **per-draw part index**

Each drawcall sets the part index. This means if our model has lots of parts, we get a ton of drawcalls.

For CAD parts this often has the **risk of being slowest technique**, but might be the simplest starting point.

The performance outcome of rendering techniques always depends on your data, if your "parts" tend to have plenty triangles,
then this way is totally fine to use.

``` cpp
// drawcall setup
  // We encode the partIndex in the "baseInstance" value of each drawcall.
  // This makes this technique also friendly with multi-draw-indirect structure.
  // We don't really use instancing, so the value will be always be available
  // as-is in the drawcall.
  vkCmdDrawIndexed(cmd, part.indexCount, 1, part.firstIndex, part.firstVertex,
                        part.ID);

// vertex shader simply passes through the partIndex
// in Vulkan gl_InstanceIndex is the result of gl_InstanceID + gl_BaseInstance
  // input
  layout(location=0) in Interpolants {
    ...
    flat uint partIndex;
  } OUT;

  // code
  ...
  OUT.partIndex = gl_InstanceIndex;

// fragment shader picks up the value from the vertex-shader
  // input
  layout(location=0) in Interpolants {
    ...
    flat uint partIndex;
  } IN;

  // code
  ...
  int partIndex = IN.partIndex;

```

### **per-triangle part index**

In this variant we store a `triangle partID buffer` for each triangle of the object's geometry. This maps each triangle to the part it belongs to. Depending on the number of parts we can use `uint8_t`, `uint16_t` or `uint32_t` array. In this example we used `uint32_t` and in the UI you can see the memory cost under `triangle ids`.

While this costs a bit of extra memory, it tends to be the fastest variant, as we can still render the entire object in one shot, independent how many parts it contains, that is why we **recommend this setup**.

``` cpp
// drawcall setup
  // similar as before we hijack the "baseInstance" value to get a cheap per-draw
  // value. This time we store the geometry's buffer offset into the `triangle partID buffer`.
  vkCmdDrawIndexed(cmd, geometry.indexCount, 1, geometry.firstIndex, geometry.firstVertex, 
                        geometry.partTriCountsOffset);

// fragment shader
  // lookup each triangle's partId
  // "PUSH.idsAddr" is a pushconstant that contains the buffer_reference address
  //                for array that contains the part ID per triangle.
  // "IN_ID.idsOffset" is used a bit like "firstIndex" in a drawcall, it allows us 
  //                   to store many parts worth of partTriCounts in the buffer.
  //                   It is piped through the vertex-shader as in the simple setup.
  int partIndex = int(PUSH.idsAddr.d[gl_PrimitiveID + int(IN_ID.idsOffset)]);

```

The easiest and often fastest option to do this lookup is inside the fragment-shader. The renderers with the `fs` suffix do the lookup there:
- [drawid_primid.frag.glsl](drawid_primid.frag.glsl).

Another alternative is using a geometry-shader and compute a new gl_PrimitiveID passed to the fragment shader. The renderers with the `gs` perform the lookup in the geometry-shader stage:
- [drawid_primid_gs.geo.glsl](drawid_primid_gs.geo.glsl). 

Using the geometry-shader version is typically slower than the fragement-shader, and we also do the operation/fetching prior early depth culling. So we don't recommend using the geometry-shader technique either.

### **per-triangle search part index**

In this technique we try to lower the memory footprint of the previous technique. Before we had one partID per triangle, but what if we just store how many triangles each part stores and then figure out which part we are. 

This means we only need the number of triangles each part has, which is a lot less memory (see `part ids` in UI), as it no longer depends on the actual number of triangles.

We batch some parts together as single drawcall and then search based on `gl_PrimitiveID`. 

The drawcalls are batched to contain up to `SEARCH_COUNT` many parts and our shader-code is optimized for this.
In our sample a value of 16 worked well.

``` cpp
// drawcall setup
  // We batch our geometry parts into a reduced number of drawcalls.
  // This time we encode some crucial information about the batch in the "baseInstance"
  // "partCount" tells us how many parts are in the batch (up to SEARCH_COUNT many)
  // "partOffset" is the partID of the first part within this batch.
  vkCmdDrawIndexed(cmd, batch.indexCount, 1, batch.firstIndex, batch.firstVertex,
                        batch.partCount | (batch.partOffset << 8));


// fragment shader

  // PUSH.idsAddr points to partTriCounts.
  uints_in partTriCounts = PUSH.idsAddr;

  // the batch meta info that we get per-draw
  uint partOffset = IN_ID.idsOffset >> 8;
  uint partCount  = IN_ID.idsOffset & 0xFF;

  int begin = 0;
  int partIndex = 0;

  // unroll support is provided via GL_EXT_control_flow_attributes
  [[unroll]]
  for (int i = 0; i < SEARCH_COUNT; i++)
  {
    // for each part in the batch get number of triangles
    // (we pad our partTriCounts buffer at the end so that this hardcoded search window never
    //  creates out-of-memory access)
    
    // don't make this load part of a dynamic condition, so that the compiler
    // can batch-load all SEARCH_COUNT many loads in separate registers, which reduces
    // memory latency.
    partTriangleCount = int(partTriCounts.d[partOffset + i]);

    // we hardcoded this loop in the shader hence we add the `(i < partCount)` condition
    // which is dynamic per batch
    // if the part is valid then look if the current gl_PrimitiveID fits in the range
    [[flatten]]
    if (i < partCount 
      && gl_PrimitiveID >= begin 
      && gl_PrimitiveID < begin + partTriangleCount)
    {
      partIndex = i;
    }
    // shift begin of next range
    begin += partTriangleCount;
  }
```

### Passing per draw information
The sample implements different code paths to pass per-draw information, which can be switched between using the `per-draw parameters` UI option.
Especially at very high frequencies (low number of triangles/work per draw) the approaches can make a difference.

The [`per_draw_inputs.glsl`](per_draw_inputs.glsl) file does wrap these different methods.

#### Push Constants
When selecting `pushconstants` in the UI, the sample will supply per-draw
parameters to the GPU via via Vulkan's 'Push Constants'. These are comparable to
'Uniforms' in OpenGL. Push Constants provide a convenient way to provide shaders
with data that changes dynamically between draw calls. Traditionally this data
might be things like the ModelViewProjection matrix and material or lighting
parameters. Unlike updates to descriptor sets, push constant data updates are
embedded into the command buffer. This way no additional synchronization or 
cache management is needed, which makes push constants a very easy way to handle.
The downside to push constants is the additional overhead - they should not
be used to update a lot of parameters for each draw call. Our code sample attempts
to minimize the number of push constant updates by making only updates to
parameters that changed between draw calls.

#### Multi-Draw Indirect and gl_BaseInstance
As the previous techniques sometimes rely on passing some information efficiently
per draw, let's look at different possibilities. The UI option 
`MDI & gl_BaseInstance` offers a path that is using _Multi-Draw-Indirect_ to
issue draw calls and a technique based around 'gl_BaseInstance' to pass per-draw
parameters to the shader(s).

Multi-Draw-Indirect (MDI) offers a way in Vulkan to submit many draws at once
with just one command, `vkCmdDraw[Indexed]Indirect`, thereby potentially
increasing performance considerably. With `vkCmdDraw[Indexed]Indirect`, the
per-draw parameters that are typically provided as function parameters to
`vkCmdDrawIndexed` like number of indices, first vertex, base instance etc.
need to be provided in an "Indirect Buffer" filled with
`VkDrawIndexedIndirectCommand` objects that describe the individual draws to
the GPU. We do loose one important feature, though: the ability to change state
inbetween these draws - in particular changes to the push constants. All draws
executed as part of an MDI drawcall share the same state. Therefore we need to
find a different way to pass the required transform matrix, material ID, part
identifier etc to the shaders. Currently, the only way in core Vulkan to 
communicate a per-draw identifier is to use the 
`VkDraw[Indexed]IndirectCommand::firstInstance` member. 
This parameter is passed through to the vertex shader as `gl_BaseInstance` 
(available in GLSL 4.6) and has no effect on rendering if instancing is not
used. Since our sample is not using instancing we are free to use this 
parameter to pass a user defined ID along to the vertex shader. When the draws
do not actually make use of instancing, gl_InstanceID can be used synonymously 
to gl_BaseInstance which may work better on some hardware. Using gl_BaseInstance,
the shader can identify which part the current draw belongs to and use it to do
a lookup into a storage buffer containing the actual draw parameters needed
for the current draw. In our sample's case, we keep a storage buffer containing
an array of `DrawPushData` around that we index with gl_InstanceID.

```
struct DrawPushData
{
  // Common to all vertex shaders
  uint matrixIndex;

  uint flexible;

  // Simple per-part fragment push constants for MODE_PER_DRAW_BASEINST
  uint materialIndex;

  // Added to the part ID when shading() so the same ID for different objects is
  // a different color.
  uint uniquePartOffset;

  // Address bound contains different content per mode:
  // - MODE_PER_TRI_ID*: trianglePartIds - per-triangle part IDs
  // - MODE_PER_TRI_*BATCH_PART_SEARCH*: partTriCounts - per-part triangle counts
  // - MODE_PER_TRI_*GLOBAL_PART_SEARCH*: partTriOffsets - running per-part triangle offsets
  BUFFER_REFERENCE(uints_in, idsAddr);
};
```
It provides the vertex shader with the transform matrix, the fragment shader with
the material index and means to identify the object the currently drawn triangle
belongs to with the algorithms described prior.

Notice that we don't pass these parameters individually as varyings (in/out parameters)
between shader stages. Instead, the vertex shader only passes the current draw ID along in
a flat varying. This is done to minimize passing data between the shader stages. Saving
on inter-stage data in turn saves on-chip memory and thus allows for better utilization of
the GPU, in particular for large model rendering, when the the number of rendered primitives
reaches and surpasses the amount of pixels on screen.

Each shader stage that needs per-draw information from the per-draw buffer does its own
lookup. This lookup will be highly uniform and thus cause the data to be kept in L1/L2
cache with high likelyhood.

While passing the data as individual varyings is possible, it increases the amount of
on-chip memory each output vertex needs. This increased usage negatively impacts
occupancy, meaning less vertex-shader threads can be run in parallel.

#### Multi-Draw Indirect and instanced vertex attribute
Via the `MDI & instanced attribute` option in UI choses a renderer path which replaces the use of
gl_BaseInstance shader built-in through an instanced vertex attribute.

On some hardware it is faster to emulate gl_BaseInstance via an instanced vertex attribute
by providing a buffer which contains the draw index in the shape of `buffer[x] = x`.
When binding this buffer as instanced vertex attribute, this vertex attribute will then
provide the gl_BaseInstance identifier indirectly. Once the draw ID is fetched from the
instanced vertex attribute, the remaining handling of per-draw parameters remains the same
as the _MDI & gl_BaseInstance_ option.

### Performance

Summarizing our three main techniques:

```
we want to draw a geometry with 9 parts
a b c d e f g h i
```

**per-draw part index**: we get 9 drawcalls, each with one part
```
0 1 2 3 4 5 6 7 8
a b c d e f g h i
```

**per-triangle part index**: we get 1 drawcall, spanning all parts
```
0
abcdefghi
```

**per-triangle search part index**: with this technique and SEARCH_COUNT=4 we get 3 drawcalls
```
0    1    2
abcd efgh i
```

Results for a NVIDIA GeForce 3080 and `model copies = 3` with 1440p + 4x msaa and `search batch = 16` and `pushconstants`.

| renderer                                      | drawcalls | time in milliseconds |
|-----------------------------------------------|-----------|----------------------|
| per-draw part index                           |   296 049 |               12.2   |
| per-triangle part index fs                    |     6 777 |              **2.3** |
| per-triangle part index gs                    |     6 777 |                4.3   |
| per-triangle search part index fs             |    22 260 |                2.5   |
| per-triangle search part index gs             |    22 260 |                4.6   |

We can see that the per-drawcall part index clearly is the worst option for this model, as it contains of lots of parts with few triangles.
As mentioned before the easiest plug-in solution is typically having a per-triangle buffer.

For the single car the `triangle partID buffer` was around 9 MB (32-bit per triangle) and the `partID buffer` used for searches just 268 KB (32-bit per part). So if you are tight on memory the `per-triangle search fs` method may be good choice.

And as reminder evaluate the techniques with your kind of rendering setup and typical content.

## Selection Highlight

The selection highlight is done by figuring out the partIndex underneath the mouse directly in the fragment shader. Using the `atomicMin` instruction provided through `GL_EXT_shader_atomic_int64`, we store the global unique `partIndex` in the lower 32-bit and the fragment depth in the upper 64-bit. At the end of the rendering we will have the closest return in the variable. We copy this variable over and use it for the visual highlight of the next frame. See the following code also in [drawid_shading.glsl](drawid_shading.glsl)


``` cpp
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
```

**Tip for VR**

While this sample does a simple ray-test along the mouse cursor, one can use the same principle setup for an arbitrary selection ray. By treating each fragment shader invocation as a small plane we can intersect that with the selection ray. Then test if the intersection point is close to the current gl_FragCoord and if so run the atomicMin above, but with the hit distance rather than depth (means only few fragment shader invocations hit the atomicMin). That would give you a very cheap arbitrary selection ray, say controlled by VR controllers, on any visible surface almost for free. It comes with the restriction that you must have clear vision on anything you want to select, but that is often okay.

## Building
Make sure to have installed the [Vulkan-SDK](http://lunarg.com/vulkan-sdk/). Always use 64-bit build configurations.

Ideally, clone this and other interesting [nvpro-samples](https://github.com/nvpro-samples) repositories into a common subdirectory. You will always need [nvpro_core](https://github.com/nvpro-samples/nvpro_core). The nvpro_core is searched either as a subdirectory of the sample, or one directory up.

If you are interested in multiple samples, you can use [build_all](https://github.com/nvpro-samples/build_all) CMAKE as entry point, it will also give you options to enable/disable individual samples when creating the solutions.
