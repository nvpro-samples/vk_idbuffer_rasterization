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


#include <assert.h>
#include <algorithm>
#include "renderer.hpp"
#include <nvpwindow.hpp>

#include "common.h"

#pragma pack(1)


namespace idraster {
//////////////////////////////////////////////////////////////////////////

static void AddItem(std::vector<Renderer::DrawItem>& drawItems, const Renderer::Config& config, const Renderer::DrawItem& di)
{
  if(di.range.count)
  {
    drawItems.push_back(di);
  }
}

static void FillCombined(std::vector<Renderer::DrawItem>& drawItems,
                      const Renderer::Config&          config,
                      const CadScene::Object&          obj,
                      const CadScene::Geometry&        geo,
                      int                              objectIndex,
                      uint32_t maxCombine)
{
  Renderer::DrawItem di;
  di.geometryIndex = obj.geometryIndex;
  di.objectIndex   = objectIndex;
  di.materialIndex = -1;
  di.matrixIndex   = -1;
  di.objectOffset  = obj.uniquePartOffset;
  di.partCount     = 0;

  for(size_t p = 0; p < obj.parts.size(); p++)
  {
    const CadScene::ObjectPart&   part = obj.parts[p];
    const CadScene::GeometryPart& mesh = geo.parts[p];

    if(!part.active)
      continue;
    
    // finish old, start new
    if (di.matrixIndex != part.matrixIndex || (!config.ignoreMaterials && di.materialIndex != part.materialIndex) ||
      di.range.offset + di.range.count * sizeof(uint32_t) != mesh.indexSolid.offset ||
      di.partCount == maxCombine)
    {
      AddItem(drawItems, config, di);

      // new start
      di.matrixIndex   = part.matrixIndex;
      di.materialIndex = config.ignoreMaterials ? 0 : part.materialIndex;
      di.range.offset  = mesh.indexSolid.offset;
      di.partIndex     = uint32_t(p);
      di.partCount     = 0;
      di.range.count   = 0;
    }
    
    di.range.count += mesh.indexSolid.count;
    di.partCount += 1;
  }

  AddItem(drawItems, config, di);
}

static void FillIndividual(std::vector<Renderer::DrawItem>& drawItems,
                           const Renderer::Config&          config,
                           const CadScene::Object&          obj,
                           const CadScene::Geometry&        geo,
                           int                              objectIndex)
{
  for(size_t p = 0; p < obj.parts.size(); p++)
  {
    const CadScene::ObjectPart&   part = obj.parts[p];
    const CadScene::GeometryPart& mesh = geo.parts[p];

    if(!part.active)
      continue;

    Renderer::DrawItem di;
    di.geometryIndex = obj.geometryIndex;
    di.matrixIndex   = part.matrixIndex;
    di.materialIndex = part.materialIndex;
    di.partIndex     = uint32_t(p);
    di.range         = mesh.indexSolid;
    di.objectIndex   = objectIndex;
    di.objectOffset  = obj.uniquePartOffset;

    AddItem(drawItems, config, di);
  }
}

void Renderer::fillDrawItems(std::vector<DrawItem>& drawItems, const CadScene* NV_RESTRICT scene, const Config& config, uint32_t maxCombine, Stats& stats)
{
  size_t maxObjects = scene->m_objects.size();
  size_t from       = std::min(maxObjects - 1, size_t(config.objectFrom));
  maxObjects        = std::min(maxObjects, from + size_t(config.objectNum));

  for(size_t i = from; i < maxObjects; i++)
  {
    const CadScene::Object&   obj = scene->m_objects[i];
    const CadScene::Geometry& geo = scene->m_geometry[obj.geometryIndex];

    if(maxCombine)
    {
      FillCombined(drawItems, config, obj, geo, int(i), maxCombine);
    }
    else
    {
      FillIndividual(drawItems, config, obj, geo, int(i));
    }
  }

  if(config.sorted)
  {
    std::sort(drawItems.begin(), drawItems.end(), DrawItem_compare_groups);
  }

  for(size_t i = 0; i < drawItems.size(); i++)
  {
    stats.drawCalls++;
    stats.drawTriangles += drawItems[i].range.count / 3;
  }
}

}  // namespace idraster
