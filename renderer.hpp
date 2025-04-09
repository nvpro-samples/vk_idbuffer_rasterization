/*
 * Copyright (c) 2021-2025, NVIDIA CORPORATION.  All rights reserved.
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
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


#ifndef RENDERER_H__
#define RENDERER_H__

#include "resources.hpp"
#include <nvh/profiler.hpp>

// print per-thread stats
#define PRINT_TIMER_STATS 1

namespace idraster {

class Renderer
{
public:
  enum PerDrawIndexMode
  {
    PER_DRAW_PUSHCONSTANTS,
    PER_DRAW_INDEX_BASEINSTANCE,
    PER_DRAW_INDEX_ATTRIBUTE
  };

  struct Stats
  {
    uint32_t drawCalls     = 0;
    uint32_t drawTriangles = 0;
  };

  struct Config
  {
    uint32_t objectFrom;
    uint32_t objectNum;
    bool     sorted          = true;
    bool     colorizeDraws   = false;
    bool     ignoreMaterials = false;
    uint32_t searchBatch     = 16;

    // MODE_PER_TRI_GLOBAL_PART_SEARCH_FS settings
    int  globalNaryN       = 4;
    int  globalNaryMin     = 64;
    int  globalNaryMaxIter = 4;
    bool globalSearchGuess = true;

    PerDrawIndexMode perDrawParameterMode = PER_DRAW_PUSHCONSTANTS;
  };

  struct DrawItem
  {
    int                 materialIndex;
    int                 geometryIndex;
    int                 matrixIndex;
    int                 partIndex;
    int                 objectIndex;
    uint32_t            objectOffset;
    int                 partCount;
    CadScene::DrawRange range;
  };

  static inline bool DrawItem_compare_groups(const DrawItem& a, const DrawItem& b)
  {
    int diff = 0;
    diff     = diff != 0 ? diff : (a.geometryIndex - b.geometryIndex);
    diff     = diff != 0 ? diff : (a.materialIndex - b.materialIndex);
    diff     = diff != 0 ? diff : (a.matrixIndex - b.matrixIndex);
    diff     = diff != 0 ? diff : (a.partIndex - b.partIndex);

    return diff < 0;
  }

  class Type
  {
  public:
    Type() { getRegistry().push_back(this); }

  public:
    virtual bool         isAvailable(const nvvk::Context& context) const = 0;
    virtual const char*  name() const                                    = 0;
    virtual Renderer*    create() const                                  = 0;
    virtual unsigned int priority() const { return 0xFF; }

    virtual Resources* resources() = 0;
  };

  typedef std::vector<Type*> Registry;

  static Registry& getRegistry()
  {
    static Registry s_registry;
    return s_registry;
  }

public:
  virtual bool init(const CadScene* NV_RESTRICT scene, Resources* resources, const Config& config, Stats& stats)
  {
    return false;
  }
  virtual void deinit() {}
  virtual void draw(const Resources::Global& global, Stats& stats) {}

  virtual ~Renderer() {}

  void fillDrawItems(std::vector<DrawItem>& drawItems, const CadScene* NV_RESTRICT scene, const Config& config, uint32_t maxCombine, Stats& stats);

  Config                      m_config;
  const CadScene* NV_RESTRICT m_scene;
};
}  // namespace idraster

#endif
