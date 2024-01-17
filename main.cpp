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
 *
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */


/* Contact ckubisch@nvidia.com (Christoph Kubisch) for feedback */

#define DEBUG_FILTER 1

#include <imgui/imgui_helper.h>

#include <nvvk/appwindowprofiler_vk.hpp>
#include <nvvk/debug_util_vk.hpp>

#include <nvh/cameracontrol.hpp>
#include <nvh/fileoperations.hpp>
#include <nvh/geometry.hpp>

#include <algorithm>

#include "renderer.hpp"
#include "resources_vk.hpp"
#include <glm/gtc/matrix_access.hpp>

namespace idraster {
int const SAMPLE_SIZE_WIDTH(1024);
int const SAMPLE_SIZE_HEIGHT(768);

class Sample : public nvvk::AppWindowProfilerVK
{

  enum GuiEnums
  {
    GUI_RENDERER,
    GUI_PERDRAWMODE,
    GUI_MSAA,
  };

public:
  struct Tweak
  {
    int              renderer      = 0;
    int              msaa          = 4;
    int              copies        = 1;
    bool             animation     = false;
    bool             animationSpin = false;
    int              cloneaxisX    = 1;
    int              cloneaxisY    = 1;
    int              cloneaxisZ    = 1;
    float            percent       = 1.001f;
    float            partWeight    = 0.3f;
    Renderer::Config config;
  };


  bool m_useUI = true;

  ImGuiH::Registry m_ui;
  double           m_uiTime = 0;

  Tweak     m_tweak;
  Tweak     m_lastTweak;
  bool      m_lastVsync;
  glm::vec3 m_upVector = {0, 0, 1};

  CadScene                  m_scene;
  std::vector<unsigned int> m_renderersSorted;
  std::string               m_rendererName;

  Renderer* NV_RESTRICT  m_renderer;
  Resources* NV_RESTRICT m_resources;
  Resources::Global      m_shared;
  Renderer::Stats        m_renderStats;

  std::string m_modelFilename;
  double      m_animBeginTime;

  double m_lastFrameTime = 0;
  double m_frames        = 0;

  double m_statsFrameTime    = 0;
  double m_statsCpuTime      = 0;
  double m_statsGpuTime      = 0;
  double m_statsGpuDrawTime  = 0;
  double m_statsGpuBuildTime = 0;

  bool initProgram();
  bool initScene(const char* filename, int clones, int cloneaxis);
  bool initFramebuffers(int width, int height);
  void initRenderer(int type);
  void deinitRenderer();

  void setupConfigParameters();
  void setRendererFromName();

  template <typename T>
  bool tweakChanged(const T& val)
  {
    size_t offset = size_t(&val) - size_t(&m_tweak);
    return memcmp(&val, reinterpret_cast<uint8_t*>(&m_lastTweak) + offset, sizeof(T)) != 0;
  }

  Sample()
      : AppWindowProfilerVK(false)
  {
    setupConfigParameters();
    m_contextInfo.addDeviceExtension(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME, true);

    m_contextInfo.apiMajor = 1;
    m_contextInfo.apiMinor = 2;

    // validation layer bug, mismatch with geometry shader passthrough
    m_context.ignoreDebugMessage(0xb6cf33fe);

    // get access to debug labels
    m_contextInfo.addInstanceExtension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME, false);
    nvvk::DebugUtil::setEnabled(true);

#if defined(NDEBUG)
    setVsync(false);
#endif
  }

public:
  bool validateConfig() override;

  void postBenchmarkAdvance() override { setRendererFromName(); }

  bool begin() override;
  void think(double time) override;
  void resize(int width, int height) override;

  void processUI(int width, int height, double time);

  nvh::CameraControl m_control;

  void end() override;

  // return true to prevent m_window updates
  bool mouse_pos(int x, int y) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_pos(x, y);
  }
  bool mouse_button(int button, int action) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_button(button, action);
  }
  bool mouse_wheel(int wheel) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::mouse_wheel(wheel);
  }
  bool key_char(int key) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::key_char(key);
  }
  bool key_button(int button, int action, int mods) override
  {
    if(!m_useUI)
      return false;

    return ImGuiH::key_button(button, action, mods);
  }
};


bool Sample::initProgram()
{
  return true;
}

bool Sample::initScene(const char* filename, int clones, int cloneaxis)
{
  std::string modelFilename(filename);

  if(!nvh::fileExists(filename))
  {
    modelFilename = nvh::getFileName(filename);
    std::vector<std::string> searchPaths;
    searchPaths.push_back("./");
    searchPaths.push_back(exePath() + PROJECT_RELDIRECTORY);
    searchPaths.push_back(exePath() + PROJECT_DOWNLOAD_RELDIRECTORY);
    modelFilename = nvh::findFile(modelFilename, searchPaths);
  }

  m_scene.unload();

  bool status = m_scene.loadCSF(modelFilename.c_str(), clones, cloneaxis);
  if(status)
  {
    LOGI("\nscene %s\n", filename);
    LOGI("geometries: %6d\n", uint32_t(m_scene.m_geometry.size()));
    LOGI("materials:  %6d\n", uint32_t(m_scene.m_materials.size()));
    LOGI("nodes:      %6d\n", uint32_t(m_scene.m_matrices.size()));
    LOGI("objects:    %6d\n", uint32_t(m_scene.m_objects.size()));
    LOGI("\n");
  }
  else
  {
    LOGW("\ncould not load model %s\n", modelFilename.c_str());
  }

  m_shared.animUbo.numMatrices = uint(m_scene.m_matrices.size());

  return status;
}

bool Sample::initFramebuffers(int width, int height)
{
  return m_resources->initFramebuffer(width, height, m_tweak.msaa, getVsync());
}

void Sample::deinitRenderer()
{
  if(m_renderer)
  {
    m_resources->synchronize();
    m_renderer->deinit();
    delete m_renderer;
    m_renderer = NULL;
  }
}

void Sample::initRenderer(int typesort)
{
  int type = m_renderersSorted[typesort];

  deinitRenderer();

  if(Renderer::getRegistry()[type]->resources() != m_resources)
  {
    if(m_resources)
    {
      m_resources->synchronize();
      m_resources->deinit();
    }
    m_resources = Renderer::getRegistry()[type]->resources();
    bool valid  = m_resources->init(&m_context, &m_swapChain, &m_profiler);
    valid = valid && m_resources->initFramebuffer(m_windowState.m_swapSize[0], m_windowState.m_swapSize[1], m_tweak.msaa, getVsync());
    valid                = valid && m_resources->initPrograms(exePath(), std::string());
    valid                = valid && m_resources->initScene(m_scene);
    m_resources->m_frame = 0;

    if(!valid)
    {
      LOGE("resource initialization failed for renderer: %s\n", Renderer::getRegistry()[type]->name());
      exit(-1);
    }

    m_lastVsync = getVsync();
  }

  Renderer::Config config{m_tweak.config};
  config.objectFrom = 0;
  config.objectNum  = uint32_t(double(m_scene.m_objects.size()) * double(m_tweak.percent));
  config.passthrough = m_tweak.config.passthrough && m_context.hasDeviceExtension(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME);

  m_renderStats = Renderer::Stats();

  LOGI("renderer: %s\n", Renderer::getRegistry()[type]->name());
  m_renderer = Renderer::getRegistry()[type]->create();
  m_renderer->init(&m_scene, m_resources, config, m_renderStats);

  LOGI("drawCalls:    %9d\n", m_renderStats.drawCalls);
  LOGI("drawTris:     %9d\n", m_renderStats.drawTriangles);
}


void Sample::end()
{
  deinitRenderer();
  if(m_resources)
  {
    m_resources->deinit();
  }
  ResourcesVK::deinitImGui(m_context);
}


bool Sample::begin()
{
#if !PRINT_TIMER_STATS
  m_profilerPrint = false;
  m_timeInTitle   = true;
#else
  m_profilerPrint = true;
  m_timeInTitle   = true;
#endif

  m_renderer  = NULL;
  m_resources = NULL;

  ImGuiH::Init(m_windowState.m_winSize[0], m_windowState.m_winSize[1], this);
  ResourcesVK::initImGui(m_context);

  bool validated(true);
  validated = validated && initProgram();
  validated = validated
              && initScene(m_modelFilename.c_str(), m_tweak.copies - 1,
                           (m_tweak.cloneaxisX << 0) | (m_tweak.cloneaxisY << 1) | (m_tweak.cloneaxisZ << 2));

  const Renderer::Registry registry = Renderer::getRegistry();
  for(size_t i = 0; i < registry.size(); i++)
  {
    if(registry[i]->isAvailable(m_context))
    {
      uint sortkey = uint(i);
      sortkey |= registry[i]->priority() << 16;
      m_renderersSorted.push_back(sortkey);
    }
  }

  if(m_renderersSorted.empty())
  {
    LOGE("No renderers available\n");
    return false;
  }

  std::sort(m_renderersSorted.begin(), m_renderersSorted.end());

  for(size_t i = 0; i < m_renderersSorted.size(); i++)
  {
    m_renderersSorted[i] &= 0xFFFF;
  }

  for(size_t i = 0; i < m_renderersSorted.size(); i++)
  {
    LOGI("renderers found: %d %s\n", uint32_t(i), registry[m_renderersSorted[i]]->name());
  }

  setRendererFromName();

  if(m_useUI)
  {
    auto& imgui_io       = ImGui::GetIO();
    imgui_io.IniFilename = nullptr;

    for(size_t i = 0; i < m_renderersSorted.size(); i++)
    {
      m_ui.enumAdd(GUI_RENDERER, int(i), registry[m_renderersSorted[i]]->name());
    }

    m_ui.enumAdd(GUI_PERDRAWMODE, Renderer::PER_DRAW_PUSHCONSTANTS, "pushconstants");
    m_ui.enumAdd(GUI_PERDRAWMODE, Renderer::PER_DRAW_INDEX_BASEINSTANCE, "MDI & gl_BaseInstance");
    m_ui.enumAdd(GUI_PERDRAWMODE, Renderer::PER_DRAW_INDEX_ATTRIBUTE, "MDI & instanced attribute");

    m_ui.enumAdd(GUI_MSAA, 0, "none");
    m_ui.enumAdd(GUI_MSAA, 2, "2x");
    m_ui.enumAdd(GUI_MSAA, 4, "4x");
    m_ui.enumAdd(GUI_MSAA, 8, "8x");
  }

  m_control.m_sceneOrbit     = glm::vec3(m_scene.m_bbox.max + m_scene.m_bbox.min) * 0.5f;
  m_control.m_sceneDimension = glm::length((m_scene.m_bbox.max - m_scene.m_bbox.min));
  m_control.m_viewMatrix = glm::lookAt(m_control.m_sceneOrbit - (-vec3(1, 1, 1) * m_control.m_sceneDimension * 0.5f),
                                       m_control.m_sceneOrbit, m_upVector);

  m_shared.animUbo.sceneCenter    = m_control.m_sceneOrbit;
  m_shared.animUbo.sceneDimension = m_control.m_sceneDimension * 0.2f;
  m_shared.animUbo.numMatrices    = uint(m_scene.m_matrices.size());
  m_shared.sceneUbo.wLightPos     = (m_scene.m_bbox.max + m_scene.m_bbox.min) * 0.5f + m_control.m_sceneDimension;
  m_shared.sceneUbo.wLightPos.w   = 1.0;


  initRenderer(m_tweak.renderer);

  m_lastTweak = m_tweak;

  return validated;
}


void Sample::processUI(int width, int height, double time)
{
  // Update imgui configuration
  auto& imgui_io       = ImGui::GetIO();
  imgui_io.DeltaTime   = static_cast<float>(time - m_uiTime);
  imgui_io.DisplaySize = ImVec2(width, height);

  m_uiTime = time;

  ImGui::NewFrame();
  ImGui::SetNextWindowSize(ImGuiH::dpiScaled(440, 0), ImGuiCond_FirstUseEver);
  
  if(ImGui::Begin("NVIDIA " PROJECT_NAME, nullptr))
  {
    ImGui::PushItemWidth(ImGuiH::dpiScaled(280));

    m_ui.enumCombobox(GUI_RENDERER, "renderer", &m_tweak.renderer);
    m_ui.enumCombobox(GUI_PERDRAWMODE, "per-draw parameters", &m_tweak.config.perDrawParameterMode);

    if(m_context.hasDeviceExtension(VK_NV_GEOMETRY_SHADER_PASSTHROUGH_EXTENSION_NAME))
    {
      ImGui::Checkbox("use geometry shader passthrough", &m_tweak.config.passthrough);
    }
    if (ImGui::CollapsingHeader("search parameters"))
    {
      ImGui::PushItemWidth(ImGuiH::dpiScaled(170));
      ImGui::Indent( ImGuiH::dpiScaled(24) );
      ImGui::Text("local search:");
      ImGuiH::InputIntClamped("search batch", &m_tweak.config.searchBatch, 4, 32, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::Separator();
      ImGui::Text("global search:");
      ImGui::Checkbox("initial guess", &m_tweak.config.globalSearchGuess);
      ImGuiH::InputIntClamped("N-ary N", &m_tweak.config.globalNaryN, 3, 16, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
      ImGuiH::InputIntClamped("N-ary fallback at", &m_tweak.config.globalNaryMin, m_tweak.config.globalNaryN + 1, 10000,
                              1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
      ImGuiH::InputIntClamped("N-ary max iter", &m_tweak.config.globalNaryMaxIter, 0, 32, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
      ImGui::PopItemWidth();
      ImGui::Unindent( ImGuiH::dpiScaled(24) );
    }

    ImGui::Separator();
    ImGui::SliderFloat("part color weight", &m_tweak.partWeight, 0.0f, 1.00f);
    ImGui::Checkbox("colorize drawcalls", &m_tweak.config.colorizeDraws);
    ImGui::Checkbox("ignore materials", &m_tweak.config.ignoreMaterials);
    ImGui::Separator();
    ImGuiH::InputIntClamped("model copies", &m_tweak.copies, 1, 16, 1, 1, ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SliderFloat("pct visible", &m_tweak.percent, 0.0f, 1.001f);
    ImGui::Separator();
    //m_ui.enumCombobox(GUI_MSAA, "msaa", &m_tweak.msaa);
    ImGui::Checkbox("sorted once (minimized state changes)", &m_tweak.config.sorted);
    ImGui::Checkbox("animation", &m_tweak.animation);
    ImGui::Separator();
    ImGui::PopItemWidth();

    {
      int avg = 50;

      if(m_lastFrameTime == 0)
      {
        m_lastFrameTime = time;
        m_frames        = -1;
      }

      if(m_frames > 4)
      {
        double curavg = (time - m_lastFrameTime) / m_frames;
        if(curavg > 1.0 / 30.0)
        {
          avg = 10;
        }
      }

      if(m_profiler.getTotalFrames() % avg == avg - 1)
      {
        nvh::Profiler::TimerInfo info;
        m_profiler.getTimerInfo("Render", info);
        m_statsCpuTime      = info.cpu.average;
        m_statsGpuTime      = info.gpu.average;
        m_statsGpuBuildTime = 0;
        bool hasPres        = m_profiler.getTimerInfo("Pre", info);
        m_statsGpuBuildTime = hasPres ? info.gpu.average : 0;
        m_profiler.getTimerInfo("Draw", info);
        m_statsGpuDrawTime = info.gpu.average;
        m_statsFrameTime   = (time - m_lastFrameTime) / m_frames;
        m_lastFrameTime    = time;
        m_frames           = -1;
      }

      m_frames++;

      float gpuTimeF = float(m_statsGpuTime);
      float cpuTimeF = float(m_statsCpuTime);
      float bldTimef = float(m_statsGpuBuildTime);
      float drwTimef = float(m_statsGpuDrawTime);
      float maxTimeF = std::max(std::max(cpuTimeF, gpuTimeF), 0.0001f);

      //ImGui::Text("Frame          [ms]: %2.1f", m_statsFrameTime*1000.0f);
      //ImGui::Text("Render     CPU [ms]: %2.3f", cpuTimeF / 1000.0f);
      ImGui::Text("Render     GPU [ms]: %2.3f", gpuTimeF / 1000.0f);

      //ImGui::ProgressBar(cpuTimeF / maxTimeF, ImVec2(0.0f, 0.0f));
      ImGui::Separator();
      ImGui::Text(" triangle ids:  %9ld KB\n", m_scene.m_trianglePartIdsSize / 1024);
      ImGui::Text(" part ids:      %9ld KB\n", m_scene.m_partTriCountsSize / 1024);
      ImGui::Text(" draw calls:    %9d\n", m_renderStats.drawCalls);
      ImGui::Text(" draw tris:     %9d\n", m_renderStats.drawTriangles);
    }
  }
  ImGui::End();
}

void Sample::think(double time)
{
  int width  = m_windowState.m_swapSize[0];
  int height = m_windowState.m_swapSize[1];

  if(m_useUI)
  {
    processUI(width, height, time);
  }

  m_control.processActions({m_windowState.m_winSize[0], m_windowState.m_winSize[1]},
                           glm::vec2(m_windowState.m_mouseCurrent[0], m_windowState.m_mouseCurrent[1]),
                           m_windowState.m_mouseButtonFlags, m_windowState.m_mouseWheel);

  bool shadersChanged = false;
  if(m_windowState.onPress(KEY_R))
  {
    m_resources->synchronize();
    m_resources->reloadPrograms(std::string());
    shadersChanged = true;
  }

  if(tweakChanged(m_tweak.msaa) || getVsync() != m_lastVsync)
  {
    m_resources->synchronize();
    m_lastVsync = getVsync();
    m_resources->initFramebuffer(width, height, m_tweak.msaa, getVsync());
  }

  bool sceneChanged = false;
  if(tweakChanged(m_tweak.copies) || tweakChanged(m_tweak.cloneaxisX) || tweakChanged(m_tweak.cloneaxisY)
     || tweakChanged(m_tweak.cloneaxisZ))
  {
    sceneChanged = true;
    m_resources->synchronize();
    deinitRenderer();
    m_resources->deinitScene();
    initScene(m_modelFilename.c_str(), m_tweak.copies - 1,
              (m_tweak.cloneaxisX << 0) | (m_tweak.cloneaxisY << 1) | (m_tweak.cloneaxisZ << 2));
    m_resources->initScene(m_scene);
  }

  if(shadersChanged || sceneChanged || tweakChanged(m_tweak.renderer) || tweakChanged(m_tweak.config.sorted)
     || tweakChanged(m_tweak.percent) || tweakChanged(m_tweak.config.passthrough)
     || tweakChanged(m_tweak.config.searchBatch) || tweakChanged(m_tweak.config.colorizeDraws)
     || tweakChanged(m_tweak.config.ignoreMaterials) || tweakChanged(m_tweak.config.globalSearchGuess)
     || tweakChanged(m_tweak.config.globalNaryN) || tweakChanged(m_tweak.config.globalNaryMin)
     || tweakChanged(m_tweak.config.globalNaryMaxIter) || tweakChanged(m_tweak.config.perDrawParameterMode))
  {
    m_resources->synchronize();
    initRenderer(m_tweak.renderer);
  }

  m_resources->beginFrame();

  if(tweakChanged(m_tweak.animation))
  {
    m_resources->synchronize();
    m_resources->animationReset();

    m_animBeginTime = time;
  }

  {
    m_shared.winWidth  = width;
    m_shared.winHeight = height;

    SceneData& sceneUbo = m_shared.sceneUbo;

    sceneUbo.viewport = ivec2(width, height);

    glm::mat4 projection = glm::perspectiveRH_ZO(glm::radians(45.f), float(width) / float(height),
                                            m_control.m_sceneDimension * 0.001f, m_control.m_sceneDimension * 10.0f);
    projection[1][1] *= -1;
    glm::mat4 view = m_control.m_viewMatrix;

    if(m_tweak.animation && m_tweak.animationSpin)
    {
      double animTime = (time - m_animBeginTime) * 0.3 + glm::pi<float>() * 0.2;
      vec3   dir      = vec3(cos(animTime), 1, sin(animTime));
      view = glm::lookAt(m_control.m_sceneOrbit - (-dir * m_control.m_sceneDimension * 0.5f), m_control.m_sceneOrbit,
                         vec3(0, 1, 0));
    }

    sceneUbo.viewProjMatrix = projection * view;
    sceneUbo.viewMatrix     = view;
    sceneUbo.viewMatrixIT   = glm::transpose(glm::inverse(view));

    sceneUbo.viewPos = glm::row(sceneUbo.viewMatrixIT, 3);
    sceneUbo.viewDir = -glm::row(view, 2);

    sceneUbo.wLightPos   = glm::row(sceneUbo.viewMatrixIT, 3);
    sceneUbo.wLightPos.w = 1.0;

    sceneUbo.time       = float(time);
    sceneUbo.partWeight = m_tweak.partWeight;

    sceneUbo.mousePos = glm::ivec2(m_windowState.m_mouseCurrent[0], m_windowState.m_mouseCurrent[1]);
  }

  if(m_tweak.animation)
  {
    AnimationData& animUbo = m_shared.animUbo;
    animUbo.time           = float(time - m_animBeginTime);

    m_resources->animation(m_shared);
  }

  {
    m_renderer->draw(m_shared, m_renderStats);
  }

  {
    if(m_useUI)
    {
      ImGui::Render();
      m_shared.imguiDrawData = ImGui::GetDrawData();
    }
    else
    {
      m_shared.imguiDrawData = nullptr;
    }

    m_resources->blitFrame(m_shared);
  }

  m_resources->endFrame();
  m_resources->m_frame++;

  if(m_useUI)
  {
    ImGui::EndFrame();
  }

  m_lastTweak = m_tweak;
}

void Sample::resize(int width, int height)
{
  initFramebuffers(width, height);
}

void Sample::setRendererFromName()
{
  if(!m_rendererName.empty())
  {
    const Renderer::Registry registry = Renderer::getRegistry();
    for(size_t i = 0; i < m_renderersSorted.size(); i++)
    {
      if(strcmp(m_rendererName.c_str(), registry[m_renderersSorted[i]]->name()) == 0)
      {
        m_tweak.renderer = int(i);
      }
    }
  }
}

void Sample::setupConfigParameters()
{
  m_parameterList.addFilename(".csf", &m_modelFilename);
  m_parameterList.addFilename(".csf.gz", &m_modelFilename);
  m_parameterList.addFilename(".gltf", &m_modelFilename);

  m_parameterList.add("vkdevice", &m_contextInfo.compatibleDeviceIndex);

  m_parameterList.add("noui", &m_useUI, false);

  m_parameterList.add("renderer", (uint32_t*)&m_tweak.renderer);
  m_parameterList.add("renderernamed", &m_rendererName);
  m_parameterList.add("msaa", &m_tweak.msaa);
  m_parameterList.add("copies", &m_tweak.copies);
  m_parameterList.add("animation", &m_tweak.animation);
  m_parameterList.add("animationspin", &m_tweak.animationSpin);
  m_parameterList.add("minstatechanges", &m_tweak.config.sorted);
}

bool Sample::validateConfig()
{
  if(m_modelFilename.empty())
  {
    LOGI("no .csf model file specified\n");
    LOGI("exe <filename.csf/cfg> parameters...\n");
    m_parameterList.print();
    return false;
  }
  return true;
}

}  // namespace idraster

using namespace idraster;

int main(int argc, const char** argv)
{
  NVPSystem system(PROJECT_NAME);

#if defined(_WIN32) && defined(NDEBUG)
  //SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif

  Sample sample;
  {
    std::vector<std::string> directories;
    directories.push_back(NVPSystem::exePath());
    directories.push_back(NVPSystem::exePath() + "/media");
    directories.push_back(NVPSystem::exePath() + std::string(PROJECT_DOWNLOAD_RELDIRECTORY));
    sample.m_modelFilename = nvh::findFile(std::string("worldcar_parts.csf"), directories);
  }

  return sample.run(PROJECT_NAME, argc, argv, SAMPLE_SIZE_WIDTH, SAMPLE_SIZE_HEIGHT);
}
