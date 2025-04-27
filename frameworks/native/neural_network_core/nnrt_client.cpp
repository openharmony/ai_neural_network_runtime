/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
 */

#include "nnrt_client.h"

#include <dlfcn.h>
#include <string>

#include "log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
template<class T>
void LoadFunction(void* handle, const char* name, T* nnrtFunction)
{
    if (name == nullptr) {
        LOGE("LoadFunction error: the function %s does not exist.", name);
        return;
    }

    void* fn = dlsym(handle, name);
    if (fn == nullptr) {
        LOGE("LoadFunction error: unable to open function %{public}s", name);
        return;
    }

    *nnrtFunction = reinterpret_cast<T>(fn);
    return;
}

static void* libNNRtService = nullptr;

NNRtServiceApi& NNRtServiceApi::GetInstance()
{
    static NNRtServiceApi nnrtService;

    // Assumes there can be multiple instances of NN API
    std::string nnrtLibraryName = "libdllite_service_client.z.so";
    if (libNNRtService == nullptr) {
        libNNRtService = dlopen(nnrtLibraryName.c_str(), RTLD_LAZY | RTLD_NODELETE);
        if (libNNRtService == nullptr) {
            LOGE("LoadNNRtService error: unable to open library %{public}s", nnrtLibraryName.c_str());
            nnrtService.m_serviceAvailable = false;
            return nnrtService;
        }
    }

    LoadFunction(libNNRtService, "CheckModelSizeFromPath", &nnrtService.CheckModelSizeFromPath);
    LoadFunction(libNNRtService, "CheckModelSizeFromCache", &nnrtService.CheckModelSizeFromCache);
    LoadFunction(libNNRtService, "CheckModelSizeFromBuffer", &nnrtService.CheckModelSizeFromBuffer);
    LoadFunction(libNNRtService, "CheckModelSizeFromModel", &nnrtService.CheckModelSizeFromModel);
    LoadFunction(libNNRtService, "GetNNRtModelIDFromPath", &nnrtService.GetNNRtModelIDFromPath);
    LoadFunction(libNNRtService, "GetNNRtModelIDFromCache", &nnrtService.GetNNRtModelIDFromCache);
    LoadFunction(libNNRtService, "GetNNRtModelIDFromBuffer", &nnrtService.GetNNRtModelIDFromBuffer);
    LoadFunction(libNNRtService, "GetNNRtModelIDFromModel", &nnrtService.GetNNRtModelIDFromModel);
    LoadFunction(libNNRtService, "SetModelID", &nnrtService.SetModelID);
    LoadFunction(libNNRtService, "IsSupportAuthentication", &nnrtService.IsSupportAuthentication);
    LoadFunction(libNNRtService, "IsSupportScheduling", &nnrtService.IsSupportScheduling);
    LoadFunction(libNNRtService, "Authentication", &nnrtService.Authentication);
    LoadFunction(libNNRtService, "Scheduling", &nnrtService.Scheduling);
    LoadFunction(libNNRtService, "UpdateModelLatency", &nnrtService.UpdateModelLatency);
    LoadFunction(libNNRtService, "Unload", &nnrtService.Unload);
    LoadFunction(libNNRtService, "PullUpDlliteService", &nnrtService.PullUpDlliteService);

    nnrtService.m_serviceAvailable = true;
    return nnrtService;
}

bool NNRtServiceApi::IsServiceAvaliable() const
{
    return m_serviceAvailable;
}

NNRtServiceApi::~NNRtServiceApi()
{
    if (libNNRtService != nullptr) {
        dlclose(libNNRtService);
        libNNRtService = nullptr;
    }
}
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS