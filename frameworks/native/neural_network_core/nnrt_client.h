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

#ifndef NEURAL_NETWORK_RUNTIME_CLIENT_H
#define NEURAL_NETWORK_RUNTIME_CLIENT_H

#include <cstddef>
#include "executor.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class NNRtServiceApi {
public:
    static NNRtServiceApi& GetInstance();
    bool IsServiceAvaliable() const;

    int (*CheckModelSizeFromPath)(const char* path, bool& exceedLimit) = nullptr;
    int (*CheckModelSizeFromCache)(const char* path, const std::string& modelName, bool& exceedLimit) = nullptr;
    int (*CheckModelSizeFromBuffer)(const void* buffer, size_t size, bool& exceedLimit) = nullptr;
    int (*CheckModelSizeFromModel)(void* model, bool& exceedLimit) = nullptr;
    size_t (*GetNNRtModelIDFromPath)(const char*) = nullptr;
    size_t (*GetNNRtModelIDFromCache)(const char* path, const char* modelName) = nullptr;
    size_t (*GetNNRtModelIDFromBuffer)(const void* buffer, size_t size) = nullptr;
    size_t (*GetNNRtModelIDFromModel)(void* model) = nullptr;
    int (*SetModelID)(uint32_t hiaimodelID, size_t nnrtModelID) = nullptr;
    int (*IsSupportAuthentication)(bool* supportStat) = nullptr;
    int (*IsSupportScheduling)(bool* supportStat) = nullptr;
    int (*Authentication)() = nullptr;
    int (*Scheduling)(uint32_t hiaiModelId, bool* needModelLatency, const char* cachePath) = nullptr;
    int (*UpdateModelLatency)(uint32_t hiaiModelId, int modelLatency) = nullptr;
    int (*Unload)(uint32_t hiaiModelId) = nullptr;
    bool (*PullUpDlliteService)() = nullptr;
    int (*AutoReinitSetModelID)(uint32_t hiaimodelID, size_t nnrtModelID) = nullptr;
    int (*AutoReinitScheduling)(uint32_t originHiaimodelID, uint32_t hiaiModelId,
        bool* needModelLatency, const char* cachePath) = nullptr;
    int (*AutoUnload)(uint32_t originHiaimodelID, uint32_t hiaiModelId) = nullptr;
    int (*SetDeinitModelCallBack)(uint32_t hiaiModelId, OHOS::NeuralNetworkRuntime::Executor* callback) = nullptr;
    int (*UnSetDeinitModelCallBack)(uint32_t hiaiModelId) = nullptr;
private:
    bool m_serviceAvailable = false;
    NNRtServiceApi() = default;
    NNRtServiceApi(const NNRtServiceApi&) = delete;
    NNRtServiceApi& operator=(const NNRtServiceApi&) = delete;
    virtual ~NNRtServiceApi();
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_CLIENT_H