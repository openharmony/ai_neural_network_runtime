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
#include <memory>

namespace OHOS {
namespace NeuralNetworkRuntime {
class NNRtServiceApi {
public:
    static NNRtServiceApi& GetInstance();
    bool IsServiceAvaliable() const;

    int (*CheckModelSizeFromPath)(const char* path, bool& exceedLimit) = nullptr;
    int (*CheckModelSizeFromBuffer)(const void* buffer, size_t size, bool& exceedLimit) = nullptr;
    int (*CheckModelSizeFromModel)(void* model, bool& exceedLimit) = nullptr;
    size_t (*GetNNRtModelIDFromPath)(const char*) = nullptr;
    size_t (*GetNNRtModelIDFromBuffer)(const void* buffer, size_t size) = nullptr;
    size_t (*GetNNRtModelIDFromModel)(void* model) = nullptr;
    int (*SetModelID)(int callingPid, uint32_t hiaimodelID, size_t nnrtModelID) = nullptr;
    int (*IsSupportAuthentication)(bool* supportStat) = nullptr;
    int (*IsSupportScheduling)(bool* supportStat) = nullptr;
    int (*Authentication)(int callingPid) = nullptr;
    int (*Scheduling)(uint32_t hiaiModelId, bool* needModelLatency) = nullptr;
    int (*UpdateModelLatency)(uint32_t hiaiModelId, int modelLatency) = nullptr;
    int (*Unload)(uint32_t hiaiModelId) = nullptr;

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