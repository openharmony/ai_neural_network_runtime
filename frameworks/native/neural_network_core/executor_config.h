/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
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

#ifndef NEURAL_NETWORK_RUNTIME_EXECUTOR_CONFIG_H
#define NEURAL_NETWORK_RUNTIME_EXECUTOR_CONFIG_H

namespace OHOS {
namespace NeuralNetworkRuntime {
struct ExecutorConfig {
    bool isNeedModelLatency {false};
    int32_t callingPid {-1};
    uint32_t hiaiModelId {0};
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_EXECUTOR_CONFIG_H