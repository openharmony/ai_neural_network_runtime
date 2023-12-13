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

#ifndef NEURAL_NETWORK_RUNTIME_COMPILATION_H
#define NEURAL_NETWORK_RUNTIME_COMPILATION_H

#include <vector>
#include <utility>
#include <memory>
#include <unordered_map>

#include "compiler.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
struct Compilation {
    size_t backendID {0};
    void* nnModel {nullptr};
    char* offlineModelPath {nullptr};
    std::pair<void*, size_t> offlineModelBuffer;
    char* cachePath {nullptr};
    uint32_t cacheVersion {0};
    std::pair<void*, size_t> cacheBuffer;
    OH_NN_Priority priority {OH_NN_PRIORITY_NONE};
    OH_NN_PerformanceMode performance {OH_NN_PERFORMANCE_NONE};
    bool enableFp16 {false};
    Compiler* compiler {nullptr};
    std::vector<std::shared_ptr<void>> options;
    std::unordered_map<std::string, std::vector<char>> configs;

    ~Compilation()
    {
        options.clear();
    }
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_COMPILATION_H