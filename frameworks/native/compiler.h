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

#ifndef NEURAL_NETWORK_RUNTIME_COMPILER_H
#define NEURAL_NETWORK_RUNTIME_COMPILER_H

#include <memory>
#include <unordered_map>

#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"
#include "cpp_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class Compiler {
public:
    Compiler() = default;
    virtual ~Compiler() = default;

    virtual size_t GetBackendID() const = 0;

    virtual OH_NN_ReturnCode SetCacheDir(const std::string& cacheModelPath, uint32_t version) = 0;
    virtual OH_NN_ReturnCode SetPerformance(OH_NN_PerformanceMode performance) = 0;
    virtual OH_NN_ReturnCode SetPriority(OH_NN_Priority priority) = 0;
    virtual OH_NN_ReturnCode SetEnableFp16(bool isFp16) = 0;

    virtual bool IsBuild() const = 0;
    virtual OH_NN_ReturnCode Build() = 0;

    virtual OH_NN_ReturnCode SaveToCacheFile() const = 0;
    virtual OH_NN_ReturnCode RestoreFromCacheFile() = 0;
    virtual OH_NN_ReturnCode SaveToCacheBuffer(const void* buffer, size_t length, size_t* modelSize) const = 0;
    virtual OH_NN_ReturnCode RestoreFromCacheBuffer(const void* buffer, size_t length) = 0;

    virtual OH_NN_ReturnCode SetExtensionConfig(const std::unordered_map<std::string, std::vector<char>>& configs) = 0;
    virtual OH_NN_ReturnCode SetOptions(const std::vector<std::shared_ptr<void>>& options) = 0;
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_COMPILER_H