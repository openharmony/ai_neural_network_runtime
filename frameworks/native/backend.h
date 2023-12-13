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

#ifndef NEURAL_NETWORK_CORE_BACKEND_H
#define NEURAL_NETWORK_CORE_BACKEND_H

#include <string>
#include <memory>

#include "compilation.h"
#include "compiler.h"
#include "executor.h"
#include "tensor.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class Backend {
public:
    Backend() = default;
    virtual ~Backend() = default;

    virtual size_t GetBackendID() const = 0;
    virtual OH_NN_ReturnCode GetBackendName(std::string& name) const = 0;
    virtual OH_NN_ReturnCode GetBackendType(OH_NN_DeviceType& backendType) const = 0;
    virtual OH_NN_ReturnCode GetBackendStatus(DeviceStatus& status) const = 0;

    virtual Compiler* CreateCompiler(Compilation* compilation) = 0;
    virtual OH_NN_ReturnCode DestroyCompiler(Compiler* compiler) = 0;

    virtual Executor* CreateExecutor(Compilation* compilation) = 0;
    virtual OH_NN_ReturnCode DestroyExecutor(Executor* executor) = 0;

    virtual Tensor* CreateTensor(TensorDesc* desc) = 0;
    virtual OH_NN_ReturnCode DestroyTensor(Tensor* tensor) = 0;
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif  // NEURAL_NETWORK_CORE_BACKEND_H
