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

#ifndef NEURAL_NETWORK_RUNTIME_EXECUTOR_H
#define NEURAL_NETWORK_RUNTIME_EXECUTOR_H

#include <string>
#include <memory>

#include "compiler.h"
#include "tensor_desc.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class Executor {
public:
    Executor() = default;
    Executor(Compiler* compiler);
    virtual ~Executor() = default;

    virtual OH_NN_ReturnCode GetInputDimRange(size_t inputIndex,
                                              size_t** minInputDims,
                                              size_t** maxInputDims,
                                              size_t* shapeNum) const = 0;
    virtual OH_NN_ReturnCode GetOutputShape(uint32_t outputIndex, int32_t** shape, uint32_t* shapeNum) const = 0;

    virtual size_t GetInputNum() const = 0;
    virtual size_t GetOutputNum() const = 0;
    virtual NN_TensorDesc* CreateInputTensorDesc(size_t index) const = 0;
    virtual NN_TensorDesc* CreateOutputTensorDesc(size_t index) const = 0;

    virtual OH_NN_ReturnCode SetOnRunDone(NN_OnRunDone onRunDone) = 0;
    virtual OH_NN_ReturnCode SetOnServiceDied(NN_OnServiceDied onServiceDied) = 0;
    virtual OH_NN_ReturnCode RunSync(NN_Tensor* inputTensors[],
                                     size_t inputSize,
                                     NN_Tensor* outputTensors[],
                                     size_t outputSize) = 0;
    virtual OH_NN_ReturnCode RunAsync(NN_Tensor* inputTensors[],
                                      size_t inputSize,
                                      NN_Tensor* outputTensors[],
                                      size_t outputSize,
                                      int32_t timeout,
                                      void* userData) = 0;
    virtual size_t GetBackendID() = 0;
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif  // NEURAL_NETWORK_RUNTIME_EXECUTOR_H
