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

#ifndef NEURAL_NETWORK_RUNTIME_EXECUTOR_H
#define NEURAL_NETWORK_RUNTIME_EXECUTOR_H

#include "compilation.h"
#include "execution_plan.h"
#include "nn_tensor.h"
#include "interfaces/kits/c/neural_network_runtime.h"
#include "device.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class Executor {
public:
    explicit Executor(const Compilation* compilation);
    ~Executor();

    OH_NN_ReturnCode SetInput(uint32_t index, const OH_NN_Tensor& nnTensor, const void* buffer, size_t length);
    OH_NN_ReturnCode SetInputFromMemory(uint32_t index, const OH_NN_Tensor& nnTensor, const OH_NN_Memory& memory);
    OH_NN_ReturnCode SetOutput(uint32_t index, void* buffer, size_t length);
    OH_NN_ReturnCode SetOutputFromMemory(uint32_t index, const OH_NN_Memory& memory);
    OH_NN_ReturnCode GetOutputShape(uint32_t index, int32_t** dimensions, uint32_t& dimensionCount);

    OH_NN_ReturnCode CreateInputMemory(uint32_t index, size_t length, OH_NN_Memory** memory);
    OH_NN_ReturnCode CreateOutputMemory(uint32_t index, size_t length, OH_NN_Memory** memory);
    OH_NN_ReturnCode DestroyInputMemory(uint32_t index, OH_NN_Memory** memory);
    OH_NN_ReturnCode DestroyOutputMemory(uint32_t index, OH_NN_Memory** memory);

    OH_NN_ReturnCode Run();

private:
    OH_NN_ReturnCode BuildInputTensor(uint32_t index, const OH_NN_Tensor& nnTensor,
                                      std::shared_ptr<NNTensor> inputTensor) const;
    OH_NN_ReturnCode SetInputTensorWithCurrentBuffer(uint32_t index, std::shared_ptr<NNTensor> inputTensor,
                                                     const void* buffer, size_t dataLength, size_t curBufferLength);
    void SetInputTensorWithNewBuffer(uint32_t index, std::shared_ptr<NNTensor> inputTensor,
                                     const void* inputBuffer, size_t length, bool isInnerMem);

private:
    struct ExeTensor {
        std::shared_ptr<NNTensor> tensor;
        void* userBuffer;
        size_t userBufferLength;
        bool isInnerMem;
    };
    bool m_isRun {false};
    std::vector<std::shared_ptr<NNTensor>> m_modelInputs;
    std::vector<std::shared_ptr<NNTensor>> m_modelOutputs;
    std::shared_ptr<ExecutionPlan> m_executionPlan {nullptr};
    std::unordered_map<int, std::vector<int32_t>> m_outputDimensions;
    std::unordered_map<int, ExeTensor> m_inputTensors;
    std::unordered_map<int, ExeTensor> m_outputTensors;
    std::unordered_map<int, std::vector<void*>> m_inputCreatedMem;
    std::unordered_map<int, std::vector<void*>> m_outputCreatedMem;
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif