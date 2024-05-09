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

#ifndef NEURAL_NETWORK_RUNTIME_NNEXECUTOR_H
#define NEURAL_NETWORK_RUNTIME_NNEXECUTOR_H

#include "executor.h"
#include "device.h"
#include "prepared_model.h"
#include "nn_tensor.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class NNExecutor : public Executor {
public:
    NNExecutor(size_t backendID,
               std::shared_ptr<Device> device,
               std::shared_ptr<PreparedModel> preparedModel,
               const std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& inputTensorDescs,
               const std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& outputTensorDescs);
    ~NNExecutor() override;

    OH_NN_ReturnCode GetInputDimRange(size_t inputIndex,
                                      size_t** minInputDims,
                                      size_t** maxInputDims,
                                      size_t* shapeNum) const override;
    OH_NN_ReturnCode GetOutputShape(uint32_t outputIndex, int32_t** shape, uint32_t* shapeNum) const override;

    size_t GetInputNum() const override;
    size_t GetOutputNum() const override;
    NN_TensorDesc* CreateInputTensorDesc(size_t index) const override;
    NN_TensorDesc* CreateOutputTensorDesc(size_t index) const override;

    OH_NN_ReturnCode SetOnRunDone(NN_OnRunDone onRunDone) override;
    OH_NN_ReturnCode SetOnServiceDied(NN_OnServiceDied onServiceDied) override;
    OH_NN_ReturnCode RunSync(NN_Tensor* inputTensors[],
                             size_t inputSize,
                             NN_Tensor* outputTensors[],
                             size_t outputSize) override;
    OH_NN_ReturnCode RunAsync(NN_Tensor* inputTensors[],
                              size_t inputSize,
                              NN_Tensor* outputTensors[],
                              size_t outputSize,
                              int32_t timeout,
                              void* userData) override;
    size_t GetBackendID() override;

    // The following APIs are compatible with older versions
    OH_NN_ReturnCode SetInput(uint32_t index, const OH_NN_Tensor& nnTensor, const void* buffer, size_t length);
    OH_NN_ReturnCode SetInputFromMemory(uint32_t index, const OH_NN_Tensor& nnTensor, const OH_NN_Memory& memory);
    OH_NN_ReturnCode SetOutput(uint32_t index, void* buffer, size_t length);
    OH_NN_ReturnCode SetOutputFromMemory(uint32_t index, const OH_NN_Memory& memory);

    OH_NN_ReturnCode CreateInputMemory(uint32_t index, size_t length, OH_NN_Memory** memory);
    OH_NN_ReturnCode CreateOutputMemory(uint32_t index, size_t length, OH_NN_Memory** memory);
    OH_NN_ReturnCode DestroyInputMemory(uint32_t index, OH_NN_Memory** memory);
    OH_NN_ReturnCode DestroyOutputMemory(uint32_t index, OH_NN_Memory** memory);

    OH_NN_ReturnCode Run();

private:
    OH_NN_ReturnCode GetInputDimVec() const;
    OH_NN_ReturnCode CheckInputDimRanges(NN_Tensor* inputTensors[], size_t inputSize);

    // The following APIs are compatible with older versions
    OH_NN_ReturnCode Run(const std::vector<std::shared_ptr<NNTensor>>& inputTensors,
                         std::vector<std::shared_ptr<NNTensor>>& outputTensors);
    bool CompareAttribute(
        const std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>& tensorDesc, const NNTensor& tensor) const;
    std::shared_ptr<NNTensor> BuildNNTensorFromDesc(
        const std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>& tensorDesc);
    OH_NN_ReturnCode BuildInputTensor(uint32_t index, const OH_NN_Tensor& nnTensor,
                                      std::shared_ptr<NNTensor> inputTensor) const;
    OH_NN_ReturnCode SetInputTensorWithCurrentBuffer(uint32_t index, std::shared_ptr<NNTensor> inputTensor,
                                                     const void* buffer, size_t dataLength, size_t curBufferLength);
    void SetInputTensorWithNewBuffer(uint32_t index, std::shared_ptr<NNTensor> inputTensor,
                                     const void* inputBuffer, size_t length, bool isInnerMem);
    OH_NN_ReturnCode CheckInputDimRanges(uint32_t index, const OH_NN_Tensor& nnTensor) const;

private:
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    // The following parameters are provided for compatibility with older versions
    struct ExeTensor {
        std::shared_ptr<NNTensor> tensor {nullptr};
        void* userBuffer {nullptr};
        size_t userBufferLength {0};
        bool isInnerMem {false};
    };
    bool m_isRun {false};
    std::unordered_map<int, ExeTensor> m_inputTensors;
    std::unordered_map<int, ExeTensor> m_outputTensors;
    std::unordered_map<int, std::vector<void*>> m_inputCreatedMem;
    std::unordered_map<int, std::vector<void*>> m_outputCreatedMem;
    mutable std::vector<std::vector<size_t>> m_minInputDimsVec;
    mutable std::vector<std::vector<size_t>> m_maxInputDimsVec;
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif  // NEURAL_NETWORK_RUNTIME_NNEXECUTOR_H
