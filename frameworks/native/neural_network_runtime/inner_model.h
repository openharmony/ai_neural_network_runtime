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

#ifndef NEURAL_NETWORK_RUNTIME_INNER_MODEL_H
#define NEURAL_NETWORK_RUNTIME_INNER_MODEL_H

#include <memory>
#include <unordered_map>

#include "mindir.h"
#include "ops_builder.h"
#include "tensor_desc.h"
#include "interfaces/innerkits/c/neural_network_runtime_inner.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class InnerModel {
public:
    InnerModel();

    bool IsBuild() const;
    OH_NN_ReturnCode BuildFromLiteGraph(const mindspore::lite::LiteGraph* liteGraph, const Buffer& quantBuffer,
                                        const std::string& modelName, const std::string& isProfiling,
                                        std::map<std::string, std::string>& opLayouts);
    OH_NN_ReturnCode BuildFromMetaGraph(const void* metaGraph, const Buffer& quantBuffer,
                                        const std::string& modelName, const std::string& isProfiling,
                                        std::map<std::string, std::string>& opLayouts);
    OH_NN_ReturnCode AddTensor(const OH_NN_Tensor& nnTensor);
    OH_NN_ReturnCode AddTensorDesc(const NN_TensorDesc* nnTensorDesc);
    OH_NN_ReturnCode SetTensorQuantParam(uint32_t index, const NN_QuantParam* quantParam);
    OH_NN_ReturnCode SetTensorType(uint32_t index, OH_NN_TensorType tensorType);
    OH_NN_ReturnCode SetTensorValue(uint32_t index, const void* buffer, size_t length);
    OH_NN_ReturnCode AddOperation(OH_NN_OperationType opType,
                                  const OH_NN_UInt32Array& paramIndices,
                                  const OH_NN_UInt32Array& inputIndices,
                                  const OH_NN_UInt32Array& outputIndices);
    OH_NN_ReturnCode GetSupportedOperations(size_t deviceID, const bool** isSupported, uint32_t& opCount);
    OH_NN_ReturnCode SpecifyInputsAndOutputs(
        const OH_NN_UInt32Array& inputIndices, const OH_NN_UInt32Array& outputIndices);
    OH_NN_ReturnCode SetInputsAndOutputsInfo(const OH_NN_TensorInfo* inputsInfo, size_t inputSize,
        const OH_NN_TensorInfo* outputsInfo, size_t outputSize);
    OH_NN_ReturnCode Build();
    std::vector<std::shared_ptr<NNTensor>> GetInputTensors() const;
    std::vector<std::shared_ptr<NNTensor>> GetOutputTensors() const;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> GetInputTensorDescs() const;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> GetOutputTensorDescs() const;
    std::shared_ptr<mindspore::lite::LiteGraph> GetLiteGraphs() const;
    void* GetMetaGraph() const;
    Buffer GetQuantBuffer() const;
    std::string GetModelName() const;
    std::string GetProfiling() const;
    std::map<std::string, std::string> GetOpLayouts() const;
    TuningStrategy GetTuningStrategy() const;
    void SetTuningStrategy(const TuningStrategy tuningStrategy);

private:
    void AddTensorsToLiteGraph(std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID);
    OH_NN_ReturnCode AddNodesToLiteGraph(const std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID);
    OH_NN_ReturnCode ValidateInputAndOutput(
        const OH_NN_UInt32Array& inputIndices, const OH_NN_UInt32Array& outputIndices) const;
    OH_NN_ReturnCode ValidateTensorArray(const OH_NN_UInt32Array& indices) const;
    OH_NN_ReturnCode CheckParameters() const;

private:
    std::vector<char> m_supportedOperations; // std::vector<bool> not support data(), use std::vector<char> instead.
    std::vector<uint32_t> m_inputIndices;
    std::vector<uint32_t> m_outputIndices;
    std::vector<std::unique_ptr<Ops::OpsBuilder>> m_ops;
    std::vector<std::shared_ptr<NNTensor>> m_allTensors;
    std::vector<std::shared_ptr<NNTensor>> m_inputTensors; // Used to pass input tensors to compilation.
    std::vector<std::shared_ptr<NNTensor>> m_outputTensors; // Used to pass output tensors to compilation.
    std::shared_ptr<mindspore::lite::LiteGraph> m_liteGraph {nullptr};
    void* m_metaGraph {nullptr};
    Buffer m_quantBuffer = {nullptr, 0};
    std::string m_modelName;
    std::string m_isProfiling;
    std::map<std::string, std::string> m_opLayouts;
    TuningStrategy m_tuningStrategy{TuningStrategy::OFF};
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_INNER_MODEL_H
