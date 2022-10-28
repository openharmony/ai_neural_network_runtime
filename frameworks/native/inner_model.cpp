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

#include "inner_model.h"

#include <new>
#include <unordered_map>
#include <vector>

#include "securec.h"

#include "common/utils.h"
#include "common/scoped_trace.h"
#include "device_manager.h"
#include "hdi_device.h"
#include "validation.h"
#include "ops_builder.h"
#include "ops_registry.h"
#include "transform.h"

namespace MSLITE = mindspore::lite;

namespace OHOS {
namespace NeuralNetworkRuntime {
const std::string NNR_MODEL = "NNR_Model";
const std::string LOADED_NNR_MODEL = "Loaded_NNR_Model";

namespace {
class LiteGraphDeleter {
public:
    void operator()(MSLITE::LiteGraph* liteGraph) const
    {
        MindIR_LiteGraph_Destroy(&liteGraph);
    }
};

std::shared_ptr<NNTensor> ConstructNNTensorFromLiteGraphTensor(const MSLITE::TensorPtr msTensor)
{
    MSLITE::DataType msDataType = MSLITE::MindIR_Tensor_GetDataType(msTensor);
    OH_NN_DataType dataType = MSToNN::TransformDataType(msDataType);
    std::vector<int32_t> msDims = MSLITE::MindIR_Tensor_GetDims(msTensor);
    std::vector<MSLITE::QuantParam> msQuantParams = MSLITE::MindIR_Tensor_GetQuantParams(msTensor);
    std::vector<QuantParam> nnQuantParams = MSToNN::TransformQuantParams(msQuantParams);

    std::shared_ptr<NNTensor> nnTensor = CreateSharedPtr<NNTensor>();
    if (nnTensor == nullptr) {
        LOGE("ConstructNNTensorFromLiteGraphTensor failed, error happened when creating NNTensor.");
        return nullptr;
    }

    OH_NN_ReturnCode ret = nnTensor->Build(dataType, msDims, nnQuantParams, OH_NN_TENSOR);
    if (ret != OH_NN_SUCCESS) {
        LOGE("ConstructNNTensorFromLiteGraphTensor failed, error happened when building NNTensor with attributes.");
        return nullptr;
    }

    return nnTensor;
}

OH_NN_ReturnCode ConstructNNTensorsFromLiteGraph(const MSLITE::LiteGraph* liteGraph,
                                                 const std::vector<uint32_t>& indices,
                                                 std::vector<std::shared_ptr<NNTensor>>& nnTensors)
{
    if (indices.empty()) {
        LOGE("ConstructNNTensorsFromLiteGraph failed, passed empty indices list.");
        return OH_NN_INVALID_PARAMETER;
    }

    uint32_t maximumIndex = *(std::max_element(indices.begin(), indices.end()));
    if (maximumIndex >= liteGraph->all_tensors_.size()) {
        LOGE("ConstructNNTensorsFromLiteGraph failed, index exceed size of all_tensors inside liteGraph.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::shared_ptr<NNTensor> nnTensor;
    for (uint32_t i : indices) {
        nnTensor = ConstructNNTensorFromLiteGraphTensor(liteGraph->all_tensors_[i]);
        if (nnTensor == nullptr) {
            LOGE("ConstructNNTensorsFromLiteGraph failed, failed to construct NNTensor from LiteGraphTensor.");
            return OH_NN_NULL_PTR;
        }

        nnTensors.emplace_back(nnTensor);
    }

    return OH_NN_SUCCESS;
}
} // anonymous namespace

InnerModel::InnerModel() {}

bool InnerModel::IsBuild() const
{
    return (m_liteGraph != nullptr);
}

OH_NN_ReturnCode InnerModel::BuildFromLiteGraph(const MSLITE::LiteGraph* liteGraph)
{
    NNRT_TRACE_NAME("Build model from lite graph");
    if (liteGraph == nullptr) {
        LOGE("BuildFromLiteGraph failed, passed empty liteGraph.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_liteGraph != nullptr) {
        LOGE("BuildFromLiteGraph failed, liteGraph has been built or loaded before.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!m_allTensors.empty() || !m_ops.empty()) {
        LOGE("BuildFromLiteGraph failed, please LoadLiteGraph without adding tensor and operations.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    m_inputTensors.clear();
    OH_NN_ReturnCode ret = ConstructNNTensorsFromLiteGraph(liteGraph, liteGraph->input_indices_, m_inputTensors);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildFromLiteGraph failed, error happened when constructing input NNTensors from liteGraph.");
        return ret;
    }

    m_outputTensors.clear();
    ret = ConstructNNTensorsFromLiteGraph(liteGraph, liteGraph->output_indices_, m_outputTensors);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildFromLiteGraph failed, error happened when constructing output NNTensors from liteGraph.");
        return ret;
    }

    m_liteGraph.reset(const_cast<MSLITE::LiteGraph*>(liteGraph), LiteGraphDeleter());
    m_liteGraph->name_ = LOADED_NNR_MODEL;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::AddTensor(const OH_NN_Tensor& nnTensor)
{
    if (m_liteGraph != nullptr) {
        LOGE("AddTensor failed, AddTensor is forbidden after Finish() or LoadLiteGraph() has been called.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    std::shared_ptr<NNTensor> tensor = CreateSharedPtr<NNTensor>();
    if (tensor == nullptr) {
        LOGE("AddTensor failed, error happened when creating NNTensor.");
        return OH_NN_MEMORY_ERROR;
    }

    OH_NN_ReturnCode ret = tensor->BuildFromOHNNTensor(nnTensor);
    if (ret != OH_NN_SUCCESS) {
        LOGE("AddTensor failed, error happened when build NNTensor from OH_NN_Tensor.");
        return ret;
    }

    // The NNTensor is named as "Tensor: <tensor index>"".
    tensor->SetName("Tensor: " + std::to_string(m_allTensors.size()));
    m_allTensors.emplace_back(tensor);

    return OH_NN_SUCCESS;
}

// DOTO: 圈复杂度待优化
OH_NN_ReturnCode InnerModel::SetTensorValue(uint32_t index, const void* buffer, size_t length)
{
    if (m_liteGraph != nullptr) {
        LOGE("SetTensorValue failed, SetTensorValue is forbidden after Finish() or LoadLiteGraph() has been called.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (index >= m_allTensors.size()) {
        LOGE("SetTensorValue failed, passed index %u out of the number of added tensors.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    const std::shared_ptr<NNTensor> tensor = m_allTensors[index];
    if (tensor->GetBuffer() != nullptr) {
        LOGE("SetTensorValue failed, tensor has been set value twice. Tensor index: %u.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    if (buffer == nullptr) {
        LOGW("SetTensorValue passed empty buffer, which makes no effect.");
        return OH_NN_SUCCESS;
    }

    if (tensor->IsDynamicShape()) {
        LOGE("SetTensorValue failed, cannot set value to tensor with dynamic shape.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (length != tensor->GetDataLength()) {
        LOGE("SetTensorValue failed, get buffer length %zu different from the byte size of tensor %zu.",
             length, tensor->GetDataLength());
        return OH_NN_INVALID_PARAMETER;
    }

    // Data will be released inside NNTensor if it is set inside NNTensor using SetBuffer().
    void* data = new (std::nothrow) char[length];
    if (data == nullptr) {
        LOGE("SetTensorValue failed, please check whether it runs out of memory.");
        return OH_NN_MEMORY_ERROR;
    }

    errno_t ret = memcpy_s(data, length, buffer, length);
    if (ret != EOK) {
        LOGE("SetTensorValue failed, please the information of error number %d from memcpy_s.", ret);
        delete [] reinterpret_cast<char*>(data);
        return OH_NN_FAILED;
    }

    tensor->SetBuffer(data, length);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::ValidateInputAndOutput(
    const OH_NN_UInt32Array& inputIndices, const OH_NN_UInt32Array& outputIndices) const
{
    OH_NN_ReturnCode ret = ValidateTensorArray(inputIndices);
    if (ret != OH_NN_SUCCESS) {
        LOGE("ValidateInputAndOutput failed, please check input indices.");
        return ret;
    }

    ret = ValidateTensorArray(outputIndices);
    if (ret != OH_NN_SUCCESS) {
        LOGE("ValidateInputAndOutput failed, please check output indices.");
        return ret;
    }

    if (inputIndices.size == 0) {
        LOGE("ValidateInputAndOutput failed, passed empty input indices.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (outputIndices.size == 0) {
        LOGE("ValidateInputAndOutput failed, passed empty output indices.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::shared_ptr<NNTensor> tensor{nullptr};
    for (uint32_t i = 0; i < inputIndices.size; i++) {
        tensor = m_allTensors[inputIndices.data[i]];
        if (tensor->GetType() != OH_NN_TENSOR) {
            LOGE("ValidateInputAndOutput failed, tensor set as input should has type of OH_NN_TENSOR, but receive %d."
                 "Tensor index: %u.", tensor->GetType(), i);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    for (uint32_t i = 0; i < outputIndices.size; i++) {
        tensor = m_allTensors[outputIndices.data[i]];
        if (tensor->GetType() != OH_NN_TENSOR) {
            LOGE("ValidateInputAndOutput failed, tensor set as output should has type of OH_NN_TENSOR, but receive %d."
                 "Tensor index: %u.", tensor->GetType(), i);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    // The number of inputIndices and outputIndices are usually small, so O(n**2) iteration is fine.
    for (uint32_t i = 0; i < inputIndices.size; i++) {
        for (uint32_t j = 0; j < outputIndices.size; j++) {
            if (inputIndices.data[i] == outputIndices.data[j]) {
                LOGE("ValidateInputAndOutput failed, should not set an tensor as input and output at the same time, "
                     "input index %u, output index %u", inputIndices.data[i], outputIndices.data[j]);
                return OH_NN_INVALID_PARAMETER;
            }
        }
    }
    return OH_NN_SUCCESS;
}

/* Check whether the indices exceed the number of added tensors. */
OH_NN_ReturnCode InnerModel::ValidateTensorArray(const OH_NN_UInt32Array& indices) const
{
    OH_NN_ReturnCode ret = Validation::ValidateArray(indices.data, indices.size);
    if (ret != OH_NN_SUCCESS) {
        LOGE("ValidateTensorArray failed, please check the validity of indices.");
        return ret;
    }

    for (uint32_t i = 0; i < indices.size; i++) {
        if (indices.data[i] >= m_allTensors.size()) {
            LOGE("ValidateTensors failed, index %u is out of the number of added tensors.", indices.data[i]);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::AddOperation(OH_NN_OperationType opType, const OH_NN_UInt32Array& paramIndices,
                                          const OH_NN_UInt32Array& inputIndices, const OH_NN_UInt32Array& outputIndices)
{
    if (m_liteGraph != nullptr) {
        LOGE("AddOperation failed, AddOperation is forbidden after after Finish() or LoadLiteGraph() has been called.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode ret = ValidateInputAndOutput(inputIndices, outputIndices);
    if (ret != OH_NN_SUCCESS) {
        LOGE("AddOperation failed, please check inputIndices and outputIndices.");
        return ret;
    }
    std::vector<uint32_t> inputs = ConstructVectorFromArray(inputIndices.data, inputIndices.size);
    std::vector<uint32_t> outputs = ConstructVectorFromArray(outputIndices.data, outputIndices.size);

    ret = ValidateTensorArray(paramIndices);
    if (ret != OH_NN_SUCCESS) {
        LOGE("AddOperation failed, please check paramIndices.");
        return ret;
    }
    std::vector<uint32_t> parameters = ConstructVectorFromArray(paramIndices.data, paramIndices.size);

    Ops::OpsRegistry& opsRegistry = Ops::OpsRegistry::GetSingleton();
    std::unique_ptr<Ops::OpsBuilder> opsBuilder = opsRegistry.GetOpsBuilder(opType);
    if (opsBuilder == nullptr) {
        LOGE("AddOperation failed, cannot add operation of type: %d.", opType);
        return OH_NN_INVALID_PARAMETER;
    }

    ret = opsBuilder->Build(parameters, inputs, outputs, m_allTensors);
    if (ret != OH_NN_SUCCESS) {
        LOGE("AddOperation failed, error happens when build operations.");
        return ret;
    }

    m_ops.emplace_back(std::move(opsBuilder));
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::SpecifyInputsAndOutputs(
    const OH_NN_UInt32Array& inputIndices, const OH_NN_UInt32Array& outputIndices)
{
    if (m_liteGraph != nullptr) {
        LOGE("SpecifyInputsAndOutputs failed, "
             "SpecifyInputsAndOutputs is forbidden after Finish() or LoadLiteGraph() has been called.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!m_inputTensors.empty()) {
        LOGE("SpecifyInputsAndOutputs failed, SpecifyInputsAndOutputs should not be called twice.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode ret = ValidateInputAndOutput(inputIndices, outputIndices);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SpecifyInputsAndOutputs failed, please check inputIndices and outputIndices.");
        return ret;
    }

    m_inputIndices = ConstructVectorFromArray(inputIndices.data, inputIndices.size);
    m_outputIndices = ConstructVectorFromArray(outputIndices.data, outputIndices.size);

    for (uint32_t i : m_inputIndices) {
        m_inputTensors.emplace_back(m_allTensors[i]);
    }

    for (uint32_t i : m_outputIndices) {
        m_outputTensors.emplace_back(m_allTensors[i]);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::Build()
{
    NNRT_TRACE_NAME("Build model");
    if (m_liteGraph != nullptr) {
        LOGE("Build failed,"
             " OH_NNModel is not allowed to build again after Build() or BuildFromLiteGraph() has been called.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_allTensors.empty()) {
        LOGE("Build failed, no OH_NN_Tensor has been added. Must call AddTensor before Build().");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_ops.empty()) {
        LOGE("Build failed, no operation has beed added. Must call AddOperation before Build().");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if ((m_inputIndices.empty()) || (m_outputIndices.empty())) {
        LOGE("Build failed, inputs and outputs are unspecified. Must call SpecifyInputsAndOutputs before Build().");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    MSLITE::LiteGraph* pLiteGraph = new (std::nothrow) MSLITE::LiteGraph();
    if (pLiteGraph == nullptr) {
        LOGE("Build failed, error happend when creating LiteGraph.");
        return OH_NN_MEMORY_ERROR;
    }
    m_liteGraph.reset(pLiteGraph, LiteGraphDeleter());

    m_liteGraph->name_ = NNR_MODEL;

    std::unordered_map<uint32_t, uint32_t> modelIDToGraphID;
    AddTensorsToLiteGraph(modelIDToGraphID);

    OH_NN_ReturnCode ret = AddNodesToLiteGraph(modelIDToGraphID);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // subGraph will be released by LiteGraph if it is added into instance of LiteGraph.
    MSLITE::LiteGraph::SubGraph* subGraph = new (std::nothrow) MSLITE::LiteGraph::SubGraph();
    if (subGraph == nullptr) {
        LOGE("AddNodesToLiteGraph failed, error happened when creating subgraph.");
        return OH_NN_NULL_PTR;
    }

    subGraph->name_ = "NNRt_SubGraph"; // Name of subGraph
    subGraph->input_indices_ = m_liteGraph->input_indices_;
    subGraph->output_indices_ = m_liteGraph->output_indices_;
    uint32_t nodeCount = static_cast<uint32_t>(m_ops.size()); // m_ops.size() smaller than UINT32_MAX
    for (uint32_t i = 0; i < nodeCount; i++) {
        subGraph->node_indices_.emplace_back(i);
    }
    m_liteGraph->sub_graphs_.emplace_back(subGraph);

    return OH_NN_SUCCESS;
}

void InnerModel::AddTensorsToLiteGraph(std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID)
{
    uint32_t graphID = 0;
    LiteGraphTensorPtr tensor(nullptr, DestroyLiteGraphTensor);
    size_t tensorCount = m_allTensors.size();
    for (size_t i = 0; i < tensorCount; i++) {
        const std::shared_ptr<NNTensor>& nnTensor = m_allTensors[i];
        // If the tensor is used as operation parameter, it will not convert to the tensor of LiteGraph.
        if (nnTensor->IsOpParameter()) {
            continue;
        }

        tensor = nnTensor->ConvertToLiteGraphTensor();
        m_liteGraph->all_tensors_.emplace_back(tensor.release());
        modelIDToGraphID[i] = graphID++;
    }

    // Note: Indices in m_inputIndices and m_outputIndices have been checked in SpecifyInputAndOutput(), there is no
    // need to check twice.
    std::vector<uint32_t>& inputIndices = m_liteGraph->input_indices_;
    for (uint32_t index : m_inputIndices) {
        inputIndices.emplace_back(modelIDToGraphID.at(index));
    }

    std::vector<uint32_t>& outputIndices = m_liteGraph->output_indices_;
    for (uint32_t index : m_outputIndices) {
        outputIndices.emplace_back(modelIDToGraphID.at(index));
    }
}

OH_NN_ReturnCode InnerModel::AddNodesToLiteGraph(const std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID)
{
    MSLITE::LiteGraph::Node* node{nullptr};
    size_t opCount = m_ops.size();
    Ops::LiteGraphPrimitvePtr primitive = {nullptr, DestroyLiteGraphTensor};
    for (size_t i = 0; i < opCount; i++) {
        std::unique_ptr<Ops::OpsBuilder>& op = m_ops[i];
        // node will be released by LiteGraph if it is added into instance of LiteGraph.
        node = new(std::nothrow) MSLITE::LiteGraph::Node();
        if (node == nullptr) {
            LOGE("AddNodesToLiteGraph failed, error happened when creating LiteGraph tensor.");
            return OH_NN_NULL_PTR;
        }

        node->name_ = op->GetName() + ":" + std::to_string(i);
        node->quant_type_ = NNToMS::TransformQuantType(op->GetQuantType());

        op->GetInputIndex(node->input_indices_, modelIDToGraphID);
        op->GetOutputIndex(node->output_indices_, modelIDToGraphID);

        primitive = op->GetPrimitive();
        if (primitive == nullptr) {
            LOGE("Build %s primitive failed.", op->GetName().c_str());
            delete node;
            return OH_NN_FAILED;
        }

        node->primitive_ = primitive.release();
        m_liteGraph->all_nodes_.emplace_back(node);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::GetSupportedOperations(size_t deviceID, const bool** isSupported, uint32_t& opCount)
{
    if (m_liteGraph == nullptr) {
        LOGE("GetSupportedOperations failed. GetSupportedOperations() must be called after Finish().");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    DeviceManager& deviceManager = DeviceManager::GetInstance();

    std::shared_ptr<Device> device = deviceManager.GetDevice(deviceID);
    if (device == nullptr) {
        LOGE("GetSupportedOperations failed, retrieve device failed.");
        return OH_NN_FAILED;
    }

    std::vector<bool> supportedOperations;
    OH_NN_ReturnCode ret = device->GetSupportedOperation(m_liteGraph, supportedOperations);
    if (ret != OH_NN_SUCCESS) {
        LOGE("GetSupportedOperations failed, error happened when get supported operations from devices.");
        return ret;
    }

    m_supportedOperations.clear();
    std::copy(supportedOperations.begin(), supportedOperations.end(), std::back_inserter(m_supportedOperations));

    *isSupported = reinterpret_cast<bool*>(m_supportedOperations.data());
    opCount = m_supportedOperations.size();

    return OH_NN_SUCCESS;
}

std::shared_ptr<MSLITE::LiteGraph> InnerModel::GetLiteGraphs() const
{
    return m_liteGraph;
}

std::vector<std::shared_ptr<NNTensor>> InnerModel::GetInputTensors() const
{
    return m_inputTensors;
}

std::vector<std::shared_ptr<NNTensor>> InnerModel::GetOutputTensors() const
{
    return m_outputTensors;
}
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS