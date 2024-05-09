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
#include "backend_manager.h"
#include "validation.h"
#include "ops_builder.h"
#include "ops_registry.h"
#include "transform.h"
#include "nnbackend.h"

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
    OH_NN_Format nnFormat = MSToNN::TransformFormat(MSLITE::MindIR_Tensor_GetFormat(msTensor));

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

    nnTensor->SetFormat(nnFormat);

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
    return ((m_liteGraph != nullptr) || (m_metaGraph != nullptr));
}

OH_NN_ReturnCode InnerModel::BuildFromLiteGraph(const MSLITE::LiteGraph* liteGraph, const Buffer& quantBuffer,
    const std::string& modelName, const std::string& isProfiling, std::map<std::string, std::string>& opLayouts)
{
    NNRT_TRACE_NAME("Build model from lite graph");
    if (liteGraph == nullptr) {
        LOGE("BuildFromLiteGraph failed, passed empty liteGraph.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (IsBuild()) {
        LOGE("BuildFromLiteGraph failed, inner model has been built or loaded before.");
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

    m_quantBuffer = quantBuffer;
    m_modelName = modelName;
    m_isProfiling = isProfiling;
    m_opLayouts = opLayouts;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::BuildFromMetaGraph(
    const void* metaGraph, const Buffer& quantBuffer, const std::string& modelName, const std::string& isProfiling,
    std::map<std::string, std::string>& opLayouts)
{
    NNRT_TRACE_NAME("Build model from meta graph");
    if (metaGraph == nullptr) {
        LOGE("BuildFromMetaGraph failed, passed empty metaGraph.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (IsBuild()) {
        LOGE("BuildFromMetaGraph failed, inner model has been built or loaded before.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_allTensors.empty()) {
        LOGE("BuildFromMetaGraph failed, SetInputsAndOutputsInfo should be called before building metaGraph.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    m_metaGraph = const_cast<void*>(metaGraph);
    m_quantBuffer = quantBuffer;
    m_modelName = modelName;
    m_isProfiling = isProfiling;
    m_opLayouts = opLayouts;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::AddTensor(const OH_NN_Tensor& nnTensor)
{
    if (IsBuild()) {
        LOGE("AddTensor failed, AddTensor is forbidden after model has been built.");
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

OH_NN_ReturnCode InnerModel::AddTensorDesc(const NN_TensorDesc* nnTensorDesc)
{
    if (nnTensorDesc == nullptr) {
        LOGE("AddTensorDesc failed, passed nullptr to nnTensorDesc.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::shared_ptr<NNTensor> tensor = CreateSharedPtr<NNTensor>();
    if (tensor == nullptr) {
        LOGE("AddTensorDesc failed, error happened when creating NNTensor.");
        return OH_NN_MEMORY_ERROR;
    }

    OH_NN_ReturnCode returnCode = tensor->BuildFromTensorDesc(nnTensorDesc);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("AddTensorDesc failed, error happened when build NNTensor from OH_NNCore_TensorDesc.");
        return returnCode;
    }

    // The NNTensor is named as "Tensor: <tensor index>"".
    tensor->SetName("Tensor: " + std::to_string(m_allTensors.size()));
    m_allTensors.emplace_back(tensor);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::SetTensorType(uint32_t index, OH_NN_TensorType tensorType)
{
    if (IsBuild()) {
        LOGE("SetTensorType failed, SetTensorType is forbidden after model has been built.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (index >= m_allTensors.size()) {
        LOGE("SetTensorType failed, passed index %u out of the number of added tensors.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    std::shared_ptr<NNTensor> tensor = m_allTensors[index];
    OH_NN_ReturnCode returnCode = tensor->SetTensorType(tensorType);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("SetTensorType failed, error happened when setting tensor type.");
    }

    return returnCode;
}

OH_NN_ReturnCode InnerModel::SetTensorQuantParam(uint32_t index, const NN_QuantParam* quantParam)
{
    if (IsBuild()) {
        LOGE("SetTensorQuantParam failed, SetTensorValue is forbidden after model has been built.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (index >= m_allTensors.size()) {
        LOGE("SetTensorQuantParam failed, passed index %u out of the number of added tensors.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    std::shared_ptr<NNTensor> tensor = m_allTensors[index];
    // quantParam is validated in outer function, no need to check it here.
    OH_NN_ReturnCode returnCode = tensor->SetQuantParam(quantParam);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("SetTensorQuantParam failed, error happened when set quant param.");
    }

    return returnCode;
}

// DOTO: 圈复杂度待优化
OH_NN_ReturnCode InnerModel::SetTensorValue(uint32_t index, const void* buffer, size_t length)
{
    if (IsBuild()) {
        LOGE("SetTensorValue failed, SetTensorValue is forbidden after model has been built.");
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

    size_t allTensorsSize = m_allTensors.size();
    for (uint32_t i = 0; i < indices.size; i++) {
        if (indices.data[i] >= allTensorsSize) {
            LOGE("ValidateTensors failed, index %{public}u is out of the number of added tensors.", indices.data[i]);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::AddOperation(OH_NN_OperationType opType, const OH_NN_UInt32Array& paramIndices,
                                          const OH_NN_UInt32Array& inputIndices, const OH_NN_UInt32Array& outputIndices)
{
    if (IsBuild()) {
        LOGE("AddOperation failed, AddOperation is forbidden after model has been built.");
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

    const Ops::OpsRegistry& opsRegistry = Ops::OpsRegistry::GetSingleton();
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
    if (IsBuild()) {
        LOGE("SpecifyInputsAndOutputs failed, SpecifyInputsAndOutputs is forbidden after model has been built.");
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

    std::transform(m_inputIndices.begin(), m_inputIndices.end(), std::back_inserter(m_inputTensors),
        [this](uint32_t i) {
            return m_allTensors[i];
        });

    std::transform(m_outputIndices.begin(), m_outputIndices.end(), std::back_inserter(m_outputTensors),
        [this](uint32_t i) {
            return m_allTensors[i];
        });

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::CheckParameters() const
{
    if (m_liteGraph != nullptr) {
        LOGE("CheckParameters failed, liteGraph is not nullptr.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_metaGraph != nullptr) {
        LOGE("CheckParameters failed, metaGraph is not nullptr.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!m_allTensors.empty()) {
        LOGE("CheckParameters failed, m_allTensors is not empty.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!(m_inputTensors.empty() && (m_inputIndices.empty()))) {
        LOGE("CheckParameters failed, m_inputTensors is not empty.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!(m_outputTensors.empty() && (m_outputIndices.empty()))) {
        LOGE("CheckParameters failed, m_outputTensors is not empty.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::SetInputsAndOutputsInfo(const OH_NN_TensorInfo* inputsInfo, size_t inputSize,
    const OH_NN_TensorInfo* outputsInfo, size_t outputSize)
{
    OH_NN_ReturnCode ret = CheckParameters();
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetInputsAndOutputsInfo failed, error happened when checking parameters.");
        return ret;
    }

    // 根据inputsInfo设置输入NNTensor
    for (size_t i = 0; i < inputSize; ++i) {
        std::shared_ptr<NNTensor> tensor = CreateSharedPtr<NNTensor>();
        if (tensor == nullptr) {
            LOGE("SetInputsAndOutputsInfo failed, error happened when creating input NNTensor.");
            return OH_NN_MEMORY_ERROR;
        }

        ret = tensor->BuildFromOHNNTensorInfo(inputsInfo[i]);
        if (ret != OH_NN_SUCCESS) {
            LOGE("SetInputsAndOutputsInfo failed, error happened when building input NNTensor from info.");
            return ret;
        }
        m_inputIndices.emplace_back(i);
        m_allTensors.emplace_back(tensor);
        m_inputTensors.emplace_back(tensor);
    }

    // 根据outputsInfo设置输入NNTensor
    for (size_t i = 0; i < outputSize; ++i) {
        std::shared_ptr<NNTensor> tensor = CreateSharedPtr<NNTensor>();
        if (tensor == nullptr) {
            LOGE("SetInputsAndOutputsInfo failed, error happened when creating output NNTensor.");
            return OH_NN_MEMORY_ERROR;
        }

        ret = tensor->BuildFromOHNNTensorInfo(outputsInfo[i]);
        if (ret != OH_NN_SUCCESS) {
            LOGE("SetInputsAndOutputsInfo failed, error happened when building output NNTensor from info.");
            return ret;
        }
        m_outputIndices.emplace_back(i + inputSize);
        m_allTensors.emplace_back(tensor);
        m_outputTensors.emplace_back(tensor);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode InnerModel::Build()
{
    NNRT_TRACE_NAME("Build model");
    if (IsBuild()) {
        LOGE("Build failed, OH_NNModel_Finish() shouldn't be called after OH_NNModel_Finish() or "
             "OH_NNModel_BuildFromMetaGraph() or OH_NNModel_BuildFromLiteGraph().");
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
    std::transform(m_inputIndices.begin(), m_inputIndices.end(), std::back_inserter(inputIndices),
        [modelIDToGraphID](uint32_t index) {return modelIDToGraphID.at(index);});

    std::vector<uint32_t>& outputIndices = m_liteGraph->output_indices_;
    std::transform(m_outputIndices.begin(), m_outputIndices.end(), std::back_inserter(outputIndices),
        [modelIDToGraphID](uint32_t index) {return modelIDToGraphID.at(index);});
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

    BackendManager& backendManager = BackendManager::GetInstance();

    std::shared_ptr<Backend> backend = backendManager.GetBackend(deviceID);
    if (backend == nullptr) {
        LOGE("GetSupportedOperations failed, retrieve backend failed.");
        return OH_NN_FAILED;
    }

    std::vector<bool> supportedOperations;
    std::shared_ptr<NNBackend> nnBackend = std::reinterpret_pointer_cast<NNBackend>(backend);
    OH_NN_ReturnCode ret = nnBackend->GetSupportedOperation(m_liteGraph, supportedOperations);
    if (ret != OH_NN_SUCCESS) {
        LOGE("GetSupportedOperations failed, error happened when get supported operations from backends.");
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

std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> InnerModel::GetInputTensorDescs() const
{
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> inputTensorDescs;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> tensorDescPair;
    for (auto inputTensor : m_inputTensors) {
        tensorDescPair.first = OHOS::NeuralNetworkRuntime::CreateSharedPtr<TensorDesc>();
        inputTensor->ConvertToTensorDesc(*(tensorDescPair.first.get()));
        tensorDescPair.second = inputTensor->GetType();
        inputTensorDescs.emplace_back(tensorDescPair);
    }

    return inputTensorDescs;
}

std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> InnerModel::GetOutputTensorDescs() const
{
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> outputTensorDescs;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> tensorDescPair;
    for (auto outputTensor : m_outputTensors) {
        tensorDescPair.first = OHOS::NeuralNetworkRuntime::CreateSharedPtr<TensorDesc>();
        outputTensor->ConvertToTensorDesc(*(tensorDescPair.first.get()));
        tensorDescPair.second = outputTensor->GetType();
        outputTensorDescs.emplace_back(tensorDescPair);
    }

    return outputTensorDescs;
}

void* InnerModel::GetMetaGraph() const
{
    return m_metaGraph;
}

Buffer InnerModel::GetQuantBuffer() const
{
    return m_quantBuffer;
}

std::string InnerModel::GetModelName() const
{
    return m_modelName;
}

std::string InnerModel::GetProfiling() const
{
    return m_isProfiling;
}

std::map<std::string, std::string> InnerModel::GetOpLayouts() const
{
    return m_opLayouts;
}

TuningStrategy InnerModel::GetTuningStrategy() const
{
    return m_tuningStrategy;
}

void InnerModel::SetTuningStrategy(const TuningStrategy tuningStrategy)
{
    m_tuningStrategy = tuningStrategy;
}
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
