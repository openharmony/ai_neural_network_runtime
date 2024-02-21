/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#include "lstm_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 6;
static const int OUTPUT_NUM = 3;
static const int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "LSTM";

LSTMBuilder::LSTMBuilder() {}

LSTMBuilder::~LSTMBuilder() {}

OH_NN_ReturnCode LSTMBuilder::SetBidirectional(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[LSTM] The bidirectional should be type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The bidirectional should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_bidirectional = *(static_cast<bool*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetHasBias(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[LSTM] The hasBias should be type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The hasBias should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_hasBias = *(static_cast<bool*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetInputSize(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[LSTM] The inputSize should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The inputSize should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_inputSize = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetHiddenSize(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[LSTM] The hiddenSize should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The hiddenSize should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_hiddenSize = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetNumLayers(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[LSTM] The numLayers should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The numLayers should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_numLayers = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetNumDirections(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[LSTM] The numDirections should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The numDirections should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_numDirections = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetDropout(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[LSTM] The dropout should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The dropout should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_dropout = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetZoneoutCell(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[LSTM] The zoneoutCell should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The zoneoutCell should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_zoneoutCell = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetZoneoutHidden(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[LSTM] The zoneoutHidden should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The zoneoutHidden should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_zoneoutHidden = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::SetProjSize(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[LSTM] The projSize should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[LSTM] The projSize should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[LSTM] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_projSize = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::ParseParam(const std::vector<uint32_t>& paramsIndex,
                                         const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    OH_NN_ReturnCode returnCode;
    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_LSTM_BIDIRECTIONAL:
                returnCode = SetBidirectional(tensor);
                break;
            case OH_NN_LSTM_HAS_BIAS:
                returnCode = SetHasBias(tensor);
                break;
            case OH_NN_LSTM_INPUT_SIZE:
                returnCode = SetInputSize(tensor);
                break;
            case OH_NN_LSTM_HIDDEN_SIZE:
                returnCode = SetHiddenSize(tensor);
                break;
            case OH_NN_LSTM_NUM_LAYERS:
                returnCode = SetNumLayers(tensor);
                break;
            case OH_NN_LSTM_NUM_DIRECTIONS:
                returnCode = SetNumDirections(tensor);
                break;
            case OH_NN_LSTM_DROPOUT:
                returnCode = SetDropout(tensor);
                break;
            case OH_NN_LSTM_ZONEOUT_CELL:
                returnCode = SetZoneoutCell(tensor);
                break;
            case OH_NN_LSTM_ZONEOUT_HIDDEN:
                returnCode = SetZoneoutHidden(tensor);
                break;
            case OH_NN_LSTM_PROJ_SIZE:
                returnCode = SetProjSize(tensor);
                break;
            default:
                LOGE("[LSTM] Build failed, param invalid, type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[LSTM] Build failed, passed invalid param.");
            return returnCode;
        }
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode LSTMBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                    const std::vector<uint32_t>& inputsIndex,
                                    const std::vector<uint32_t>& outputsIndex,
                                    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[LSTM] Build failed, the LSTM operation has been build. cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[LSTM] Build failed, passed invalid input or output index.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    ret = ParseParam(paramsIndex, allTensors);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[LSTM] ParseParam failed, passed invalid param.");
        return ret;
    }

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr LSTMBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[LSTM] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_LSTM_CreatePrimitive(m_bidirectional, m_hasBias, m_inputSize,
        m_hiddenSize, m_numLayers, m_numDirections, m_dropout, m_zoneoutCell, m_zoneoutHidden, m_projSize);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(LSTMBuilder, OH_NN_OPS_LSTM);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS