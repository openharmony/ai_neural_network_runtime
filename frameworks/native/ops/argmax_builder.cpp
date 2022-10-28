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

#include "argmax_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "ArgMax";

ArgMaxBuilder::ArgMaxBuilder() {}

ArgMaxBuilder::~ArgMaxBuilder() {}

OH_NN_ReturnCode ArgMaxBuilder::SetAxis(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[ArgMax] SetAxis failed, the axis should be type HNN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ArgMax] SetAxis GetBuffer return nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_axis = *(static_cast<int64_t*>(buffer));
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ArgMaxBuilder::SetKeepdims(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[ArgMax] SetKeepdims failed, the keep_dims should be type HNN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ArgMax] SetKeepdims GetBuffer return nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_keepDims = *(static_cast<bool*>(buffer));

    return OH_NN_SUCCESS;
}

/**
 * Build method.
 * 1.build primitive of ops.
 * 2.build inputIndex of ops.
 * 3.build outputIndex of ops.
 */
OH_NN_ReturnCode ArgMaxBuilder::Build(const std::vector<uint32_t>& paramsIndex,
    const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex,
    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[ArgMax] Build failed, build operation has been completed, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[ArgMax] Build failed, passed invalid input or output index.");
        return returnCode;
    }
    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (int i : paramsIndex) {
        const std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_ARG_MAX_AXIS:
                returnCode = SetAxis(tensor);
                break;
            case OH_NN_ARG_MAX_KEEPDIMS:
                returnCode = SetKeepdims(tensor);
                break;
            default:
                LOGE("[ArgMax] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[ArgMax] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr ArgMaxBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[ArgMax] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_ArgMaxFusion_CreatePrimitive(m_axis, m_topK, m_keepDims, m_outMaxValue);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}
REGISTER_OPS(ArgMaxBuilder, OH_NN_OPS_ARG_MAX);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
