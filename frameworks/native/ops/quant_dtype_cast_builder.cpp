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

#include "quant_dtype_cast_builder.h"

#include "mindir.h"

#include "frameworks/native/ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "QuantDTypeCast";

QuantDTypeCastBuilder::QuantDTypeCastBuilder() {}

QuantDTypeCastBuilder::~QuantDTypeCastBuilder() {}

OH_NN_ReturnCode QuantDTypeCastBuilder::SetSrcT(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[QuantDTypeCast] SetSrcT failed, the src_t should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[QuantDTypeCast] SetSrcT failed, the src_t passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_src_t = static_cast<uint64_t*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode QuantDTypeCastBuilder::SetDstT(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[QuantDTypeCast] SetDstT failed, the dst_t should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[QuantDTypeCast] SetDstT failed, the dst_t passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_dst_t = static_cast<uint64_t*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode QuantDTypeCastBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                              const std::vector<uint32_t>& inputsIndex,
                                              const std::vector<uint32_t>& outputsIndex,
                                              const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[QuantDTypeCast] Build failed, the QuantDTypeCast operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[QuantDTypeCast] Build failed, passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (uint32_t i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_QUANT_DTYPE_CAST_SRC_T:
                returnCode = SetSrcT(tensor);
                break;
            case OH_NN_QUANT_DTYPE_CAST_DST_T:
                returnCode = SetDstT(tensor);
                break;
            default:
                LOGE("[QuantDTypeCast] Build failed, parameter type is invalid. type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[QuantDTypeCast] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr QuantDTypeCastBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[QuantDTypeCast] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_QuantDTypeCast_CreatePrimitive(*m_src_t, *m_dst_t);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(QuantDTypeCastBuilder, OH_NN_OPS_QUANT_DTYPE_CAST);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS