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

#include "strided_slice_builder.h"

#include "mindir.h"

#include "interfaces/kits/c/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 4;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "StridedSlice";

StridedSliceBuilder::StridedSliceBuilder() {}

StridedSliceBuilder::~StridedSliceBuilder() {}

OH_NN_ReturnCode StridedSliceBuilder::SetInputOutput(const std::vector<uint32_t>& inputsIndex,
                                                     const std::vector<uint32_t>& outputsIndex,
                                                     const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[StridedSliceBuilder] Passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode StridedSliceBuilder::SetBeginMask(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[StridedSliceBuilder] The 5th input beginMask should be type HNN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[StridedSliceBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_begin_mask = *(static_cast<int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode StridedSliceBuilder::SetEndMask(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[StridedSliceBuilder] The 6th input endMask should be type HNN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[StridedSliceBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_end_mask = *(static_cast<int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode StridedSliceBuilder::SetEllipsisMask(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[StridedSliceBuilder] The 7th input ellipsisMask should be type HNN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[StridedSliceBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_ellipsis_mask = *(static_cast<int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode StridedSliceBuilder::SetNewAxisMask(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[StridedSliceBuilder] The 8th input newAxisMask should be type HNN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[StridedSliceBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_new_axis_mask = *(static_cast<int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode StridedSliceBuilder::SetShrinkAxisMask(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[StridedSliceBuilder] The 9th input shrinkAxisMAsk should be type HNN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[StridedSliceBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_shrink_axis_mask = *(static_cast<int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

/**
 * Build method.
 * 1.set attr of ops.
 * 2.set inputIndex of ops.
 * 3.set outputIndex of ops.
 */
OH_NN_ReturnCode StridedSliceBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                            const std::vector<uint32_t>& inputsIndex,
                                            const std::vector<uint32_t>& outputsIndex,
                                            const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[StridedSliceBuilder] StridedSlice operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = SetInputOutput(inputsIndex, outputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[StridedSliceBuilder] Set index of inputs or outputs failed.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_STRIDED_SLICE_BEGIN_MASK:
                returnCode = SetBeginMask(tensor);
                break;
            case OH_NN_STRIDED_SLICE_END_MASK:
                returnCode = SetEndMask(tensor);
                break;
            case OH_NN_STRIDED_SLICE_ELLIPSIS_MASK:
                returnCode = SetEllipsisMask(tensor);
                break;
            case OH_NN_STRIDED_SLICE_NEW_AXIS_MASK:
                returnCode = SetNewAxisMask(tensor);
                break;
            case OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK:
                returnCode = SetShrinkAxisMask(tensor);
                break;
            default:
                LOGE("[StridedSliceBuilder] Parameter Type is invalid. type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[StridedSliceBuilder] Passed invalid param.");
            return returnCode;
        }
    }

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr StridedSliceBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[StridedSliceBuilder] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = mindspore::lite::MindIR_StridedSlice_CreatePrimitive(m_begin_mask, m_end_mask, m_ellipsis_mask,
        m_new_axis_mask, m_shrink_axis_mask);
    if (primitive == nullptr) {
        LOGE("[StridedSliceBuilder] MindIR_StridedSlice_CreatePrimitive failed.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(StridedSliceBuilder, OH_NN_OPS_STRIDED_SLICE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
