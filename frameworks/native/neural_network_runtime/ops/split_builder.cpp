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

#include "split_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const std::string OP_NAME = "Split";

SplitBuilder::SplitBuilder() {}

SplitBuilder::~SplitBuilder() {}

OH_NN_ReturnCode SplitBuilder::SetInputAndOutput(const std::vector<uint32_t> &inputsIndex,
    const std::vector<uint32_t> &outputsIndex, const std::vector<std::shared_ptr<NNTensor>> &allTensors)
{
    auto inputSize = inputsIndex.size();
    if (inputSize != INPUT_NUM) {
        LOGE("[SplitBuilder] The number of inputsIndex should be %d, its number is %zu.", INPUT_NUM, inputSize);
        return OH_NN_INVALID_PARAMETER;
    }

    auto allTensorSize = allTensors.size();
    for (auto index : inputsIndex) {
        if (index >= allTensorSize) {
            LOGE("[SplitBuilder] InputsIndex of Split is out of range.");
            return OH_NN_INVALID_PARAMETER;
        }
    }

    for (auto index : outputsIndex) {
        if (index >= allTensorSize) {
            LOGE("[SplitBuilder] OutputsIndex of Split is out of range.");
            return OH_NN_INVALID_PARAMETER;
        }
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode SplitBuilder::SetAxis(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[SplitBuilder] The 4th input axis should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != 1) {
        LOGE("[SplitBuilder] The 4th input axis should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[SplitBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_axis = *(static_cast<const int64_t *>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode SplitBuilder::SetOutputNum(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[SplitBuilder] The 2nd input outputNum should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != 1) {
        LOGE("[SoftmaxBuilder] The 2nd input outputNum should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_output_num = *(static_cast<const int64_t *>(tensor->GetBuffer()));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode SplitBuilder::SetSizeSplits(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[SplitBuilder] The 3rd input sizeSplit should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    const int64_t *size_splits_data_ptr = reinterpret_cast<const int64_t *>(tensor->GetBuffer());
    for (uint32_t i = 0; i < tensor->GetElementCount(); i++) {
        m_size_splits.push_back(*size_splits_data_ptr++);
    }

    return OH_NN_SUCCESS;
}

/**
 * Build method.
 * 1.set attr of ops.
 * 2.set inputIndex of ops.
 * 3.set outputIndex of ops.
 */
OH_NN_ReturnCode SplitBuilder::Build(const std::vector<uint32_t> &paramsIndex,
                                     const std::vector<uint32_t> &inputsIndex,
                                     const std::vector<uint32_t> &outputsIndex,
                                     const std::vector<std::shared_ptr<NNTensor>> &allTensors)
{
    if (m_isBuild) {
        LOGE("[SplitBuilder] Split operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = SetInputAndOutput(inputsIndex, outputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[SplitBuilder] Set index of inputs or outputs failed.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_SPLIT_AXIS:
                returnCode = SetAxis(tensor);
                break;
            case OH_NN_SPLIT_OUTPUT_NUM:
                returnCode = SetOutputNum(tensor);
                break;
            case OH_NN_SPLIT_SIZE_SPLITS:
                returnCode = SetSizeSplits(tensor);
                break;
            default:
                LOGE("[SplitBuilder] Parameter Type is invalid. type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[SplitBuilder] Passed invalid param.");
            return returnCode;
        }
    }

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphTensorPtr SplitBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[SplitBuilder] Cannot get primitive before call build.");
        return { nullptr, DestroyLiteGraphPrimitive };
    }

    auto primitive = mindspore::lite::MindIR_Split_CreatePrimitive(m_output_num, m_size_splits, m_axis);
    if (primitive == nullptr) {
        LOGE("[SplitBuilder] MindIR_Split_CreatePrimitive failed.");
        return { nullptr, DestroyLiteGraphPrimitive };
    }

    LiteGraphTensorPtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(SplitBuilder, OH_NN_OPS_SPLIT);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
