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

#include "space_to_batch_nd_builder.h"

#include "mindir.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "SpaceToBatchND";
static const int PADDINGS_DATA_SIZE = 2;
static const int VECT_DATA_SIZE = 2;
static const int BLOCKSHAPE_RANK = 1;
static const int PADDINGS_RANK = 2;
static const int BLOCK_SIZE = 2;
static const int PADDINGS_SIZE = 4;

SpaceToBatchNDBuilder::SpaceToBatchNDBuilder() {}

SpaceToBatchNDBuilder::~SpaceToBatchNDBuilder() {}

OH_NN_ReturnCode SpaceToBatchNDBuilder::SetBlockShape(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[SpaceToBatchNDBuilder] The 2nd input blockShape should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto blockshape_shape = tensor->GetDimensions();
    if (blockshape_shape.size() != BLOCKSHAPE_RANK) {
        LOGE("[SpaceToBatchNDBuilder] Invalid rank of shape of 2nd input blockShape, should be 1 dimensions.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != BLOCK_SIZE) {
        LOGE("[SpaceToBatchNDBuilder] The 2nd input blockShape size should be 2.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[SpaceToBatchNDBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const int64_t* blockShapeData = reinterpret_cast<const int64_t*>(buffer);
    const uint32_t elementSize = tensor->GetElementCount();
    for (uint32_t i = 0; i < elementSize; ++i) {
        block_shape.push_back(blockShapeData[i]);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode SpaceToBatchNDBuilder::SetPaddings(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[SpaceToBatchNDBuilder] The 3rd input paddings should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto paddings_shape = tensor->GetDimensions();
    if (paddings_shape.size() != PADDINGS_RANK) {
        LOGE("[SpaceToBatchNDBuilder] Invalid rank of shape of 3rd input paddings, should be 2 dimensions.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != PADDINGS_SIZE) {
        LOGE("[SpaceToBatchNDBuilder] The 3rd input paddings size should be 4.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode returnCode = SetPadData(tensor);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[SpaceToBatchNDBuilder] SetPadData failed.");
        return returnCode;
    }

    return OH_NN_SUCCESS;
}
/**
 * Build method.
 * 1.set attr of ops.
 * 2.set inputIndex of ops.
 * 3.set outputIndex of ops.
 */
OH_NN_ReturnCode SpaceToBatchNDBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                              const std::vector<uint32_t>& inputsIndex,
                                              const std::vector<uint32_t>& outputsIndex,
                                              const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[SpaceToBatchNDBuilder] SpaceToBatchND operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[SpaceToBatchNDBuilder] Passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE:
                returnCode = SetBlockShape(tensor);
                break;
            case OH_NN_SPACE_TO_BATCH_ND_PADDINGS:
                returnCode = SetPaddings(tensor);
                break;
            default:
                LOGE("[SpaceToBatchNDBuilder] Parameter Type is invalid. type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[SpaceToBatchNDBuilder] Passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode SpaceToBatchNDBuilder::SetPadData(std::shared_ptr<NNTensor> tensor)
{
    paddings.clear();

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[SpaceToBatchNDBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const int64_t* paddingsData = reinterpret_cast<const int64_t*>(buffer);
    for (int i = 0; i < PADDINGS_DATA_SIZE; i++) {
        std::vector<int64_t> vect_data;
        vect_data.reserve(VECT_DATA_SIZE);
        for (int i = 0; i < VECT_DATA_SIZE; ++i) {
            vect_data.push_back(paddingsData[i]);
        }
        paddings.push_back(vect_data);
    }
    return OH_NN_SUCCESS;
}

LiteGraphTensorPtr SpaceToBatchNDBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[SpaceToBatchNDBuilder] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = mindspore::lite::MindIR_SpaceToBatchND_CreatePrimitive(block_shape, paddings);
    if (primitive == nullptr) {
        LOGE("[SpaceToBatchNDBuilder] MindIR_SpaceToBatchND_CreatePrimitive failed.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    LiteGraphTensorPtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(SpaceToBatchNDBuilder, OH_NN_OPS_SPACE_TO_BATCH_ND);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
