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

#include "batch_to_space_nd_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const int CROPS_ROWS = 2;
static const int CROPS_COLUMN = 2;
static const std::string OP_NAME = "BatchToSpaceND";

BatchToSpaceNDBuilder::BatchToSpaceNDBuilder() {}

BatchToSpaceNDBuilder::~BatchToSpaceNDBuilder() {}

OH_NN_ReturnCode BatchToSpaceNDBuilder::SetInputBlock(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[BatchToSpaceND] SetInputBlock failed, the BlockSize should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[BatchToSpaceND] SetInputBlock GetBuffer return nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    int64_t* pBlockSize = static_cast<int64_t*>(buffer);

    uint32_t elementCount = tensor->GetElementCount();
    for (uint32_t i = 0; i < elementCount; ++i) {
        m_blockSize.emplace_back(*pBlockSize);
        ++pBlockSize;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode BatchToSpaceNDBuilder::SetInputCrops(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[BatchToSpaceND] SetInputCrops failed, the Crops should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[BatchToSpaceND] SetInputCrops GetBuffer return nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    int64_t* pCropsData = static_cast<int64_t*>(buffer);

    std::vector<std::vector<int64_t>> cropsData;
    for (int i = 0; i < CROPS_ROWS; i++) {
        std::vector<int64_t> vect_data;
        vect_data.reserve(CROPS_COLUMN);
        for (int j = 0; j < CROPS_COLUMN; j++) {
            vect_data.push_back(*pCropsData++);
        }
        cropsData.push_back(vect_data);
    }
    m_crops = cropsData;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode BatchToSpaceNDBuilder::Build(const std::vector<uint32_t>& paramsIndex,
    const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex,
    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[BatchToSpaceND] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[BatchToSpaceND] Build failed, passed invalid input or output index.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE:
                returnCode = SetInputBlock(tensor);
                break;
            case OH_NN_BATCH_TO_SPACE_ND_CROPS:
                returnCode = SetInputCrops(tensor);
                break;
            default:
                LOGE("[BatchToSpaceND] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[BatchToSpaceND] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr BatchToSpaceNDBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[BatchToSpaceND] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_BatchToSpaceND_CreatePrimitive(m_blockSize, m_crops);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(BatchToSpaceNDBuilder, OH_NN_OPS_BATCH_TO_SPACE_ND);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
