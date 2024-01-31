/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
#include <iostream>
#include "depth_to_space_builder.h"

#include "transform.h"
#include "validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "DepthToSpace";

DepthToSpaceBuilder::DepthToSpaceBuilder() {}

DepthToSpaceBuilder::~DepthToSpaceBuilder() {}

OH_NN_ReturnCode DepthToSpaceBuilder::SetBlockSize(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[DepthToSpace] The blockSize should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DepthToSpace] The blockSize should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DepthToSpace] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_blockSize = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthToSpaceBuilder::SetFormat(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[DepthToSpace] The format should be type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DepthToSpace] The format should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DepthToSpace] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    int8_t* formatData = static_cast<int8_t*>(buffer);
    
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateTensorFormat(static_cast<OH_NN_Format>(*formatData))) {
        LOGE("[DepthToSpace] SetFormat failed. Format type is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto pFormat = (OH_NN_Format)(*formatData);
    m_format = NNToMS::TransformFormat(pFormat);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthToSpaceBuilder::SetMode(std::shared_ptr<NNTensor> tensor)
{
    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[DepthToSpace] The mode should be type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DepthToSpace] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_mode = static_cast<char*>(buffer);
    
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthToSpaceBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                            const std::vector<uint32_t>& inputsIndex,
                                            const std::vector<uint32_t>& outputsIndex,
                                            const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[DepthToSpace] Build failed, the depthToSpace operation has been build. cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[DepthToSpace] Build failed, passed invalid input or output index.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;
    
    OH_NN_ReturnCode returnCode;
    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        switch (tensor->GetType()) {
            case OH_NN_DEPTH_TO_SPACE_BLOCK_SIZE:
                returnCode = SetBlockSize(tensor);
                break;
            case OH_NN_DEPTH_TO_SPACE_FORMAT:
                returnCode = SetFormat(tensor);
                break;
            case OH_NN_DEPTH_TO_SPACE_MODE:
                returnCode = SetMode(tensor);
                break;
            default:
                LOGE("[DepthToSpace] Build failed, param invalid, type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[DepthToSpace] Build failed, passed invalid param.");
            return returnCode;
        }
    }
    
    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr DepthToSpaceBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[DepthToSpace] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_DepthToSpace_CreatePrimitive(m_blockSize, m_format, m_mode);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(DepthToSpaceBuilder, OH_NN_OPS_DEPTH_TO_SPACE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS