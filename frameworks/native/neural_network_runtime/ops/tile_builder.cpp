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

#include "tile_builder.h"

#include "mindir.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const int PARAM_MAX_NUM = 1;
static const std::string OP_NAME = "Tile";

TileBuilder::TileBuilder() {}

TileBuilder::~TileBuilder() {}

OH_NN_ReturnCode TileBuilder::SetDims(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[TileBuilder] The dims should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_dims.clear();

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[TileBuilder] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    int64_t* pDims = static_cast<int64_t*>(buffer);

    uint32_t elementCount = tensor->GetElementCount();
    for (uint32_t i = 0; i < elementCount; ++i) {
        m_dims.emplace_back(*pDims);
        ++pDims;
    }
    return OH_NN_SUCCESS;
}

/**
 * Build method.
 * 1.set attr of ops.
 * 2.set inputIndex of ops.
 * 3.set outputIndex of ops.
 */
OH_NN_ReturnCode TileBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                    const std::vector<uint32_t>& inputsIndex,
                                    const std::vector<uint32_t>& outputsIndex,
                                    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[TileBuilder] Tile operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[TileBuilder] Passed invalid input or output index.");
        return returnCode;
    }

    returnCode = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[TileBuilder] Passed invalid param index.");
        return returnCode;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            returnCode = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[TileBuilder] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[TileBuilder] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr TileBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[TileBuilder] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = mindspore::lite::MindIR_TileFusion_CreatePrimitive(m_dims);
    if (primitive == nullptr) {
        LOGE("[TileBuilder] MindIR_TileFusion_CreatePrimitive failed.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(TileBuilder, OH_NN_OPS_TILE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
