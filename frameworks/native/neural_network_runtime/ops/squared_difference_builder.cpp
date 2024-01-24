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

#include "squared_difference_builder.h"

#include "mindir.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "SquaredDifference";

SquaredDifferenceBuilder::SquaredDifferenceBuilder() {}

SquaredDifferenceBuilder::~SquaredDifferenceBuilder() {}

/**
 * Build method.
 * 1.set attr of ops.
 * 2.set inputIndex of ops.
 * 3.set outputIndex of ops.
 */
OH_NN_ReturnCode SquaredDifferenceBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                                 const std::vector<uint32_t>& inputsIndex,
                                                 const std::vector<uint32_t>& outputsIndex,
                                                 const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[SquaredDifferenceBuilder] SquaredDifference operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[SquaredDifferenceBuilder] Passed invalid input or output index.");
        return returnCode;
    }

    if (!paramsIndex.empty()) {
        LOGE("[SquaredDifferenceBuilder] squaredDifference expects no parameters, but receive %zu", paramsIndex.size());
        return OH_NN_INVALID_PARAMETER;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphTensorPtr SquaredDifferenceBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[SquaredDifferenceBuilder] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = mindspore::lite::MindIR_SquaredDifference_CreatePrimitive();
    if (primitive == nullptr) {
        LOGE("[SquaredDifferenceBuilder] MindIR_SquaredDifference_CreatePrimitive failed.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    LiteGraphTensorPtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(SquaredDifferenceBuilder, OH_NN_OPS_SQUARED_DIFFERENCE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS