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

#include "expandims_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "ExpandDims";

ExpandDimsBuilder::ExpandDimsBuilder() {}

ExpandDimsBuilder::~ExpandDimsBuilder() {}

OH_NN_ReturnCode ExpandDimsBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                          const std::vector<uint32_t>& inputsIndex,
                                          const std::vector<uint32_t>& outputsIndex,
                                          const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[ExpandDims] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[ExpandDims] Build failed, the input or output index of ExpandDims operation is invalid.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    if (!paramsIndex.empty()) {
        LOGE("[ExpandDims] Build failed, expandDims expects no parameters");
        return OH_NN_INVALID_PARAMETER;
    }

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr ExpandDimsBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[ExpandDims] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_ExpandDims_CreatePrimitive();
    LiteGraphPrimitvePtr  graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(ExpandDimsBuilder, OH_NN_OPS_EXPAND_DIMS);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
