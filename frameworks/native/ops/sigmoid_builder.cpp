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

#include "sigmoid_builder.h"

#include "mindir.h"

#include "frameworks/native/ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "Sigmoid";

SigmoidBuilder::SigmoidBuilder() {}

SigmoidBuilder::~SigmoidBuilder() {}

OH_NN_ReturnCode SigmoidBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                       const std::vector<uint32_t>& inputsIndex,
                                       const std::vector<uint32_t>& outputsIndex,
                                       const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[SigmoidBuilder] Build failed, the Sigmoid operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[SigmoidBuilder] Build failed, passed invalid input or output index.");
        return returnCode;
    }

    if (!paramsIndex.empty()) {
        LOGW("[SigmoidBuilder] Build failed, the Sigmoid expects no parameters, but receive %zu", paramsIndex.size());
        return OH_NN_INVALID_PARAMETER;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr SigmoidBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[SigmoidBuilder] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    float alpha{0.0f};
    float minVal{0.0f};
    float maxVal{0.0f};
    bool approximate{false};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_SIGMOID};

    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha, minVal,
                                                                         maxVal, approximate);

    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(SigmoidBuilder, OH_NN_OPS_SIGMOID);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS