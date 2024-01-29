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

#include "tanh_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const std::string OP_NAME = "Tanh";

TanhBuilder::TanhBuilder() {}

TanhBuilder::~TanhBuilder() {}

/**
 * Build method.
 * 1.set attr of ops.
 * 2.set inputIndex of ops.
 * 3.set outputIndex of ops.
 */
OH_NN_ReturnCode TanhBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                    const std::vector<uint32_t>& inputsIndex,
                                    const std::vector<uint32_t>& outputsIndex,
                                    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[TanhBuilder] Tanh operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[TanhBuilder] Passed invalid input or output index.");
        return returnCode;
    }

    if (!paramsIndex.empty()) {
        LOGE("[TanhBuilder] TanhBuilder expects no parameters, but receive %zu", paramsIndex.size());
        return OH_NN_INVALID_PARAMETER;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr TanhBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[TanhBuilder] Cannot get primitive before call build.");
        return { nullptr, DestroyLiteGraphPrimitive };
    }

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    auto primitive =
        mindspore::lite::MindIR_Activation_CreatePrimitive(m_activationType, alpha, minVal, maxVal, approximate);
    if (primitive == nullptr) {
        LOGE("[TanhBuilder] Create primitive of Tanh failed.");
        return { nullptr, DestroyLiteGraphPrimitive };
    }

    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(TanhBuilder, OH_NN_OPS_TANH);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS