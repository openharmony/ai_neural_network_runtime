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

#include "batchnorm_builder.h"

#include "mindir.h"

#include "frameworks/native/ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 5;
static const int OUTPUT_NUM = 1;
static const int SCALAR_LENGTH = 1;
const std::string OP_NAME = "BatchNorm";

BatchNormBuilder::BatchNormBuilder() {}

BatchNormBuilder::~BatchNormBuilder() {}

OH_NN_ReturnCode BatchNormBuilder::SetEpsilon(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[BatchNorm] SetEpsilon failed, the Epsilon should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[BatchNorm] SetEpsilon failed, the Epsilon shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[BatchNorm] SetEpsilon failed, the epsilon passed a empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_epsilon = *static_cast<float*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode BatchNormBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                         const std::vector<uint32_t>& inputsIndex,
                                         const std::vector<uint32_t>& outputsIndex,
                                         const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[BatchNorm] Build failed, batchNorm operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[BatchNorm] Build failed, passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_BATCH_NORM_EPSILON:
                returnCode = SetEpsilon(tensor);
                break;
            default:
                LOGE("[BatchNorm] Parameter Type is invalid, type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[BatchNorm] BatchNorm Build failed,, Passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr BatchNormBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[BatchNorm] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_FusedBatchNorm_CreatePrimitive(m_epsilon);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(BatchNormBuilder, OH_NN_OPS_BATCH_NORM);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS