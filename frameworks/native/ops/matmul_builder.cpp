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

#include "matmul_builder.h"

#include "frameworks/native/transform.h"
#include "frameworks/native/validation.h"
#include "frameworks/native/ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const int SCALE_LENGTH = 1;
static const std::string OP_NAME = "Matmul";

MatmulBuilder::MatmulBuilder() {}

MatmulBuilder::~MatmulBuilder() {}

OH_NN_ReturnCode MatmulBuilder::SetTransposeA(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[Matmul] Matmul SetTransposeA failed. The transposeA should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[Matmul] Matmul SetTransposeA failed. The transposeA should have type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Matmul] SetTransposeA failed, the transposeA passed a empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_transposeA = *static_cast<bool*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode MatmulBuilder::SetTransposeB(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[Matmul] Matmul SetTransposeB failed. The transposeB should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[Matmul] Matmul SetTransposeB failed. The transposeB TransposeY should have type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Matmul] SetTransposeB failed, the transposeB passed a empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_transposeB = *static_cast<bool*>(buffer);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode MatmulBuilder::SetActivationType(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[Matmul] Matmul SetActivationType failed. The shape of activation should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[Matmul] Matmul SetActivationType failed. The activation should be type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Matmul] SetActivationType failed, the activationType passed a empty buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    int8_t* fuseData = static_cast<int8_t*>(buffer);
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*fuseData))) {
        LOGE("[Matmul] Matmul SetActivationType failed. Fuse activation type is invalid");
        return OH_NN_INVALID_PARAMETER;
    }

    auto fuseType = (OH_NN_FuseType)(*fuseData);
    m_activationType = NNToMS::TransfromFusionType(fuseType);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode MatmulBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                      const std::vector<uint32_t>& inputsIndex,
                                      const std::vector<uint32_t>& outputsIndex,
                                      const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Matmul] Matmul Build failed. operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Matmul] Matmul Build failed. Passed invalid input or output indices.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_MATMUL_TRANSPOSE_A:
                returnCode = SetTransposeA(tensor);
                break;
            case OH_NN_MATMUL_TRANSPOSE_B:
                returnCode = SetTransposeB(tensor);
                break;
            case OH_NN_MATMUL_ACTIVATION_TYPE:
                returnCode = SetActivationType(tensor);
                break;
            default:
                LOGE("[Matmul] Parameter Type is invalid, type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Matmul] Matmul Build failed. Passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr MatmulBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Matmul] Matmul GetPrimitive failed. Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = mindspore::lite::MindIR_MatMulFusion_CreatePrimitive(m_transposeA, m_transposeB, m_activationType);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(MatmulBuilder, OH_NN_OPS_MATMUL);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS