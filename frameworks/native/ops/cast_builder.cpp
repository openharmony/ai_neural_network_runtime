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

#include "cast_builder.h"

#include "frameworks/native/transform.h"
#include "frameworks/native/validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 2;
static const int OUTPUT_NUM = 1;
static const int INPUT_TYPE = 1;
static const std::string OP_NAME = "Cast";

CastBuilder::CastBuilder() {}

CastBuilder::~CastBuilder() {}

OH_NN_ReturnCode CastBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                    const std::vector<uint32_t>& inputsIndex,
                                    const std::vector<uint32_t>& outputsIndex,
                                    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Cast] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }
    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Cast] Build failed, the input or output index of Cast operation is invalid.");
        return ret;
    }
    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    auto castType = allTensors[inputsIndex[INPUT_TYPE]]->GetBuffer();
    if (castType == nullptr) {
        LOGE("[Cast] Build castType GetBuffer return nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    OH_NN_DataType* castTypeInt = reinterpret_cast<OH_NN_DataType *>(castType);
    if (!Validation::ValidateTensorDataType(*castTypeInt)) {
        LOGE("[Cast] Type of cast operator is not validation.");
        return OH_NN_INVALID_PARAMETER;
    }
    *castTypeInt = (OH_NN_DataType)NNToHDI::TransDataType(*castTypeInt);

    if (!paramsIndex.empty()) {
        LOGE("[Cast] Cast expects no parameters");
        return OH_NN_INVALID_PARAMETER;
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);
    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr CastBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Cast] Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = mindspore::lite::MindIR_Cast_CreatePrimitive();
    LiteGraphPrimitvePtr  graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(CastBuilder, OH_NN_OPS_CAST);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
