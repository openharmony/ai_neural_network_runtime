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

#include "maxpool_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const std::string OP_NAME = "MaxPool";

MaxPoolBuilder::MaxPoolBuilder() {}

MaxPoolBuilder::~MaxPoolBuilder() {}

OH_NN_ReturnCode MaxPoolBuilder::Build(const std::vector<uint32_t>& paramsIndex,
    const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex,
    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    OH_NN_ReturnCode returnCode = PoolingBuild(paramsIndex, inputsIndex, outputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[MaxPool] Build failed, PoolingBuild failed.");
        return returnCode;
    }

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr MaxPoolBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[MaxPool] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = MindIR_MaxPoolFusion_CreatePrimitive(m_kernelSize, m_strides, m_pad,
        m_padMode, m_format, m_global, m_activationType);
    LiteGraphPrimitvePtr  graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(MaxPoolBuilder, OH_NN_OPS_MAX_POOL);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
