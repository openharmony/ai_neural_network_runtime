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

#include "compilation.h"
#include "hdi_device_v1_0.h"
#include "test/unittest/common/v1_0/mock_idevice.h"

OH_NN_ReturnCode OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;

namespace OHOS {
namespace NeuralNetworkRuntime {

void* HDIDeviceV1_0::AllocateBuffer(size_t length)
{
    if (length == 0) {
        LOGE("The length param is invalid, length=0");
        return nullptr;
    }

    void* buffer = (void*)malloc(length);
    if (buffer == nullptr) {
        LOGE("alloct buffer failed");
        return nullptr;
    }

    if (OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_INVALID_PARAMETER) {
        OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        return nullptr;
    }
    return buffer;
}

OH_NN_ReturnCode HDIDeviceV1_0::ReleaseBuffer(const void* buffer)
{
    if (buffer == nullptr) {
        LOGE("alloct buffer failed");
        return OH_NN_FAILED;
    }
    free(const_cast<void *>(buffer));
    buffer = nullptr;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIPreparedModelV1_0::Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims, std::vector<bool>& isOutputBufferEnough)
{
    if (inputs.empty() || outputs.empty()) {
        return OH_NN_INVALID_PARAMETER;
    }

    if (OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_FAILED) {
        OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        return OH_NN_INVALID_PARAMETER;
    }

    isOutputBufferEnough.emplace_back(true);
    outputsDims.emplace_back(outputs[0].dimensions);

    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS