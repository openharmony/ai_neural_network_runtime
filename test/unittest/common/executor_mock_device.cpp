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

#include "frameworks/native/compilation.h"
#include "frameworks/native/execution_plan.h"
#include "frameworks/native/hdi_device.h"
#include "test/unittest/common/mock_idevice.h"

OH_NN_ReturnCode OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;

namespace OHOS {
namespace NeuralNetworkRuntime {
std::shared_ptr<Device> ExecutionPlan::GetInputDevice() const
{
    sptr<OHOS::HDI::Nnrt::V1_0::INnrtDevice> idevice
        = sptr<OHOS::HDI::Nnrt::V1_0::MockIDevice>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::MockIDevice());
    std::shared_ptr<Device> device = std::make_shared<HDIDevice>(idevice);
    return device;
}

std::shared_ptr<Device> ExecutionPlan::GetOutputDevice() const
{
    sptr<OHOS::HDI::Nnrt::V1_0::INnrtDevice> idevice
        = sptr<OHOS::HDI::Nnrt::V1_0::MockIDevice>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::MockIDevice());
    std::shared_ptr<Device> device = std::make_shared<HDIDevice>(idevice);
    return device;
}

void* HDIDevice::AllocateBuffer(size_t length)
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

OH_NN_ReturnCode HDIDevice::ReleaseBuffer(const void* buffer)
{
    if (buffer == nullptr) {
        LOGE("alloct buffer failed");
        return OH_NN_FAILED;
    }
    free(const_cast<void *>(buffer));
    buffer = nullptr;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIPreparedModel::Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
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

std::shared_ptr<ExecutionPlan> Compilation::GetExecutionPlan() const
{
    sptr<OHOS::HDI::Nnrt::V1_0::IPreparedModel> hdiPreparedModel = OHOS::sptr<OHOS::HDI::Nnrt::V1_0::HDI::Nnrt::V1_0
        ::MockIPreparedModel>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::HDI::Nnrt::V1_0::MockIPreparedModel());

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModel>(hdiPreparedModel);
    sptr<OHOS::HDI::Nnrt::V1_0::INnrtDevice> idevice
        = OHOS::sptr<OHOS::HDI::Nnrt::V1_0::MockIDevice>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::MockIDevice());
    std::shared_ptr<Device> device = std::make_shared<HDIDevice>(idevice);
    ExecutionPlan executor(preparedModel, device);
    std::shared_ptr<ExecutionPlan> pExcutor = std::make_shared<ExecutionPlan>(executor);
    return pExcutor;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS