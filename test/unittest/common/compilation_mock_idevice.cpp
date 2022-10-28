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

#include "common/utils.h"
#include "frameworks/native/device_manager.h"
#include "frameworks/native/hdi_device.h"
#include "frameworks/native/nn_tensor.h"
#include "test/unittest/common/mock_idevice.h"

OH_NN_ReturnCode OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;

namespace OHOS {
namespace NeuralNetworkRuntime {
std::shared_ptr<Device> DeviceManager::GetDevice(size_t deviceId) const
{
    sptr<OHOS::HDI::Nnrt::V1_0::INnrtDevice> idevice
        = sptr<OHOS::HDI::Nnrt::V1_0::MockIDevice>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::MockIDevice());
    if (idevice == nullptr) {
        LOGE("DeviceManager mock GetDevice failed, error happened when new sptr");
        return nullptr;
    }

    std::shared_ptr<Device> device = CreateSharedPtr<HDIDevice>(idevice);
    if (device == nullptr) {
        LOGE("DeviceManager mock GetDevice failed, the device is nullptr");
        return nullptr;
    }

    if (deviceId == 0) {
        LOGE("DeviceManager mock GetDevice failed, the passed parameter deviceId is 0");
        return nullptr;
    } else {
        return device;
    }
}

OH_NN_ReturnCode HDIDevice::IsModelCacheSupported(bool& isSupported)
{
    // isSupported is false when expecting to return success
    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_SUCCESS) {
        // In order not to affect other use cases, set to the OH_NN_OPERATION_FORBIDDEN
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_SUCCESS;
    }

    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_FAILED) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_FAILED;
    }

    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                                  std::vector<bool>& ops)
{
    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_INVALID_FILE) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        ops.emplace_back(true);
        return OH_NN_SUCCESS;
    }

    if (model == nullptr) {
        LOGE("HDIDevice mock GetSupportedOperation failed, Model is nullptr, cannot query supported operation.");
        return OH_NN_NULL_PTR;
    }

    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_SUCCESS) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        ops.emplace_back(false);
        return OH_NN_SUCCESS;
    }

    ops.emplace_back(true);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsDynamicInputSupported(bool& isSupported)
{
    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_FAILED) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_FAILED;
    }

    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_INVALID_PATH) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_SUCCESS;
    }

    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsPerformanceModeSupported(bool& isSupported)
{
    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_FAILED) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_FAILED;
    }

    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_SUCCESS) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_SUCCESS;
    }

    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsPrioritySupported(bool& isSupported)
{
    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_INVALID_PARAMETER) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_INVALID_PARAMETER;
    }

    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_SUCCESS) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_SUCCESS;
    }

    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsFloat16PrecisionSupported(bool& isSupported)
{
    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_SUCCESS) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_SUCCESS;
    }

    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_MEMORY_ERROR) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        isSupported = false;
        return OH_NN_MEMORY_ERROR;
    }

    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                         const ModelConfig& config,
                                         std::shared_ptr<PreparedModel>& preparedModel)
{
    if (model == nullptr) {
        LOGE("HDIDevice mock PrepareModel failed, the model is nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    if (config.enableFloat16 == false) {
        LOGE("HDIDevice mock PrepareModel failed, the enableFloat16 is false");
        return OH_NN_FAILED;
    }

    sptr<OHOS::HDI::Nnrt::V1_0::IPreparedModel> hdiPreparedModel = sptr<OHOS::HDI::Nnrt::V1_0
        ::MockIPreparedModel>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::MockIPreparedModel());
    if (hdiPreparedModel == nullptr) {
        LOGE("HDIDevice mock PrepareModel failed, error happened when new sptr");
        return OH_NN_NULL_PTR;
    }

    preparedModel = CreateSharedPtr<HDIPreparedModel>(hdiPreparedModel);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIPreparedModel::ExportModelCache(std::vector<ModelBuffer>& modelCache)
{
    if (!modelCache.empty()) {
        LOGE("HDIPreparedModel mock ExportModelCache failed, the modelCache is not empty");
        return OH_NN_INVALID_PARAMETER;
    }

    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_FAILED) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        return OH_NN_FAILED;
    }

    int bufferSize = 13;
    ModelBuffer modelBuffer;
    std::string aBuffer = "mock_buffer_a";
    modelBuffer.buffer = (void*)aBuffer.c_str();
    modelBuffer.length = bufferSize;
    modelCache.emplace_back(modelBuffer);

    ModelBuffer modelBuffer2;
    std::string bBuffer = "mock_buffer_b";
    modelBuffer2.buffer = (void*)bBuffer.c_str();
    modelBuffer2.length = bufferSize;
    modelCache.emplace_back(modelBuffer2);

    return OH_NN_SUCCESS;
}

void* HDIDevice::AllocateBuffer(size_t length)
{
    if (length == 0) {
        LOGE("HDIDevice mock AllocateBuffer failed, the length param is invalid");
        return nullptr;
    }

    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_NULL_PTR) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        return nullptr;
    }

    void* buffer = (void*)malloc(length);
    if (buffer == nullptr) {
        LOGE("HDIDevice mock AllocateBuffer failed, the buffer is nullptr");
        return nullptr;
    }
    return buffer;
}

OH_NN_ReturnCode HDIDevice::ReleaseBuffer(const void* buffer)
{
    if (buffer == nullptr) {
        LOGE("HDIDevice mock ReleaseBuffer failed, the buffer is nullptr");
        return OH_NN_NULL_PTR;
    }

    free(const_cast<void *>(buffer));
    buffer = nullptr;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::PrepareModelFromModelCache(const std::vector<ModelBuffer>& modelCache,
                                                       const ModelConfig& config,
                                                       std::shared_ptr<PreparedModel>& preparedModel)
{
    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_FAILED) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        return OH_NN_FAILED;
    }

    if (modelCache.size() == 0 || config.enableFloat16 == false) {
        LOGE("HDIDevice mock PrepareModel failed, the modelCache size equals 0 or enableFloat16 is false");
        return OH_NN_FAILED;
    }

    sptr<OHOS::HDI::Nnrt::V1_0::IPreparedModel> hdiPreparedModel = sptr<OHOS::HDI::Nnrt::V1_0
        ::MockIPreparedModel>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::MockIPreparedModel());
    if (hdiPreparedModel == nullptr) {
        LOGE("HDIDevice mock PrepareModelFromModelCache failed, error happened when new sptr");
        return OH_NN_NULL_PTR;
    }

    preparedModel = CreateSharedPtr<HDIPreparedModel>(hdiPreparedModel);

    return OH_NN_SUCCESS;
}

bool NNTensor::IsDynamicShape() const
{
    if (HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_FAILED) {
        HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        return false;
    }

    return true;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS