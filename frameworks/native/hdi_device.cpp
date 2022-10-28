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

#include "hdi_device.h"

#include "hdf_base.h"
#include "mindir.h"

#include "hdi_prepared_model.h"
#include "memory_manager.h"
#include "transform.h"
#include "common/log.h"
#include "common/utils.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
HDIDevice::HDIDevice(OHOS::sptr<V1_0::INnrtDevice> device) : m_iDevice(device)
{
    device->GetVersion(m_hdiVersion.first, m_hdiVersion.second);
}

OH_NN_ReturnCode HDIDevice::GetDeviceName(std::string& name)
{
    auto ret = m_iDevice->GetDeviceName(name);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI device name failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::GetVendorName(std::string& name)
{
    auto ret = m_iDevice->GetVendorName(name);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI device vendor name failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::GetDeviceType(OH_NN_DeviceType& deviceType)
{
    V1_0::DeviceType iDeviceType;
    auto ret = m_iDevice->GetDeviceType(iDeviceType);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI device type failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }

    deviceType = HDIToNN::TransHDIDeviceType(iDeviceType);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::GetDeviceStatus(DeviceStatus& status)
{
    V1_0::DeviceStatus iDeviceStatus;
    auto ret = m_iDevice->GetDeviceStatus(iDeviceStatus);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI device status failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    status = HDIToNN::TransHDIDeviceStatus(iDeviceStatus);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                                  std::vector<bool>& ops)
{
    if (model == nullptr) {
        LOGE("Model is nullptr, cannot query supported operation.");
        return OH_NN_NULL_PTR;
    }

    V1_0::SharedBuffer tensorBuffer {INVALID_FD, 0, 0, 0};
    size_t tensorSize = mindspore::lite::MindIR_LiteGraph_GetConstTensorSize(model.get());
    int32_t hdiRet {0};
    if (tensorSize > 0) {
        hdiRet = m_iDevice->AllocateBuffer(tensorSize, tensorBuffer);
        if (hdiRet != HDF_SUCCESS || tensorBuffer.fd == INVALID_FD) {
            LOGE("Allocate tensor buffer error when get supported operation. ErrorCode: %d", hdiRet);
            return OH_NN_FAILED;
        }
    }

    auto iModel = mindspore::lite::MindIR_LiteGraph_To_Model(model.get(), tensorBuffer);
    if (iModel == nullptr) {
        LOGE("Parse litegraph to hdi model failed.");
        ReleaseSharedBuffer(tensorBuffer);
        return OH_NN_FAILED;
    }

    hdiRet = m_iDevice->GetSupportedOperation(*iModel, ops);

    mindspore::lite::MindIR_Model_Destroy(&iModel);
    auto ret = ReleaseSharedBuffer(tensorBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Release tensorBuffer failed.");
        return OH_NN_FAILED;
    }
    if (hdiRet != HDF_SUCCESS) {
        LOGE("Get supported operation failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsFloat16PrecisionSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsFloat16PrecisionSupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query fp16 precision supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsPerformanceModeSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsPerformanceModeSupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query performance mode supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsPrioritySupported(bool& isSupported)
{
    auto ret = m_iDevice->IsPrioritySupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query priority supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsDynamicInputSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsDynamicInputSupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query dynamic input supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsModelCacheSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsModelCacheSupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query cache model supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                         const ModelConfig& config,
                                         std::shared_ptr<PreparedModel>& preparedModel)
{
    if (model == nullptr) {
        LOGE("Model is nullptr, cannot prepare model.");
        return OH_NN_INVALID_PARAMETER;
    }

    V1_0::SharedBuffer tensorBuffer {INVALID_FD, 0, 0, 0};
    size_t tensorSize = mindspore::lite::MindIR_LiteGraph_GetConstTensorSize(model.get());
    int32_t hdiRet {0};
    if (tensorSize > 0) {
        hdiRet = m_iDevice->AllocateBuffer(tensorSize, tensorBuffer);
        if (hdiRet != HDF_SUCCESS || tensorBuffer.fd == INVALID_FD) {
            LOGE("Allocate tensor buffer error when prepare model. ErrorCode: %d", hdiRet);
            return OH_NN_FAILED;
        }
    }

    V1_0::Model* iModel = mindspore::lite::MindIR_LiteGraph_To_Model(model.get(), tensorBuffer);
    if (iModel == nullptr) {
        LOGE("Parse litegraph to hdi model failed.");
        ReleaseSharedBuffer(tensorBuffer);
        return OH_NN_FAILED;
    }

    V1_0::ModelConfig iModelConfig;
    iModelConfig.enableFloat16 = config.enableFloat16;
    iModelConfig.mode = NNToHDI::TransPerformanceMode(config.mode);
    iModelConfig.priority = NNToHDI::TransPriority(config.priority);
    OHOS::sptr<V1_0::IPreparedModel> iPreparedModel;

    auto preparedRet = m_iDevice->PrepareModel(*iModel, iModelConfig, iPreparedModel);

    mindspore::lite::MindIR_Model_Destroy(&iModel);
    auto ret = ReleaseSharedBuffer(tensorBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Release tensorBuffer failed.");
        return OH_NN_FAILED;
    }
    if (preparedRet != HDF_SUCCESS || iPreparedModel == nullptr) {
        LOGE("Prepare model failed. ErrorCode=%d", preparedRet);
        return OH_NN_FAILED;
    }

    preparedModel = CreateSharedPtr<HDIPreparedModel>(iPreparedModel);
    if (preparedModel == nullptr) {
        LOGE("Prepare model failed, because fail to create preparedModel instance.");
        return OH_NN_MEMORY_ERROR;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::PrepareModelFromModelCache(const std::vector<ModelBuffer>& modelCache,
                                                       const ModelConfig& config,
                                                       std::shared_ptr<PreparedModel>& preparedModel)
{
    std::vector<V1_0::SharedBuffer> iBuffers;
    auto memManager = MemoryManager::GetInstance();
    Memory memory;
    OH_NN_ReturnCode ret;
    size_t modelCacheSize = modelCache.size();
    for (size_t i = 0; i < modelCacheSize; i++) {
        ret = memManager->GetMemory(modelCache[i].buffer, memory);
        if (ret != OH_NN_SUCCESS) {
            LOGE("The %zuth model cache is invalid. Please put valid model cache.", i + 1);
            return ret;
        }
        iBuffers.emplace_back(V1_0::SharedBuffer {memory.fd, memory.length, 0, memory.length});
    }

    V1_0::ModelConfig iModelConfig;
    iModelConfig.enableFloat16 = config.enableFloat16;
    iModelConfig.mode = NNToHDI::TransPerformanceMode(config.mode);
    iModelConfig.priority = NNToHDI::TransPriority(config.priority);

    OHOS::sptr<V1_0::IPreparedModel> iPreparedModel;
    auto hdiRet = m_iDevice->PrepareModelFromModelCache(iBuffers, iModelConfig, iPreparedModel);
    if (hdiRet != HDF_SUCCESS) {
        LOGE("Prepare model from cache failed. ErrorCode=%d", hdiRet);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }

    preparedModel = CreateSharedPtr<HDIPreparedModel>(iPreparedModel);
    if (preparedModel == nullptr) {
        LOGE("Prepare model from model cache failed, because fail to create preparedModel instance.");
        return OH_NN_MEMORY_ERROR;
    }
    return OH_NN_SUCCESS;
}

void* HDIDevice::AllocateBuffer(size_t length)
{
    if (length == 0) {
        LOGE("The length param is invalid, length=0");
        return nullptr;
    }

    V1_0::SharedBuffer buffer;
    auto ret = m_iDevice->AllocateBuffer(length, buffer);
    if (ret != HDF_SUCCESS) {
        LOGE("Allocate buffer error. ErrorCode: %d", ret);
        return nullptr;
    }

    auto memManager = MemoryManager::GetInstance();
    auto addr = memManager->MapMemory(buffer.fd, length);
    if (addr == nullptr) {
        LOGE("Map fd to address failed.");
    }
    return addr;
}

OH_NN_ReturnCode HDIDevice::ReleaseBuffer(const void* buffer)
{
    if (buffer == nullptr) {
        LOGE("Buffer is nullptr, no need to release.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto memManager = MemoryManager::GetInstance();
    Memory memory;
    auto ret = memManager->GetMemory(buffer, memory);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Invalid Buffer, it is not NNRt buffer.");
        return ret;
    }

    V1_0::SharedBuffer hdiBuffer {memory.fd, memory.length, 0, memory.length};
    auto deviceResult = m_iDevice->ReleaseBuffer(hdiBuffer);
    if (deviceResult != HDF_SUCCESS) {
        LOGE("Device release buffer error. ErrorCode: %d", deviceResult);
        return OH_NN_FAILED;
    }

    ret = memManager->UnMapMemory(buffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Unmap memory failed.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::ReleaseSharedBuffer(const V1_0::SharedBuffer& buffer)
{
    if (buffer.fd == INVALID_FD) {
        LOGI("No need to release. fd=%d", INVALID_FD);
        return OH_NN_SUCCESS;
    }

    auto ret = m_iDevice->ReleaseBuffer(buffer);
    if (ret != HDF_SUCCESS) {
        LOGE("Device release buffer error. ErrorCode=%d", ret);
        return OH_NN_FAILED;
    }
    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS