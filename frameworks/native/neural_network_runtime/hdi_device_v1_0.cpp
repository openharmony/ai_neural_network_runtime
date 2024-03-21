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

#include "hdi_device_v1_0.h"

#include "hdf_base.h"
#include "mindir.h"

#include "hdi_prepared_model_v1_0.h"
#include "lite_graph_to_hdi_model_v1_0.h"
#include "memory_manager.h"
#include "transform.h"
#include "common/log.h"
#include "common/utils.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace {
OH_NN_DeviceType TransHDIDeviceV1_0Type(const V1_0::DeviceType& iDeviceType)
{
    switch (iDeviceType) {
        case V1_0::DeviceType::CPU:
            return OH_NN_CPU;
        case V1_0::DeviceType::GPU:
            return OH_NN_GPU;
        case V1_0::DeviceType::ACCELERATOR:
            return OH_NN_ACCELERATOR;
        default:
            return OH_NN_OTHERS;
    }
}

DeviceStatus TransHDIDeviceV1_0Status(const V1_0::DeviceStatus& iDeviceStatus)
{
    switch (iDeviceStatus) {
        case V1_0::DeviceStatus::AVAILABLE:
            return DeviceStatus::AVAILABLE;
        case V1_0::DeviceStatus::BUSY:
            return DeviceStatus::BUSY;
        case V1_0::DeviceStatus::OFFLINE:
            return DeviceStatus::OFFLINE;
        default:
            return DeviceStatus::UNKNOWN;
    }
}

V1_0::PerformanceMode TransPerformanceMode(const OH_NN_PerformanceMode& mode)
{
    switch (mode) {
        case OH_NN_PERFORMANCE_LOW:
            return V1_0::PerformanceMode::PERFORMANCE_LOW;
        case OH_NN_PERFORMANCE_MEDIUM:
            return V1_0::PerformanceMode::PERFORMANCE_MEDIUM;
        case OH_NN_PERFORMANCE_HIGH:
            return V1_0::PerformanceMode::PERFORMANCE_HIGH;
        case OH_NN_PERFORMANCE_EXTREME:
            return V1_0::PerformanceMode::PERFORMANCE_EXTREME;
        default:
            return V1_0::PerformanceMode::PERFORMANCE_NONE;
    }
}

V1_0::Priority TransPriority(const OH_NN_Priority& priority)
{
    switch (priority) {
        case OH_NN_PRIORITY_LOW:
            return V1_0::Priority::PRIORITY_LOW;
        case OH_NN_PRIORITY_MEDIUM:
            return V1_0::Priority::PRIORITY_MEDIUM;
        case OH_NN_PRIORITY_HIGH:
            return V1_0::Priority::PRIORITY_HIGH;
        default:
            return V1_0::Priority::PRIORITY_NONE;
    }
}
}

HDIDeviceV1_0::HDIDeviceV1_0(OHOS::sptr<V1_0::INnrtDevice> device) : m_iDevice(device)
{}

OH_NN_ReturnCode HDIDeviceV1_0::GetDeviceName(std::string& name)
{
    auto ret = m_iDevice->GetDeviceName(name);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI device name failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::GetVendorName(std::string& name)
{
    auto ret = m_iDevice->GetVendorName(name);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI device vendor name failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::GetVersion(std::string& version)
{
    auto ret = m_iDevice->GetVersion(m_hdiVersion.first, m_hdiVersion.second);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI version failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    version = 'v' + std::to_string(m_hdiVersion.first) + '_' + std::to_string(m_hdiVersion.second);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::GetDeviceType(OH_NN_DeviceType& deviceType)
{
    V1_0::DeviceType iDeviceType;
    auto ret = m_iDevice->GetDeviceType(iDeviceType);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI device type failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }

    deviceType = TransHDIDeviceV1_0Type(iDeviceType);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::GetDeviceStatus(DeviceStatus& status)
{
    V1_0::DeviceStatus iDeviceStatus;
    auto ret = m_iDevice->GetDeviceStatus(iDeviceStatus);
    if (ret != HDF_SUCCESS) {
        LOGE("Get HDI device status failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    status = TransHDIDeviceV1_0Status(iDeviceStatus);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
    std::vector<bool>& ops)
{
    if (model == nullptr) {
        LOGE("Model is nullptr, cannot query supported operation.");
        return OH_NN_NULL_PTR;
    }

    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {INVALID_FD, 0, 0, 0};
    size_t tensorSize = mindspore::lite::MindIR_LiteGraph_GetConstTensorSize(model.get());
    int32_t hdiRet {0};
    if (tensorSize > 0) {
        hdiRet = m_iDevice->AllocateBuffer(tensorSize, tensorBuffer);
        if (hdiRet != HDF_SUCCESS || tensorBuffer.fd == INVALID_FD) {
            LOGE("Allocate tensor buffer error when get supported operation. ErrorCode: %d", hdiRet);
            return OH_NN_FAILED;
        }
    }

    auto iModel = V1::LiteGraph_To_HDIModel(model.get(), tensorBuffer);
    if (iModel == nullptr) {
        LOGE("Parse litegraph to hdi model failed.");
        ReleaseSharedBuffer(tensorBuffer);
        return OH_NN_FAILED;
    }

    hdiRet = m_iDevice->GetSupportedOperation(*iModel, ops);

    V1::HDIModel_Destroy(&iModel);
    auto ret = ReleaseSharedBuffer(tensorBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Release tensorBuffer failed.");
        return OH_NN_FAILED;
    }
    if (hdiRet != HDF_SUCCESS) {
        LOGE("Get supported operation failed. ErrorCode=%d", hdiRet);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsFloat16PrecisionSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsFloat16PrecisionSupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query fp16 precision supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsPerformanceModeSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsPerformanceModeSupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query performance mode supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsPrioritySupported(bool& isSupported)
{
    auto ret = m_iDevice->IsPrioritySupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query priority supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsDynamicInputSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsDynamicInputSupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query dynamic input supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsModelCacheSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsModelCacheSupported(isSupported);
    if (ret != HDF_SUCCESS) {
        LOGE("Query cache model supported failed. ErrorCode=%d", ret);
        return OH_NN_UNAVAILABLE_DEVICE;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
    const Buffer& quantBuffer, const ModelConfig& config, std::shared_ptr<PreparedModel>& preparedModel)
{
    if (model == nullptr) {
        LOGE("Model is nullptr, cannot prepare model.");
        return OH_NN_INVALID_PARAMETER;
    }

    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {INVALID_FD, 0, 0, 0};
    size_t tensorSize = mindspore::lite::MindIR_LiteGraph_GetConstTensorSize(model.get());
    int32_t hdiRet {0};
    if (tensorSize > 0) {
        hdiRet = m_iDevice->AllocateBuffer(tensorSize, tensorBuffer);
        if (hdiRet != HDF_SUCCESS || tensorBuffer.fd == INVALID_FD) {
            LOGE("Allocate tensor buffer error when prepare model. ErrorCode: %d", hdiRet);
            return OH_NN_FAILED;
        }
    }

    V1_0::Model* iModel = V1::LiteGraph_To_HDIModel(model.get(), tensorBuffer);
    if (iModel == nullptr) {
        LOGE("Parse litegraph to hdi model failed.");
        ReleaseSharedBuffer(tensorBuffer);
        return OH_NN_FAILED;
    }

    V1_0::ModelConfig iModelConfig;
    iModelConfig.enableFloat16 = config.enableFloat16;
    iModelConfig.mode = TransPerformanceMode(config.mode);
    iModelConfig.priority = TransPriority(config.priority);
    OHOS::sptr<V1_0::IPreparedModel> iPreparedModel;

    auto preparedRet = m_iDevice->PrepareModel(*iModel, iModelConfig, iPreparedModel);

    V1::HDIModel_Destroy(&iModel);
    auto ret = ReleaseSharedBuffer(tensorBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Release tensorBuffer failed.");
        return OH_NN_FAILED;
    }
    if (preparedRet != HDF_SUCCESS || iPreparedModel == nullptr) {
        LOGE("Prepare model failed. ErrorCode=%d", preparedRet);
        return OH_NN_FAILED;
    }

    preparedModel = CreateSharedPtr<HDIPreparedModelV1_0>(iPreparedModel);
    if (preparedModel == nullptr) {
        LOGE("Prepare model failed, because fail to create preparedModel instance.");
        return OH_NN_MEMORY_ERROR;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::PrepareModel(const void* metaGraph,
                                             const Buffer& quantBuffer,
                                             const ModelConfig& config,
                                             std::shared_ptr<PreparedModel>& preparedModel)
{
    return OH_NN_OPERATION_FORBIDDEN;
}

OH_NN_ReturnCode HDIDeviceV1_0::PrepareModelFromModelCache(const std::vector<Buffer>& modelCache,
    const ModelConfig& config, std::shared_ptr<PreparedModel>& preparedModel)
{
    std::vector<V1_0::SharedBuffer> iBuffers;
    auto memManager = MemoryManager::GetInstance();
    Memory memory;
    OH_NN_ReturnCode ret;
    size_t modelCacheSize = modelCache.size();
    for (size_t i = 0; i < modelCacheSize; i++) {
        ret = memManager->GetMemory(modelCache[i].data, memory);
        if (ret != OH_NN_SUCCESS) {
            LOGE("The %zuth model cache is invalid. Please put valid model cache.", i + 1);
            return ret;
        }
        iBuffers.emplace_back(V1_0::SharedBuffer {memory.fd, memory.length, 0, memory.length});
    }

    V1_0::ModelConfig iModelConfig;
    iModelConfig.enableFloat16 = config.enableFloat16;
    iModelConfig.mode = TransPerformanceMode(config.mode);
    iModelConfig.priority = TransPriority(config.priority);

    OHOS::sptr<V1_0::IPreparedModel> iPreparedModel;
    auto hdiRet = m_iDevice->PrepareModelFromModelCache(iBuffers, iModelConfig, iPreparedModel);
    if (hdiRet != HDF_SUCCESS) {
        LOGE("Prepare model from cache failed. ErrorCode=%d", hdiRet);
        return OH_NN_UNAVAILABLE_DEVICE;
    }

    preparedModel = CreateSharedPtr<HDIPreparedModelV1_0>(iPreparedModel);
    if (preparedModel == nullptr) {
        LOGE("Prepare model from model cache failed, because fail to create preparedModel instance.");
        return OH_NN_MEMORY_ERROR;
    }
    return OH_NN_SUCCESS;
}

void* HDIDeviceV1_0::AllocateBuffer(size_t length)
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

OH_NN_ReturnCode HDIDeviceV1_0::AllocateBuffer(size_t length, int& fd)
{
    if (length == 0) {
        LOGE("The length param is invalid, length=0");
        return OH_NN_INVALID_PARAMETER;
    }

    V1_0::SharedBuffer buffer;
    auto ret = m_iDevice->AllocateBuffer(length, buffer);
    if (ret != HDF_SUCCESS) {
        LOGE("Allocate buffer error. ErrorCode: %d", ret);
        return OH_NN_MEMORY_ERROR;
    }

    fd = buffer.fd;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::ReleaseBuffer(int fd, size_t length)
{
    V1_0::SharedBuffer hdiBuffer {fd, length, 0, length};
    auto deviceResult = m_iDevice->ReleaseBuffer(hdiBuffer);
    if (deviceResult != HDF_SUCCESS) {
        LOGE("Device release buffer error. ErrorCode: %d", deviceResult);
        return OH_NN_MEMORY_ERROR;
    }
    return OH_NN_SUCCESS;
}


void* HDIDeviceV1_0::AllocateTensorBuffer(size_t length, std::shared_ptr<TensorDesc> tensor)
{
    return AllocateBuffer(length);
}

void* HDIDeviceV1_0::AllocateTensorBuffer(size_t length, std::shared_ptr<NNTensor> tensor)
{
    return AllocateBuffer(length);
}

OH_NN_ReturnCode HDIDeviceV1_0::ReleaseBuffer(const void* buffer)
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

OH_NN_ReturnCode HDIDeviceV1_0::ReleaseSharedBuffer(const V1_0::SharedBuffer& buffer)
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

OH_NN_ReturnCode HDIDeviceV1_0::PrepareOfflineModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                                    const ModelConfig& config,
                                                    std::shared_ptr<PreparedModel>& preparedModel)
{
    LOGE("HDIDeviceV1.0 not support PrepareOfflineModel.");
    return OH_NN_OPERATION_FORBIDDEN;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS
