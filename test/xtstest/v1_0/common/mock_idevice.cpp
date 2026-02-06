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
#include <string>
#include <sys/mman.h>

#include "const.h"
#include "mock_idevice.h"
#include "hdi_device_v1_0.h"
#include "log.h"
#include "utils.h"
#include "nnbackend.h"
#include "backend_registrar.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
std::shared_ptr<Backend> HDIDeviceV1_0Creator()
{
    std::string deviceName;
    std::string vendorName;
    std::string version;

    // only one device from HDI now.
    OHOS::sptr<V1_0::INnrtDevice> iDevice = V1_0::INnrtDevice::Get();
    if (iDevice == nullptr) {
        LOGW("Get HDI device failed.");
        return nullptr;
    }

    auto hdiRet = iDevice->GetDeviceName(deviceName);
    if (hdiRet != HDF_SUCCESS) {
        LOGW("Get device name failed. ErrorCode=%d", hdiRet);
        return nullptr;
    }

    hdiRet = iDevice->GetVendorName(vendorName);
    if (hdiRet != HDF_SUCCESS) {
        LOGW("Get vendor name failed. ErrorCode=%d", hdiRet);
        return nullptr;
    }

    std::pair<uint32_t, uint32_t> hdiVersion;
    hdiRet = iDevice->GetVersion(hdiVersion.first, hdiVersion.second);
    if (hdiRet != HDF_SUCCESS) {
        LOGW("Get version failed. ErrorCode=%d", hdiRet);
        return nullptr;
    }
    version = 'v' + std::to_string(hdiVersion.first) + '_' + std::to_string(hdiVersion.second);
    const std::string& backendName = GenUniqueName(deviceName, vendorName, version);

    std::shared_ptr<Device> device = CreateSharedPtr<HDIDeviceV1_0>(iDevice);
    if (device == nullptr) {
        LOGW("Failed to register device, because fail to create device instance.");
        return nullptr;
    }

    std::shared_ptr<Backend> backend = std::make_shared<NNBackend>(device, std::hash<std::string>{}(backendName));
    if (backend == nullptr) {
        LOGW("Failed to register backend, because fail to create backend.");
    }
    return backend;
}

REGISTER_BACKEND(HDIDeviceV1_0, HDIDeviceV1_0Creator)
} // namespace NeuralNetworkRuntime
} // namespace OHOS

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {

sptr<INnrtDevice> INnrtDevice::Get(bool isStub)
{
    return INnrtDevice::Get("mock_device_service", isStub);
}

sptr<INnrtDevice> INnrtDevice::Get(const std::string &serviceName, bool isStub)
{
    if (isStub) {
        return nullptr;
    }
    sptr<INnrtDevice> mockIDevice = sptr<MockIDevice>(MockIDevice::GetInstance());
    return mockIDevice;
}

MockIDevice::MockIDevice()
{
    m_bufferFd = 0; 
}

MockIDevice::~MockIDevice()
{
    for (auto fd : m_fds) {
        close(fd);
    }
}

MockIPreparedModel::~MockIPreparedModel()
{
    for (auto fd : m_fds) {
        close(fd);
    }
}

MockIDevice *MockIDevice::GetInstance()
{
    static MockIDevice iDevice;
    return &iDevice;
}

void MockIDevice::SetFP16Supported(bool isSupported)
{
    m_fp16 = isSupported;
}

void MockIDevice::SetPerformanceSupported(bool isSupported)
{
    m_performance = isSupported;
}

void MockIDevice::SetPrioritySupported(bool isSupported)
{
    m_priority = isSupported;
}

void MockIDevice::SetModelCacheSupported(bool isSupported)
{
    m_cache = isSupported;
}

void MockIDevice::SetOperationsSupported(std::vector<bool> isSupported)
{
    m_operations = isSupported;
}

void MockIDevice::SetDynamicInputSupported(bool isSupported)
{
    m_dynamic = isSupported;
}

int32_t MockIDevice::GetDeviceName(std::string& name)
{
    name = "Device-CPU";
    return HDF_SUCCESS;
}

int32_t MockIDevice::GetVendorName(std::string& name)
{
    name = "TestVendor";
    return HDF_SUCCESS;
}

int32_t MockIDevice::GetDeviceType(DeviceType& deviceType)
{
    deviceType = DeviceType::CPU;
    return HDF_SUCCESS;
}

int32_t MockIDevice::GetDeviceStatus(DeviceStatus& status)
{
    status = DeviceStatus::AVAILABLE;
    return HDF_SUCCESS;
}

int32_t MockIDevice::GetVersion(uint32_t &majorVersion, uint32_t &minorVersion)
{
    majorVersion = 1;
    minorVersion = 0;
    return HDF_SUCCESS;
}

int32_t MockIDevice::GetSupportedOperation(const Model& model, std::vector<bool>& ops)
{
    ops = m_operations;
    return HDF_SUCCESS;
}

int32_t MockIDevice::IsFloat16PrecisionSupported(bool& isSupported)
{
    isSupported = m_fp16;
    return HDF_SUCCESS;
}

int32_t MockIDevice::IsPerformanceModeSupported(bool& isSupported)
{
    isSupported = m_performance;
    return HDF_SUCCESS;
}

int32_t MockIDevice::IsPrioritySupported(bool& isSupported)
{
    isSupported = m_priority;
    return HDF_SUCCESS;
}

int32_t MockIDevice::IsDynamicInputSupported(bool& isSupported)
{
    isSupported = m_dynamic;
    return HDF_SUCCESS;
}

int32_t MockIDevice::IsModelCacheSupported(bool& isSupported)
{
    isSupported = m_cache;
    return HDF_SUCCESS;
}

int32_t MockIDevice::AllocateBuffer(uint32_t length, SharedBuffer &buffer)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    buffer.fd = AshmemCreate("allocateBuffer", length);
    buffer.bufferSize = AshmemGetSize(buffer.fd);
    buffer.offset = 0;
    buffer.dataSize = length;

    AshmemSetProt(buffer.fd, PROT_READ | PROT_WRITE);
    m_fds.emplace(buffer.fd);
    m_bufferFd = buffer.fd;
    return HDF_SUCCESS;
}

int32_t MockIDevice::ReleaseBuffer(const SharedBuffer &buffer)
{
    if (m_fds.find(buffer.fd) == m_fds.end()) {
        LOGE("ReleaseBuffer:buffer fd is invalid. fd = %d", buffer.fd);
        return HDF_FAILURE;
    }
    if (close(buffer.fd) != 0) {
        LOGE("ReleaseBuffer:Close buffer fd failed. fd = %d", buffer.fd);
        return HDF_FAILURE;
    }
    return HDF_SUCCESS;
}

int32_t MockIDevice::MemoryCopy(float* data, uint32_t length)
{
    std::lock_guard<std::mutex> lock(m_mtx);
    void* mapData = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, m_bufferFd, 0);
    if (mapData == MAP_FAILED) {
        LOGE("[Mock_Device]::ExportModelCache failed, Map fd to address failed: %{public}s.", strerror(errno));
        return HDF_FAILURE;
    }

    auto memRet = memcpy_s(mapData, length, data, length);
    auto unmapResult = munmap(mapData, length);
    if (unmapResult != 0) {
        LOGE("[Mock_Device]ExportModelCache failed . Please try again.");
        return HDF_FAILURE;
    }
    if (memRet != EOK) {
        LOGE("[Mock_Device]ExportModelCache failed, failed to memcpy_s data type.");
        return HDF_FAILURE;
    }
    return HDF_SUCCESS;
}

int32_t MockIDevice::PrepareModel(const Model& model, const ModelConfig& config, sptr<IPreparedModel>& preparedModel)
{
    preparedModel = new (std::nothrow) V1_0::MockIPreparedModel();
    return HDF_SUCCESS;
}

int32_t MockIDevice::PrepareModelFromModelCache(const std::vector<SharedBuffer>& modelCache, const ModelConfig& config,
    sptr<IPreparedModel>& preparedModel)
{
    preparedModel = new (std::nothrow) V1_0::MockIPreparedModel();
    return HDF_SUCCESS;
}

int32_t MockIPreparedModel::ExportModelCache(std::vector<SharedBuffer>& modelCache)
{
    if (!modelCache.empty()) {
        LOGE("[NNRtTest] The parameters of ExportModelCache should be an empty vector.");
        return HDF_ERR_INVALID_PARAM;
    }
    uint8_t bufferData[4] = {0, 1, 2, 3};
    uint32_t size = sizeof(bufferData);
    SharedBuffer buffer;
    buffer.fd = AshmemCreate("cache", size);
    buffer.bufferSize = AshmemGetSize(buffer.fd);
    buffer.offset = 0;
    buffer.dataSize = size;
    AshmemSetProt(buffer.fd, PROT_READ | PROT_WRITE);

    void* data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, buffer.fd, 0);
    if (data == MAP_FAILED) {
        LOGE("[Mock_Device]::ExportModelCache failed, Map fd to address failed: %{public}s.", strerror(errno));
        return HDF_FAILURE;
    }

    auto memRet = memcpy_s(data, size, bufferData, size);
    auto unmapResult = munmap(data, size);
    if (unmapResult != 0) {
        LOGE("[Mock_Device]ExportModelCache failed . Please try again.");
        return HDF_FAILURE;
    }
    if (memRet != EOK) {
        LOGE("[Mock_Device]ExportModelCache failed, failed to memcpy_s data type.");
        return HDF_FAILURE;
    }
    m_fds.emplace(buffer.fd);
    modelCache.emplace_back(buffer);
    return HDF_SUCCESS;
}

int32_t MockIPreparedModel::GetVersion(uint32_t &majorVersion, uint32_t &minorVersion)
{
    majorVersion = 1;
    minorVersion = 0;
    return HDF_SUCCESS;
}

int32_t MockIPreparedModel::Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims, std::vector<bool>& isOutputBufferEnough)
{
    outputsDims = {{1, 2, 2, 1}};
    isOutputBufferEnough = {true};
    return HDF_SUCCESS;
}

} // namespace V1_0
} // namespace Nnrt
} // namespace HDI
} // namespace OHOS
