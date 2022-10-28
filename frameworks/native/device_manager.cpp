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

#include "device_manager.h"

#include "hdi_interfaces.h"
#include "hdi_device.h"
#include "common/log.h"
#include "common/utils.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
const std::vector<size_t>& DeviceManager::GetAllDeviceId()
{
    m_tmpDeviceIds.clear();
    std::shared_ptr<Device> device {nullptr};
    for (auto iter = m_devices.begin(); iter != m_devices.end(); ++iter) {
        device = iter->second;
        if (!IsValidDevice(device)) {
            continue;
        }
        m_tmpDeviceIds.emplace_back(iter->first);
    }
    return m_tmpDeviceIds;
}

std::shared_ptr<Device> DeviceManager::GetDevice(size_t deviceId) const
{
    auto iter = m_devices.find(deviceId);
    if (iter == m_devices.end()) {
        LOGE("DeviceId is not found, deviceId=%zu", deviceId);
        return nullptr;
    }

    return iter->second;
}

const std::string& DeviceManager::GetDeviceName(size_t deviceId)
{
    m_tmpDeviceName.clear();
    auto iter = m_devices.find(deviceId);
    if (iter == m_devices.end()) {
        LOGE("DeviceId is not found, deviceId=%zu", deviceId);
        return m_tmpDeviceName;
    }

    std::string deviceName;
    auto ret = iter->second->GetDeviceName(deviceName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Get device name failed.");
        return m_tmpDeviceName;
    }

    std::string vendorName;
    ret = iter->second->GetVendorName(vendorName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Get vendor name failed.");
        return m_tmpDeviceName;
    }

    m_tmpDeviceName = GenUniqueName(deviceName, vendorName);
    return m_tmpDeviceName;
}

std::string DeviceManager::GenUniqueName(const std::string& deviceName, const std::string& vendorName) const
{
    return deviceName + "_" + vendorName;
}

OH_NN_ReturnCode DeviceManager::RegisterDevice(std::function<std::shared_ptr<Device>()> creator)
{
    auto regDevice = creator();
    if (regDevice == nullptr) {
        LOGE("Cannot create device, register device failed.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!IsValidDevice(regDevice)) {
        LOGE("Device is not avaliable.");
        return OH_NN_UNAVALIDABLE_DEVICE;
    }

    std::string deviceName;
    auto ret = regDevice->GetDeviceName(deviceName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Get device name failed.");
        return ret;
    }

    std::string vendorName;
    ret = regDevice->GetVendorName(vendorName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Get vendor name failed.");
        return ret;
    }

    const std::lock_guard<std::mutex> lock(m_mtx);
    std::string uniqueName = GenUniqueName(deviceName, vendorName);
    auto setResult = m_uniqueName.emplace(uniqueName);
    if (!setResult.second) {
        LOGE("Device already exists, cannot register again. deviceName=%s, vendorName=%s",
            deviceName.c_str(), vendorName.c_str());
        return OH_NN_FAILED;
    }

    m_devices.emplace(std::hash<std::string>{}(uniqueName), regDevice);
    return OH_NN_SUCCESS;
}

void DeviceManager::DiscoverHDIDevices()
{
    // only one device from HDI now.
    OHOS::sptr<V1_0::INnrtDevice> iDevice = V1_0::INnrtDevice::Get();
    if (iDevice == nullptr) {
        LOGW("Get HDI device failed.");
        return;
    }

    std::string deviceName;
    std::string vendorName;
    auto hdiRet = iDevice->GetDeviceName(deviceName);
    if (hdiRet != HDF_SUCCESS) {
        LOGW("Get device name failed. ErrorCode=%d", hdiRet);
        return;
    }
    hdiRet = iDevice->GetVendorName(vendorName);
    if (hdiRet != HDF_SUCCESS) {
        LOGW("Get vendor name failed. ErrorCode=%d", hdiRet);
        return;
    }

    std::string uniqueName = GenUniqueName(deviceName, vendorName);
    const std::lock_guard<std::mutex> lock(m_mtx);
    auto setResult = m_uniqueName.emplace(uniqueName);
    if (!setResult.second) {
        LOGW("Device already exists, cannot register again. deviceName=%s, vendorName=%s",
            deviceName.c_str(), vendorName.c_str());
        return;
    }

    std::shared_ptr<Device> device = CreateSharedPtr<HDIDevice>(iDevice);
    if (device == nullptr) {
        LOGW("Failed to register device, because fail to create device instance.");
        return;
    }
    m_devices.emplace(std::hash<std::string>{}(uniqueName), device);
}

bool DeviceManager::IsValidDevice(std::shared_ptr<Device> device) const
{
    DeviceStatus status {DeviceStatus::UNKNOWN};
    auto ret = device->GetDeviceStatus(status);
    if (ret != OH_NN_SUCCESS || status == DeviceStatus::UNKNOWN || status == DeviceStatus::OFFLINE) {
        return false;
    }
    return true;
}
} // NeuralNetworkRuntime
} // OHOS