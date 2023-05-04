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
#include "device_discover.h"

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

    std::string version;
    ret = iter->second->GetVersion(version);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Get version failed.");
        return m_tmpDeviceName;
    }

    m_tmpDeviceName = GenUniqueName(deviceName, vendorName, version);
    return m_tmpDeviceName;
}

std::string DeviceManager::GenUniqueName(
    const std::string& deviceName, const std::string& vendorName, const std::string& version) const
{
    return deviceName + "_" + vendorName + "_" + version;
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

    std::string version;
    ret = regDevice->GetVersion(version);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Get version failed.");
        return ret;
    }

    const std::lock_guard<std::mutex> lock(m_mtx);
    std::string uniqueName = GenUniqueName(deviceName, vendorName, version);
    auto setResult = m_uniqueName.emplace(uniqueName);
    if (!setResult.second) {
        LOGE("Device already exists, cannot register again. deviceName=%s, vendorName=%s",
            deviceName.c_str(), vendorName.c_str());
        return OH_NN_FAILED;
    }

    m_devices.emplace(std::hash<std::string>{}(uniqueName), regDevice);
    return OH_NN_SUCCESS;
}

void DeviceManager::AddDevice(const std::string& deviceName, const std::string& vendorName,
    const std::string& version, std::shared_ptr<Device> device)
{
    std::string uniqueName = GenUniqueName(deviceName, vendorName, version);
    const std::lock_guard<std::mutex> lock(m_mtx);
    auto setResult = m_uniqueName.emplace(uniqueName);
    if (!setResult.second) {
        LOGW("Device already exists, cannot register again. deviceName=%s, vendorName=%s",
            deviceName.c_str(), vendorName.c_str());
        return;
    }

    m_devices.emplace(std::hash<std::string>{}(uniqueName), device);
}

void DeviceManager::DiscoverHDIDevices()
{
    std::string deviceName;
    std::string vendorName;
    std::string version;
    std::shared_ptr<Device> deviceV1_0 = DiscoverHDIDevicesV1_0(deviceName, vendorName, version);
    if (deviceV1_0 != nullptr) {
        AddDevice(deviceName, vendorName, version, deviceV1_0);
    }

    std::shared_ptr<Device> deviceV2_0 = DiscoverHDIDevicesV2_0(deviceName, vendorName, version);
    if (deviceV2_0 != nullptr) {
        AddDevice(deviceName, vendorName, version, deviceV2_0);
    }
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