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

#ifndef NEURAL_NETWORK_RUNTIME_DEVICE_MANAGER_H
#define NEURAL_NETWORK_RUNTIME_DEVICE_MANAGER_H

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <functional>
#include <mutex>

#include "device.h"
#include "interfaces/kits/c/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class DeviceManager {
public:
    const std::vector<size_t>& GetAllDeviceId();
    std::shared_ptr<Device> GetDevice(size_t deviceId) const;
    const std::string& GetDeviceName(size_t deviceId);

    // register device from C++ API
    OH_NN_ReturnCode RegisterDevice(std::function<std::shared_ptr<Device>()> creator);

    static DeviceManager& GetInstance()
    {
        static DeviceManager instance;
        instance.DiscoverHDIDevices();
        return instance;
    }

private:
    DeviceManager() = default;
    DeviceManager(const DeviceManager&) = delete;
    DeviceManager& operator=(const DeviceManager&) = delete;

    void DiscoverHDIDevices();
    std::string GenUniqueName(const std::string& deviceName, const std::string& vendorName) const;
    bool IsValidDevice(std::shared_ptr<Device> device) const;

private:
    std::unordered_set<std::string> m_uniqueName;
    // key is device id, it is the unique number.
    std::unordered_map<size_t, std::shared_ptr<Device>> m_devices;
    std::mutex m_mtx;

    std::string m_tmpDeviceName;
    std::vector<size_t> m_tmpDeviceIds;
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_DEVICE_MANAGER_H