/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#include "hdi_device_v2_1.h"
#include "hdi_returncode_utils_v2_1.h"
#include "log.h"
#include "utils.h"
#include "nnbackend.h"
#include "backend_registrar.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
void PrintRetLog(int32_t ret, int32_t nnrtSuccess, const std::string& makeName)
{
    if (ret < nnrtSuccess) {
            LOGW("%s failed. An error occurred in HDI, errorcode is %{public}d.", makeName.c_str(), ret);
        } else {
            OHOS::HDI::Nnrt::V2_1::NNRT_ReturnCode nnrtRet = static_cast<OHOS::HDI::Nnrt::V2_1::NNRT_ReturnCode>(ret);
            LOGW("%s failed. Errorcode is %{public}s.", makeName.c_str(), ConverterRetToString(nnrtRet).c_str());
        }
}

std::shared_ptr<Backend> HDIDeviceV2_1Creator()
{
    std::string deviceName;
    std::string vendorName;
    std::string version;

    // only one device from HDI now.
    OHOS::sptr<V2_1::INnrtDevice> iDevice = V2_1::INnrtDevice::Get();
    if (iDevice == nullptr) {
        LOGW("Get HDI device failed.");
        return nullptr;
    }

    auto ret = iDevice->GetDeviceName(deviceName);
    int32_t nnrtSuccess = static_cast<int32_t>(V2_1::NNRT_ReturnCode::NNRT_SUCCESS);
    if (ret != nnrtSuccess) {
        std::string makeName = "Get device name";
        PrintRetLog(ret, nnrtSuccess, makeName);
        return nullptr;
    }

    ret = iDevice->GetVendorName(vendorName);
    if (ret != nnrtSuccess) {
        std::string makeName = "Get vendor name";
        PrintRetLog(ret, nnrtSuccess, makeName);
        return nullptr;
    }

    std::pair<uint32_t, uint32_t> hdiVersion;
    ret = iDevice->GetVersion(hdiVersion.first, hdiVersion.second);
    if (ret != nnrtSuccess) {
        std::string makeName = "Get version";
        PrintRetLog(ret, nnrtSuccess, makeName);
        return nullptr;
    }
    version = 'v' + std::to_string(hdiVersion.first) + '_' + std::to_string(hdiVersion.second);
    const std::string& backendName = GenUniqueName(deviceName, vendorName, version);

    std::shared_ptr<Device> device = CreateSharedPtr<HDIDeviceV2_1>(iDevice);
    if (device == nullptr) {
        LOGW("Failed to create device, because fail to create device instance.");
        return nullptr;
    }

    std::shared_ptr<Backend> backend = CreateSharedPtr<NNBackend>(device, std::hash<std::string>{}(backendName));
    if (backend == nullptr) {
        LOGW("Failed to register backend, because fail to create backend.");
    }
    return backend;
}

REGISTER_BACKEND(HDIDeviceV2_1, HDIDeviceV2_1Creator)
} // namespace NeuralNetworkRuntime
} // namespace OHOS
