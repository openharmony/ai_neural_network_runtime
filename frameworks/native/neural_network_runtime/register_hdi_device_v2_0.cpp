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

#include "hdi_device_v2_0.h"
#include "hdi_returncode_utils.h"
#include "common/log.h"
#include "common/utils.h"
#include "nnbackend.h"
#include "backend_registrar.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
std::shared_ptr<Backend> HDIDeviceV2_0Creator()
{
    std::string deviceName;
    std::string vendorName;
    std::string version;

    // only one device from HDI now.
    OHOS::sptr<V2_0::INnrtDevice> iDevice = V2_0::INnrtDevice::Get();
    if (iDevice == nullptr) {
        LOGW("Get HDI device failed.");
        return nullptr;
    }

    auto ret = iDevice->GetDeviceName(deviceName);
    int32_t nnrtSuccess = static_cast<int32_t>(V2_0::NNRT_ReturnCode::NNRT_SUCCESS);
    if (ret != nnrtSuccess) {
        if (ret < nnrtSuccess) {
            LOGW("Get device name failed. An error occurred in HDI, errorcode is %{public}d.", ret);
        } else {
            OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode nnrtRet = static_cast<OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode>(ret);
            LOGW("Get device name failed. Errorcode is %{public}s.", ConverterRetToString(nnrtRet).c_str());
        }
        return nullptr;
    }

    ret = iDevice->GetVendorName(vendorName);
    if (ret != nnrtSuccess) {
        if (ret < nnrtSuccess) {
            LOGW("Get vendor name failed. An error occurred in HDI, errorcode is %{public}d.", ret);
        } else {
            OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode nnrtRet = static_cast<OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode>(ret);
            LOGW("Get vendor name failed. Errorcode is %{public}s.", ConverterRetToString(nnrtRet).c_str());
        }
        return nullptr;
    }

    std::pair<uint32_t, uint32_t> hdiVersion;
    ret = iDevice->GetVersion(hdiVersion.first, hdiVersion.second);
    if (ret != nnrtSuccess) {
        if (ret < nnrtSuccess) {
            LOGW("Get version failed. An error occurred in HDI, errorcode is %{public}d.", ret);
        } else {
            OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode nnrtRet = static_cast<OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode>(ret);
            LOGW("Get version failed. Errorcode is %{public}s.", ConverterRetToString(nnrtRet).c_str());
        }
        return nullptr;
    }
    version = 'v' + std::to_string(hdiVersion.first) + '_' + std::to_string(hdiVersion.second);
    const std::string& backendName = GenUniqueName(deviceName, vendorName, version);

    std::shared_ptr<Device> device = CreateSharedPtr<HDIDeviceV2_0>(iDevice);
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

REGISTER_BACKEND(HDIDeviceV2_0, HDIDeviceV2_0Creator)
} // namespace NeuralNetworkRuntime
} // namespace OHOS
