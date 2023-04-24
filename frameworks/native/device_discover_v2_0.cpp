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

#include "device_discover.h"
#include "hdi_device_v2_0.h"
#include "hdi_returncode_utils.h"
#include "common/log.h"
#include "common/utils.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
std::shared_ptr<Device> DiscoverHDIDevicesV2_0(std::string& deviceName, std::string& vendorName, std::string& version)
{
    // only one device from HDI now.
    OHOS::sptr<V2_0::INnrtDevice> iDevice = V2_0::INnrtDevice::Get();
    if (iDevice == nullptr) {
        LOGW("Get HDI device failed.");
        return nullptr;
    }

    auto ret = iDevice->GetDeviceName(deviceName);
    if (ret != V2_0::NNRT_ReturnCode::NNRT_SUCCESS) {
        if (ret < 0) {
            LOGW("Get device name failed. An error occurred in HDI, errorcode is %{public}d.", ret);
        } else if (ret > 0) {
            OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode nnrtRet = static_cast<OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode>(ret);
            LOGW("Get device name failed. Errorcode is %{public}s.", ConverterRetToString(nnrtRet).c_str());
        }
        return nullptr;
    }

    ret = iDevice->GetVendorName(vendorName);
    if (ret != V2_0::NNRT_ReturnCode::NNRT_SUCCESS) {
        if (ret < 0) {
            LOGW("Get vendor name failed. An error occurred in HDI, errorcode is %{public}d.", ret);
        } else if (ret > 0) {
            OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode nnrtRet = static_cast<OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode>(ret);
            LOGW("Get vendor name failed. Errorcode is %{public}s.", ConverterRetToString(nnrtRet).c_str());
        }
        return nullptr;
    }

    std::pair<uint32_t, uint32_t> hdiVersion;
    ret = iDevice->GetVersion(hdiVersion.first, hdiVersion.second);
    if (ret != V2_0::NNRT_ReturnCode::NNRT_SUCCESS) {
        if (ret < 0) {
            LOGW("Get version failed. An error occurred in HDI, errorcode is %{public}d.", ret);
        } else if (ret > 0) {
            OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode nnrtRet = static_cast<OHOS::HDI::Nnrt::V2_0::NNRT_ReturnCode>(ret);
            LOGW("Get version failed. Errorcode is %{public}s.", ConverterRetToString(nnrtRet).c_str());
        }
        return nullptr;
    }
    version = 'v' + std::to_string(hdiVersion.first) + '_' + std::to_string(hdiVersion.second);

    std::shared_ptr<Device> device = CreateSharedPtr<HDIDeviceV2_0>(iDevice);
    if (device == nullptr) {
        LOGW("Failed to register device, because fail to create device instance.");
    }
    return device;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS