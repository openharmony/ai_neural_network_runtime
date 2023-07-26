/*
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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
#ifndef OHOS_HDI_NNRT_V2_0_NNRTDEVICEIMPL_H
#define OHOS_HDI_NNRT_V2_0_NNRTDEVICEIMPL_H
#include"../../common/log.h"

#include "v2_0/innrt_device.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V2_0 {
class NnrtDeviceImpl : public INnrtDevice {
public:
    NnrtDeviceImpl() = default;
    virtual ~NnrtDeviceImpl() = default;

    int32_t GetDeviceName(std::string& name)
    {
        LOGI("Get device name.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t GetVendorName(std::string& name)
    {
        LOGI("Get vendor name.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t GetDeviceType(DeviceType& deviceType)
    {
        LOGI("Get device type.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t GetDeviceStatus(DeviceStatus& status)
    {
        LOGI("Get device status.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t GetSupportedOperation(const Model& model, std::vector<bool>& ops)
    {
        LOGI("Get supported operation.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t IsFloat16PrecisionSupported(bool& isSupported)
    {
        LOGI("Is float16 precision support.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t IsPerformanceModeSupported(bool& isSupported)
    {
        LOGI("Is performance mode support.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t IsPrioritySupported(bool& isSupported)
    {
        LOGI("Is priority support.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t IsDynamicInputSupported(bool& isSupported)
    {
        LOGI("Is performance mode support.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t PrepareModel(const Model& model, const ModelConfig& config, sptr<IPreparedModel>& preparedModel)
    {
        LOGI("Prepare model.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t PrepareOfflineModel(const std::vector<SharedBuffer>& modelBuffer, const ModelConfig& config,
        sptr<IPreparedModel>& preparedModel)
    {
        LOGI("Prepare offline model.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t IsModelCacheSupported(bool& isSupported)
    {
        LOGI("Is model cache support.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t PrepareModelFromModelCache(const std::vector<SharedBuffer>& modelCache, const ModelConfig& config,
            sptr<IPreparedModel>& preparedModel)
    {
        LOGI("Prepare model from model cache.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t AllocateBuffer(uint32_t length, SharedBuffer& buffer)
    {
        LOGI("Allocate buffer.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t ReleaseBuffer(const SharedBuffer& buffer)
    {
        LOGI("Release buffer.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }
};
} // V2_0
} // Nnrt
} // HDI
} // OHOS

#endif // OHOS_HDI_NNRT_V2_0_NNRTDEVICEIMPL_H