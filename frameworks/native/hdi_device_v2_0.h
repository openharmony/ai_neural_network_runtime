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

#ifndef NEURAL_NETWORK_RUNTIME_HDI_DEVICE_V2_0_H
#define NEURAL_NETWORK_RUNTIME_HDI_DEVICE_V2_0_H

#include <v2_0/nnrt_types.h>
#include <v2_0/innrt_device.h>
#include <v2_0/iprepared_model.h>
#include "refbase.h"

#include "device.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace V2_0 = OHOS::HDI::Nnrt::V2_0;
class HDIDeviceV2_0 : public Device {
public:
    explicit HDIDeviceV2_0(OHOS::sptr<V2_0::INnrtDevice> device);

    OH_NN_ReturnCode GetDeviceName(std::string& name) override;
    OH_NN_ReturnCode GetVendorName(std::string& name) override;
    OH_NN_ReturnCode GetVersion(std::string& version) override;
    OH_NN_ReturnCode GetDeviceType(OH_NN_DeviceType& deviceType) override;
    OH_NN_ReturnCode GetDeviceStatus(DeviceStatus& status) override;
    OH_NN_ReturnCode GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                           std::vector<bool>& ops) override;

    OH_NN_ReturnCode IsFloat16PrecisionSupported(bool& isSupported) override;
    OH_NN_ReturnCode IsPerformanceModeSupported(bool& isSupported) override;
    OH_NN_ReturnCode IsPrioritySupported(bool& isSupported) override;
    OH_NN_ReturnCode IsDynamicInputSupported(bool& isSupported) override;
    OH_NN_ReturnCode IsModelCacheSupported(bool& isSupported) override;

    OH_NN_ReturnCode PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                  const ModelConfig& config,
                                  std::shared_ptr<PreparedModel>& preparedModel) override;
    OH_NN_ReturnCode PrepareModelFromModelCache(const std::vector<ModelBuffer>& modelCache,
                                                const ModelConfig& config,
                                                std::shared_ptr<PreparedModel>& preparedModel) override;

    void* AllocateBuffer(size_t length) override;
    OH_NN_ReturnCode ReleaseBuffer(const void* buffer) override;

private:
    OH_NN_ReturnCode ReleaseSharedBuffer(const V2_0::SharedBuffer& buffer);

private:
    // first: major version, second: minor version
    std::pair<uint32_t, uint32_t> m_hdiVersion;
    OHOS::sptr<V2_0::INnrtDevice> m_iDevice {nullptr};
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_HDI_DEVICE_V2_0_H