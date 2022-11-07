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

#ifndef NEURAL_NETWORK_RUNTIME_DEVICE_H
#define NEURAL_NETWORK_RUNTIME_DEVICE_H

#include <string>
#include <vector>
#include <memory>

#include "interfaces/kits/c/neural_network_runtime_type.h"
#include "cpp_type.h"
#include "prepared_model.h"
#include "mindir.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class Device {
public:
    Device() = default;
    virtual ~Device() = default;

    virtual OH_NN_ReturnCode GetDeviceName(std::string& name) = 0;
    virtual OH_NN_ReturnCode GetVendorName(std::string& name) = 0;
    virtual OH_NN_ReturnCode GetDeviceType(OH_NN_DeviceType& deviceType) = 0;
    virtual OH_NN_ReturnCode GetDeviceStatus(DeviceStatus& status) = 0;
    virtual OH_NN_ReturnCode GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                                   std::vector<bool>& ops) = 0;

    virtual OH_NN_ReturnCode IsFloat16PrecisionSupported(bool& isSupported) = 0;
    virtual OH_NN_ReturnCode IsPerformanceModeSupported(bool& isSupported) = 0;
    virtual OH_NN_ReturnCode IsPrioritySupported(bool& isSupported) = 0;
    virtual OH_NN_ReturnCode IsDynamicInputSupported(bool& isSupported) = 0;
    virtual OH_NN_ReturnCode IsModelCacheSupported(bool& isSupported) = 0;

    virtual OH_NN_ReturnCode PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                          const ModelConfig& config,
                                          std::shared_ptr<PreparedModel>& preparedModel) = 0;
    virtual OH_NN_ReturnCode PrepareModelFromModelCache(const std::vector<ModelBuffer>& modelCache,
                                                        const ModelConfig& config,
                                                        std::shared_ptr<PreparedModel>& preparedModel) = 0;

    virtual void* AllocateBuffer(size_t length) = 0;
    virtual OH_NN_ReturnCode ReleaseBuffer(const void* buffer) = 0;
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_DEVICE_H