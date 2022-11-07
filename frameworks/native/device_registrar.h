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

#ifndef NEURAL_NETWORK_RUNTIME_DEVICE_REGISTRAR_H
#define NEURAL_NETWORK_RUNTIME_DEVICE_REGISTRAR_H

#include <vector>
#include <memory>
#include <functional>

#include "device.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
using CreateDevice = std::function<std::shared_ptr<Device>()>;

class DeviceRegistrar {
public:
    DeviceRegistrar(const CreateDevice creator);
    ~DeviceRegistrar() = default;
};

#define REGISTER_DEVICE(deviceName, vendorName, creator)                                                         \
    namespace {                                                                                                  \
    static OHOS::NeuralNetworkRuntime::DeviceRegistrar g_##deviceName##_##vendorName##_device_registrar(creator) \
    } // namespace
} // namespace NeuralNetworkRuntime
} // OHOS
#endif // NEURAL_NETWORK_RUNTIME_DEVICE_REGISTRAR_H