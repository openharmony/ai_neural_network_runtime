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

#include "device_registrar.h"

#include "device_manager.h"
#include "common/log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
DeviceRegistrar::DeviceRegistrar(const CreateDevice creator)
{
    auto& deviceManager = DeviceManager::GetInstance();
    auto ret = deviceManager.RegisterDevice(creator);
    if (ret != OH_NN_SUCCESS) {
        LOGW("Register device failed. ErrorCode=%d", ret);
    }
}
} // NeuralNetworkRuntime
} // OHOS