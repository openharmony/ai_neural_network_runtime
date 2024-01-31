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

#ifndef NEURAL_NETWORK_RUNTIME_HDI_DEVICE_V2_1_H
#define NEURAL_NETWORK_RUNTIME_HDI_DEVICE_V2_1_H

#include <v2_1/nnrt_types.h>
#include <v2_1/innrt_device.h>
#include <v2_1/iprepared_model.h>
#include "refbase.h"

#include "device.h"
#include "hdi_device_v2_0.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace V2_1 = OHOS::HDI::Nnrt::V2_1;
class HDIDeviceV2_1 : public HDIDeviceV2_0 {
public:
    explicit HDIDeviceV2_1(OHOS::sptr<V2_1::INnrtDevice> device);

private:
    OHOS::sptr<V2_1::INnrtDevice> m_iDevice {nullptr};
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_HDI_DEVICE_V2_1_H
