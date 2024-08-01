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

#include "mock_idevice.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V2_0 {
sptr<INnrtDevice> INnrtDevice::Get(bool isStub)
{
    return INnrtDevice::Get("device_service", isStub);
}

sptr<INnrtDevice> INnrtDevice::Get(const std::string& serviceName, bool isStub)
{
    if (isStub) {
        return nullptr;
    }

    sptr<INnrtDevice> mockIDevice = sptr<MockIDevice>(new (std::nothrow) MockIDevice());
    if (mockIDevice == nullptr) {
        return nullptr;
    }
    return mockIDevice;
}
} // V2_0
} // Nnrt
} // HDI
} // OHOS