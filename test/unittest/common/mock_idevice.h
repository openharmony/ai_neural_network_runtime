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

#ifndef NEURAL_NETWORK_RUNTIME_MOCK_IDEVICE_H
#define NEURAL_NETWORK_RUNTIME_MOCK_IDEVICE_H

#include <gmock/gmock.h>

#include "frameworks/native/hdi_prepared_model.h"
#include "frameworks/native/memory_manager.h"
#include "frameworks/native/transform.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
class MockIDevice : public INnrtDevice {
public:
    MOCK_METHOD1(GetDeviceName, int32_t(std::string&));
    MOCK_METHOD1(GetVendorName, int32_t(std::string&));
    MOCK_METHOD1(GetDeviceType, int32_t(DeviceType&));
    MOCK_METHOD1(GetDeviceStatus, int32_t(DeviceStatus&));
    MOCK_METHOD2(GetSupportedOperation, int32_t(const Model&, std::vector<bool>&));
    MOCK_METHOD1(IsFloat16PrecisionSupported, int32_t(bool&));
    MOCK_METHOD1(IsPerformanceModeSupported, int32_t(bool&));
    MOCK_METHOD1(IsPrioritySupported, int32_t(bool&));
    MOCK_METHOD1(IsDynamicInputSupported, int32_t(bool&));
    MOCK_METHOD3(PrepareModel, int32_t(const Model&, const ModelConfig&, OHOS::sptr<IPreparedModel>&));
    MOCK_METHOD1(IsModelCacheSupported, int32_t(bool&));
    MOCK_METHOD3(PrepareModelFromModelCache, int32_t(const std::vector<SharedBuffer>&, const ModelConfig&,
        OHOS::sptr<IPreparedModel>&));
    MOCK_METHOD2(AllocateBuffer, int32_t(uint32_t, SharedBuffer&));
    MOCK_METHOD1(ReleaseBuffer, int32_t(const SharedBuffer&));
    MOCK_METHOD2(GetVersion, int32_t(uint32_t&, uint32_t&));
};

class MockIPreparedModel : public IPreparedModel {
public:
    MOCK_METHOD1(ExportModelCache, int32_t(std::vector<SharedBuffer>&));
    MOCK_METHOD4(Run, int32_t(const std::vector<IOTensor>&, const std::vector<IOTensor>&,
        std::vector<std::vector<int32_t>>&, std::vector<bool>&));
    MOCK_METHOD2(GetVersion, int32_t(uint32_t&, uint32_t&));

    static OH_NN_ReturnCode m_ExpectRetCode;
};
} // V1_0
} // Nnrt
} // HDI
} // OHOS
#endif // NEURAL_NETWORK_RUNTIME_MOCK_IDEVICE_H
