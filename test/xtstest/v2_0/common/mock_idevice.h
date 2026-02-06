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
#ifndef MOCK_IDEVICE_H
#define MOCK_IDEVICE_H

#include <iostream>
#include <hdi_support.h>
#include <string_ex.h>
#include <hdf_base.h>
#include <unordered_set>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "mindir_lite_graph.h"
#include "mindir.h"

#include "securec.h"
#include "refbase.h"
#include "log.h"
#include "ashmem.h"

#include <v2_0/nnrt_types.h>
#include <v2_0/innrt_device.h>
#include <v2_0/iprepared_model.h>

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V2_0 {

class MockIDevice : public INnrtDevice {
public:
    int32_t GetSupportedOperation(const Model& model, std::vector<bool>& ops) override;

    int32_t IsFloat16PrecisionSupported(bool& isSupported) override;

    int32_t IsPerformanceModeSupported(bool& isSupported) override;

    int32_t IsPrioritySupported(bool& isSupported) override;

    int32_t IsDynamicInputSupported(bool& isSupported) override;

    int32_t IsModelCacheSupported(bool& isSupported) override;

    int32_t AllocateBuffer(uint32_t length, SharedBuffer &buffer) override;

    int32_t ReleaseBuffer(const SharedBuffer &buffer) override;

    int32_t GetDeviceName(std::string& name) override;

    int32_t GetVendorName(std::string& name) override;

    int32_t GetDeviceType(DeviceType& deviceType) override;

    int32_t GetDeviceStatus(DeviceStatus& status) override;

    int32_t GetVersion(uint32_t &majorVersion, uint32_t &minorVersion) override;

    int32_t PrepareModel(const Model& model, const ModelConfig& config, sptr<IPreparedModel>& preparedModel) override;

    int32_t PrepareOfflineModel(const std::vector<SharedBuffer>& offlineModels, const ModelConfig& config,
        sptr<OHOS::HDI::Nnrt::V2_0::IPreparedModel>& preparedModel) override;

    int32_t PrepareModelFromModelCache(const std::vector<SharedBuffer>& modelCache, const ModelConfig& config,
         sptr<IPreparedModel>& preparedModel) override;

    int32_t MemoryCopy(float* data, uint32_t length);

    void SetFP16Supported(bool isSupported);

    void SetPerformanceSupported(bool isSupported);

    void SetPrioritySupported(bool isSupported);

    void SetModelCacheSupported(bool isSupported);

    void SetOperationsSupported(std::vector<bool> isSupported);

    void SetDynamicInputSupported(bool isSupported);

    static MockIDevice *GetInstance();

    MockIDevice();
    virtual ~MockIDevice();

private:
    std::unordered_set<int> m_fds;
    int m_bufferFd;
    bool m_fp16 = true;
    bool m_performance = true;
    bool m_priority = true;
    bool m_cache = true;
    bool m_dynamic = true;
    std::vector<bool> m_operations{true};
    std::mutex m_mtx;
};

class MockIPreparedModel : public IPreparedModel {
public:
    int32_t ExportModelCache(std::vector<SharedBuffer>& modelCache) override;
    int32_t Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims) override;
    int32_t GetInputDimRanges(
        std::vector<std::vector<uint32_t>>& minInputDims, std::vector<std::vector<uint32_t>>& maxInputDims) override;
    int32_t GetVersion(uint32_t &majorVersion, uint32_t &minorVersion) override;
    MockIPreparedModel() = default;
    virtual ~MockIPreparedModel();
private:
    std::unordered_set<int> m_fds;
};

} // namespace V2_0
} // namespace Nnrt
} // namespace HDI
} // namespace OHOS
#endif // MOCK_IDEVICE_H
