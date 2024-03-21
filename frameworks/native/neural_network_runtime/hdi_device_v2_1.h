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

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace V2_1 = OHOS::HDI::Nnrt::V2_1;
class HDIDeviceV2_1 : public Device {
public:
    explicit HDIDeviceV2_1(OHOS::sptr<V2_1::INnrtDevice> device);

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
                                  const Buffer& quantBuffer,
                                  const ModelConfig& config,
                                  std::shared_ptr<PreparedModel>& preparedModel) override;
    OH_NN_ReturnCode PrepareModel(const void* metaGraph,
                                  const Buffer& quantBuffer,
                                  const ModelConfig& config,
                                  std::shared_ptr<PreparedModel>& preparedModel) override;
    OH_NN_ReturnCode PrepareModelFromModelCache(const std::vector<Buffer>& modelCache,
                                                const ModelConfig& config,
                                                std::shared_ptr<PreparedModel>& preparedModel) override;
    OH_NN_ReturnCode PrepareOfflineModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                         const ModelConfig& config,
                                         std::shared_ptr<PreparedModel>& preparedModel) override;

    void* AllocateBuffer(size_t length) override;
    void* AllocateTensorBuffer(size_t length, std::shared_ptr<TensorDesc> tensor) override;
    void* AllocateTensorBuffer(size_t length, std::shared_ptr<NNTensor> tensor) override;
    OH_NN_ReturnCode ReleaseBuffer(const void* buffer) override;

    OH_NN_ReturnCode AllocateBuffer(size_t length, int& fd) override;
    OH_NN_ReturnCode ReleaseBuffer(int fd, size_t length) override;

private:
    OH_NN_ReturnCode ReleaseSharedBuffer(const V2_1::SharedBuffer& buffer);
    OH_NN_ReturnCode GetOfflineModelFromLiteGraph(std::shared_ptr<const mindspore::lite::LiteGraph> graph,
                                                  std::vector<std::vector<uint8_t>>& offlineModels);
    OH_NN_ReturnCode AllocateDeviceBufferForOfflineModel(const std::vector<std::vector<uint8_t>>& offlineModels,
                                                         std::vector<Buffer>& deviceBuffers);
    OH_NN_ReturnCode CopyOfflineModelToDevice(const std::vector<std::vector<uint8_t>>& offlineModels,
                                              std::vector<Buffer>& deviceBuffers);
    OH_NN_ReturnCode PrepareOfflineModel(std::vector<Buffer>& deviceBuffers,
                                         const ModelConfig& config,
                                         const std::map<std::string, std::vector<int8_t>>& extensions,
                                         std::shared_ptr<PreparedModel>& preparedModel);

private:
    // first: major version, second: minor version
    std::pair<uint32_t, uint32_t> m_hdiVersion;
    OHOS::sptr<V2_1::INnrtDevice> m_iDevice {nullptr};
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_HDI_DEVICE_V2_1_H
