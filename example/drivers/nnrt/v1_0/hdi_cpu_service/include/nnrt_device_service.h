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

#ifndef OHOS_HDI_NNRT_V1_0_NNRTDEVICESERVICE_H
#define OHOS_HDI_NNRT_V1_0_NNRTDEVICESERVICE_H

#include <memory>

#include "v1_0/innrt_device.h"
#include "ashmem.h"
#include "include/api/model.h"

#include "mindspore_schema/model_generated.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
class NnrtDeviceService : public INnrtDevice {
public:
    NnrtDeviceService() = default;
    virtual ~NnrtDeviceService();

    int32_t GetDeviceName(std::string& name) override;

    int32_t GetVendorName(std::string& name) override;

    int32_t GetDeviceType(DeviceType& deviceType) override;

    int32_t GetDeviceStatus(DeviceStatus& status) override;

    int32_t GetSupportedOperation(const Model& model, std::vector<bool>& ops) override;

    int32_t IsFloat16PrecisionSupported(bool& isSupported) override;

    int32_t IsPerformanceModeSupported(bool& isSupported) override;

    int32_t IsPrioritySupported(bool& isSupported) override;

    int32_t IsDynamicInputSupported(bool& isSupported) override;

    int32_t PrepareModel(const Model& model, const ModelConfig& config, sptr<IPreparedModel>& preparedModel) override;

    int32_t IsModelCacheSupported(bool& isSupported) override;

    int32_t PrepareModelFromModelCache(const std::vector<SharedBuffer>& modelCache, const ModelConfig& config,
         sptr<IPreparedModel>& preparedModel) override;

    int32_t AllocateBuffer(uint32_t length, SharedBuffer& buffer) override;

    int32_t ReleaseBuffer(const SharedBuffer& buffer) override;

private:
    int32_t ValidateModelConfig(const ModelConfig& config) const;
    int32_t ValidateModel(const Model& model) const;
    std::shared_ptr<mindspore::schema::MetaGraphT> TransModelToGraph(const Model& model) const;
    std::unique_ptr<mindspore::schema::TensorT> TransTensor(const Tensor& tensor) const;
    std::unique_ptr<mindspore::schema::CNodeT> TransNode(const Node& node) const;
    std::unique_ptr<mindspore::schema::SubGraphT> TransSubGraph(const SubGraph& graph, const size_t numTensor) const;
    std::shared_ptr<mindspore::Context> TransModelConfig(const ModelConfig& config) const;

private:
    std::shared_ptr<mindspore::Model> m_model {nullptr};
    std::unordered_map<int, sptr<Ashmem>> m_ashmems;
};
} // V1_0
} // Nnrt
} // HDI
} // OHOS

#endif // OHOS_HDI_NNRT_V1_0_NNRTDEVICESERVICE_H