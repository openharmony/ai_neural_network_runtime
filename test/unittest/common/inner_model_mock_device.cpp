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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/utils.h"
#include "frameworks/native/inner_model.h"
#include "frameworks/native/hdi_device.h"
#include "frameworks/native/device_manager.h"
#include "frameworks/native/ops/div_builder.h"
#include "mock_idevice.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
// Mock the palce where the devicemanager GetDevice is called in inner_model build function.
std::shared_ptr<Device> DeviceManager::GetDevice(size_t deviceId) const
{
    sptr<OHOS::HDI::Nnrt::V1_0::INnrtDevice> idevice =
        sptr<OHOS::HDI::Nnrt::V1_0::MockIDevice>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::MockIDevice());

    if (idevice == nullptr) {
        LOGE("DeviceManager mock GetDevice failed, error happened when new sptr");
        return nullptr;
    } else {
        std::shared_ptr<Device> device = CreateSharedPtr<HDIDevice>(idevice);
        if (device == nullptr) {
            LOGE("DeviceManager mock GetDevice failed, device is nullptr");
            return nullptr;
        }

        if (deviceId == 0) {
            return nullptr;
        } else {
            return device;
        }
    }
}

// Mock the palce where the operator GetPrimitive is called in inner_model build function.
Ops::LiteGraphPrimitvePtr Ops::DivBuilder::GetPrimitive()
{
    Ops::LiteGraphPrimitvePtr primitive = {nullptr, DestroyLiteGraphTensor};
    return primitive;
}

// Mock the palce where the device GetSupportedOperation is called in inner_model build function.
OH_NN_ReturnCode HDIDevice::GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
    std::vector<bool>& supportedOperations)
{
    supportedOperations = {true, true, true};

    if (model->name_ == "Loaded_NNR_Model") {
        return OH_NN_UNAVALIDABLE_DEVICE;
    } else {
        return OH_NN_SUCCESS;
    }
}
} // NeuralNetworkRuntime
} // OHOS
