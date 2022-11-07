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

#ifndef NEURAL_NETWORK_RUNTIME_TRANSFORM_H
#define NEURAL_NETWORK_RUNTIME_TRANSFORM_H

#include "hdi_interfaces.h"
#include "interfaces/kits/c/neural_network_runtime_type.h"
#include "cpp_type.h"
#include "mindir.h"
#include "mindir_types.h"
#include "ops_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
template<typename T>
std::vector<T> ConstructVectorFromArray(const T* data, size_t size)
{
    std::vector<T> array;
    if (data != nullptr) {
        array.assign(data, data + size);
    }
    return array;
}

uint32_t GetTypeSize(OH_NN_DataType type);


namespace HDIToNN {
OH_NN_DeviceType TransHDIDeviceType(const V1_0::DeviceType& iDeviceType);
DeviceStatus TransHDIDeviceStatus(const V1_0::DeviceStatus& iDeviceStatus);
} // namespace HDIToNN

namespace NNToHDI {
V1_0::PerformanceMode TransPerformanceMode(const OH_NN_PerformanceMode& mode);
V1_0::Priority TransPriority(const OH_NN_Priority& priority);
V1_0::DataType TransDataType(const OH_NN_DataType& dataType);
V1_0::Format TransFormat(const OH_NN_Format& format);
V1_0::IOTensor TransIOTensor(const IOTensor& tensor);
} // namespace NNToHDI

namespace NNToMS {
mindspore::lite::DataType TransformDataType(OH_NN_DataType type);
mindspore::lite::Format TransformFormat(OH_NN_Format type);
mindspore::lite::ActivationType TransfromFusionType(OH_NN_FuseType type);
mindspore::lite::QuantType TransformQuantType(OHOS::NeuralNetworkRuntime::Ops::OpsQuantType type);
mindspore::lite::PadMode TransformPadModeValue(int8_t padMode);
} // NNToMS

namespace MSToNN {
OH_NN_DataType TransformDataType(mindspore::lite::DataType type);
std::vector<QuantParam> TransformQuantParams(std::vector<mindspore::lite::QuantParam> msQuantParams);
} // namespace MSToNN
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_TRANSFORM_H