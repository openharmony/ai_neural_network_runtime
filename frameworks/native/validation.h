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

#ifndef NEURAL_NETWORK_RUNTIME_VALIDATION_H
#define NEURAL_NETWORK_RUNTIME_VALIDATION_H

#include "common/log.h"
#include "interfaces/kits/c/neural_network_runtime.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Validation {
template<typename T>
OH_NN_ReturnCode ValidateArray(const T* data, size_t size)
{
    if ((data != nullptr) != (size > 0)) {
        LOGE("ValidateArray failed, data is %p but the length is %zu", data, size);
        return OH_NN_INVALID_PARAMETER;
    }
    return OH_NN_SUCCESS;
}

bool ValidateTensorType(OH_NN_TensorType nnTensorType);
bool ValidateTensorDataType(OH_NN_DataType dataType);
bool ValidatePerformanceMode(OH_NN_PerformanceMode performanceMode);
bool ValidatePriority(OH_NN_Priority priority);
bool ValidateFuseType(OH_NN_FuseType fuseType);
bool ValidatePadMode(int8_t padMode);
} // namespace Validation
} // NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_VALIDATION_H
