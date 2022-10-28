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

#include "mindir_types.h"

#include "validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Validation {
bool ValidateTensorDataType(OH_NN_DataType dataType)
{
    if (dataType >= OH_NN_UNKNOWN && dataType <= OH_NN_FLOAT64) {
        return true;
    }
    return false;
}

bool ValidatePerformanceMode(OH_NN_PerformanceMode performanceMode)
{
    if ((performanceMode >= OH_NN_PERFORMANCE_NONE) && (performanceMode <= OH_NN_PERFORMANCE_EXTREME)) {
        return true;
    }
    return false;
}

bool ValidatePriority(OH_NN_Priority priority)
{
    if ((priority >= OH_NN_PRIORITY_NONE) && (priority <= OH_NN_PRIORITY_HIGH)) {
        return true;
    }
    return false;
}

bool ValidateFuseType(OH_NN_FuseType fuseType)
{
    if ((fuseType >= OH_NN_FUSED_NONE) && (fuseType <= OH_NN_FUSED_RELU6)) {
        return true;
    }
    return false;
}

bool ValidatePadMode(int8_t padMode)
{
    if ((padMode >= mindspore::lite::PAD_MODE_PAD) && (padMode <= mindspore::lite::PAD_MODE_VALID)) {
        return true;
    }
    return false;
}

bool ValidateTensorType(OH_NN_TensorType nnTensorType)
{
    if ((nnTensorType >= OH_NN_TENSOR) && (nnTensorType <= OH_NN_UNSQUEEZE_AXIS)) {
        return true;
    }
    return false;
}
} // namespace Validation
} // namespace NeuralNetworkRuntime
} // namespace OHOS
