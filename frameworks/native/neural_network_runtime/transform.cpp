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

#include "transform.h"

#include "common/log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
const uint32_t BIT8_TO_BYTE = 1;
const uint32_t BIT16_TO_BYTE = 2;
const uint32_t BIT32_TO_BYTE = 4;
const uint32_t BIT64_TO_BYTE = 8;

uint32_t GetTypeSize(OH_NN_DataType type)
{
    switch (type) {
        case OH_NN_BOOL:
            return sizeof(bool);
        case OH_NN_INT8:
        case OH_NN_UINT8:
            return BIT8_TO_BYTE;
        case OH_NN_INT16:
        case OH_NN_UINT16:
        case OH_NN_FLOAT16:
            return BIT16_TO_BYTE;
        case OH_NN_INT32:
        case OH_NN_UINT32:
        case OH_NN_FLOAT32:
            return BIT32_TO_BYTE;
        case OH_NN_INT64:
        case OH_NN_UINT64:
        case OH_NN_FLOAT64:
            return BIT64_TO_BYTE;
        default:
            return 0;
    }
}

mindspore::lite::DataType NNToMS::TransformDataType(OH_NN_DataType type)
{
    switch (type) {
        case OH_NN_BOOL:
            return mindspore::lite::DATA_TYPE_BOOL;
        case OH_NN_INT8:
            return mindspore::lite::DATA_TYPE_INT8;
        case OH_NN_INT16:
            return mindspore::lite::DATA_TYPE_INT16;
        case OH_NN_INT32:
            return mindspore::lite::DATA_TYPE_INT32;
        case OH_NN_INT64:
            return mindspore::lite::DATA_TYPE_INT64;
        case OH_NN_UINT8:
            return mindspore::lite::DATA_TYPE_UINT8;
        case OH_NN_UINT16:
            return mindspore::lite::DATA_TYPE_UINT16;
        case OH_NN_UINT32:
            return mindspore::lite::DATA_TYPE_UINT32;
        case OH_NN_UINT64:
            return mindspore::lite::DATA_TYPE_UINT64;
        case OH_NN_FLOAT16:
            return mindspore::lite::DATA_TYPE_FLOAT16;
        case OH_NN_FLOAT32:
            return mindspore::lite::DATA_TYPE_FLOAT32;
        case OH_NN_FLOAT64:
            return mindspore::lite::DATA_TYPE_FLOAT64;
        default:
            return mindspore::lite::DATA_TYPE_UNKNOWN;
    }
}

mindspore::lite::Format NNToMS::TransformFormat(OH_NN_Format type)
{
    switch (type) {
        case OH_NN_FORMAT_NCHW:
            return mindspore::lite::FORMAT_NCHW;
        case OH_NN_FORMAT_NHWC:
            return mindspore::lite::FORMAT_NHWC;
        default:
            return mindspore::lite::FORMAT_NHWC;
    }
}

mindspore::lite::ActivationType NNToMS::TransfromFusionType(OH_NN_FuseType type)
{
    switch (type) {
        case OH_NN_FUSED_NONE:
            return mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION;
        case OH_NN_FUSED_RELU:
            return mindspore::lite::ACTIVATION_TYPE_RELU;
        case OH_NN_FUSED_RELU6:
            return mindspore::lite::ACTIVATION_TYPE_RELU6;
        default:
            return mindspore::lite::ACTIVATION_TYPE_UNKNOWN;
    }
}

mindspore::lite::QuantType NNToMS::TransformQuantType(OHOS::NeuralNetworkRuntime::Ops::OpsQuantType type)
{
    switch (type) {
        case OHOS::NeuralNetworkRuntime::Ops::OpsQuantType::QUANT_NONE:
            return mindspore::lite::QUANT_TYPE_NONE;
        case OHOS::NeuralNetworkRuntime::Ops::OpsQuantType::QUANT_ALL:
            return mindspore::lite::QUANT_TYPE_ALL;
        default: return mindspore::lite::QUANT_TYPE_NONE;
    }
}

mindspore::lite::PadMode NNToMS::TransformPadModeValue(int8_t padMode)
{
    // The value is an optional value of the int8_t type. The value 0 indicates the same,
    // and the value 1 indicates valid.
    return (padMode == 0) ? mindspore::lite::PadMode::PAD_MODE_SAME :
            mindspore::lite::PadMode::PAD_MODE_VALID;
}

OH_NN_DataType MSToNN::TransformDataType(mindspore::lite::DataType type)
{
    switch (type) {
        case mindspore::lite::DATA_TYPE_BOOL:
            return OH_NN_BOOL;
        case mindspore::lite::DATA_TYPE_INT8:
            return OH_NN_INT8;
        case mindspore::lite::DATA_TYPE_INT16:
            return OH_NN_INT16;
        case mindspore::lite::DATA_TYPE_INT32:
            return OH_NN_INT32;
        case mindspore::lite::DATA_TYPE_INT64:
            return OH_NN_INT64;
        case mindspore::lite::DATA_TYPE_UINT8:
            return OH_NN_UINT8;
        case mindspore::lite::DATA_TYPE_UINT16:
            return OH_NN_UINT16;
        case mindspore::lite::DATA_TYPE_UINT32:
            return OH_NN_UINT32;
        case mindspore::lite::DATA_TYPE_UINT64:
            return OH_NN_UINT64;
        case mindspore::lite::DATA_TYPE_FLOAT16:
            return OH_NN_FLOAT16;
        case mindspore::lite::DATA_TYPE_FLOAT32:
            return OH_NN_FLOAT32;
        case mindspore::lite::DATA_TYPE_FLOAT64:
            return OH_NN_FLOAT64;
        default:
            return OH_NN_UNKNOWN;
    }
}

std::vector<QuantParam> MSToNN::TransformQuantParams(std::vector<mindspore::lite::QuantParam> msQuantParams)
{
    std::vector<QuantParam> nnQuantParam;
    std::transform(msQuantParams.begin(), msQuantParams.end(), std::back_inserter(nnQuantParam),
        [](mindspore::lite::QuantParam quantParam) {
            return (QuantParam){quantParam.numBits, quantParam.scale, quantParam.zeroPoint};
        });
    return nnQuantParam;
}

OH_NN_Format MSToNN::TransformFormat(mindspore::lite::Format msFormat)
{
    if (msFormat == mindspore::lite::FORMAT_NHWC) {
        return OH_NN_Format::OH_NN_FORMAT_NHWC;
    } else if (msFormat == mindspore::lite::FORMAT_NCHW) {
        return OH_NN_Format::OH_NN_FORMAT_NCHW;
    }

    return OH_NN_Format::OH_NN_FORMAT_NONE;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS