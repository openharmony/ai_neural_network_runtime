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

#ifndef TENSORFLOW_LITE_DELEGATES_NNRT_TENSOR_MAPPING_H
#define TENSORFLOW_LITE_DELEGATES_NNRT_TENSOR_MAPPING_H

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/minimal_logging.h"

#include "nnrt_utils.h"

namespace tflite {
namespace delegate {
namespace nnrt {
constexpr uint32_t QUANT_NUMBITS = 8;

class TensorMapping {
public:
    // Given a TFLite index return the NN index. If it doesn't exist
    // return -1.
    int32_t LiteIndexToNn(int32_t index) const
    {
        const int64_t maxSize = m_liteTensorToNnTensor.size();
        if (index >= 0 && index < maxSize) {
            return m_liteTensorToNnTensor[index];
        } else {
            return INVALID_INDEX;
        }
    }

    // NN API uses non tensor tensors instead of structs. This creates one
    // and returns the index. It uses a std::vector and resizes it as needed
    // keeping -1 to unmapped values. Intermediate tensors likely will not
    // be mapped.
    const int32_t AddNewNonTensorTensor()
    {
        return m_nextNnTensorIndex++;
    }

    // Add a new mapping from `tfliteIndex` and return the NN API tensor index.
    int32_t AddNewNnTensorIndex(int32_t tfliteIndex)
    {
        const int64_t currentSize = m_liteTensorToNnTensor.size();
        if (tfliteIndex >= currentSize) {
            m_liteTensorToNnTensor.resize(tfliteIndex + 1, INVALID_INDEX);
        }
        const int32_t newTensorIndex = m_nextNnTensorIndex++;
        m_liteTensorToNnTensor[tfliteIndex] = newTensorIndex;
        return newTensorIndex;
    }

    // Get nn tensor tensor tensor num.
    int32_t GetTensorTensorNum() const
    {
        return m_nextNnTensorIndex;
    }

    // Given a TFLite index returns a TFLite type to which a tensor must be
    // converted during copying the data to the memory allocated for NN API.
    // kTfLiteNoType means no conversion is needed.
    TfLiteType GetEqualLiteTypeFromLiteIndex(int32_t index) const
    {
        const int64_t maxSize = m_indexToTypeConversion.size();
        if (index >= 0 && index < maxSize)
            return m_indexToTypeConversion[index];
        else
            return kTfLiteNoType;
    }

    // Add a new mapping from TFLite index to a type conversion.
    void AddTypeConversion(int32_t tfliteIndex, TfLiteType tfliteType)
    {
        const int64_t currentSize = m_indexToTypeConversion.size();
        if (tfliteIndex >= currentSize) {
            m_indexToTypeConversion.resize(tfliteIndex + 1, kTfLiteNoType);
        }
        m_indexToTypeConversion[tfliteIndex] = tfliteType;
    }

    // Convert TFLite tensor quant params to NNRT tensor quant params
    TfLiteStatus ConvertQuantParams(TfLiteContext* context, int32_t tensorIndex, OH_NN_QuantParam& quantParam)
    {
        TfLiteTensor* tensor = &(context->tensors[tensorIndex]);
        TfLiteType tfType = tensor->type;
        if ((tfType != kTfLiteFloat32) && (tfType != kTfLiteFloat16) && (tfType != kTfLiteBool) &&
            (tfType != kTfLiteInt32) && (tfType != kTfLiteUInt8) && (tfType != kTfLiteInt8)) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                "[TENSOR_MAPPING] type %s is not supported.", TfLiteTypeGetName(tensor->type));
            return kTfLiteError;
        }

        if (tensor->quantization.type) {
            TfLiteAffineQuantization* params = reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
            int number = params->scale->size;
            std::vector<double> scale;
            for (int i = 0; i < number; ++i) {
                scale.emplace_back(static_cast<double>(params->scale->data[i]));
            }
            m_scale.emplace_back(scale);
            quantParam.scale = m_scale.back().data();
            quantParam.zeroPoint = params->zero_point->data;
            quantParam.quantCount = number;
            m_numBits.emplace_back(number, QUANT_NUMBITS);
            quantParam.numBits = m_numBits.back().data();
        } else {
            quantParam.quantCount = 0;
        }

        return kTfLiteOk;
    }

    // Convert TFLite tensor type to NNRT tensor type
    TfLiteStatus ConvertType(TfLiteContext* context, int32_t tensorIndex, int32_t tensorFlags, OH_NN_DataType& nnType)
    {
        const bool scalarAsTensor = tensorFlags & NN_TENSOR_FLAG_SCALAR_AS_TENSOR;
        TfLiteTensor* tensor = &(context->tensors[tensorIndex]);
        TfLiteType nnTypeEquivalent = GetEqualLiteTypeFromLiteIndex(tensorIndex);
        if (tensor->type == kTfLiteFloat32) {
            nnType = OH_NN_FLOAT32;
        } else if (tensor->type == kTfLiteFloat16) {
            nnType = OH_NN_FLOAT16;
            if (scalarAsTensor) {
                nnType = OH_NN_FLOAT32;
                AddTypeConversion(tensorIndex, kTfLiteFloat32);
            }
        } else if (tensor->type == kTfLiteInt32) {
            nnType = OH_NN_INT32;
        } else if (tensor->type == kTfLiteBool) {
            nnType = OH_NN_INT8;
        } else if (tensor->type == kTfLiteUInt8) {
            nnType = (nnTypeEquivalent == kTfLiteInt32) ? OH_NN_INT32 : OH_NN_INT8;
        } else if (tensor->type == kTfLiteInt8) {
            nnType = (nnTypeEquivalent == kTfLiteInt32) ? OH_NN_INT32 : OH_NN_UINT8;
        } else {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
                "[TENSOR_MAPPING] type %s is not supported.", TfLiteTypeGetName(tensor->type));
            return kTfLiteError;
        }

        return kTfLiteOk;
    }

private:
    // Next index of nnrt tensor
    int32_t m_nextNnTensorIndex = 0;

    // Mapping from lite index. Use a std::vector for speed and code size
    // rather than a map.
    std::vector<int32_t> m_liteTensorToNnTensor;

    // Mapping from lite index to a type which tensor must be converted to during
    // the copying of the data to the memory allocated for NN API. kTfLiteNoType
    // means no conversion is needed. Use an std::vector for speed and code size
    // rather than a map.
    std::vector<TfLiteType> m_indexToTypeConversion;

    std::vector<std::vector<uint32_t>> m_numBits;

    std::vector<std::vector<double>> m_scale;
};
} // namespace nnrt
} // namespace delegate
} // namespace tflite

#endif // TENSORFLOW_LITE_DELEGATES_NNRT_TENSOR_MAPPING_H