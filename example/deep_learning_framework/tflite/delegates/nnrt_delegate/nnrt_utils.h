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

#ifndef TENSORFLOW_LITE_DELEGATES_NNRT_UTILS_H
#define TENSORFLOW_LITE_DELEGATES_NNRT_UTILS_H

#include <map>
#include <vector>
#include <unordered_map>

#include "nnrt_delegate.h"

namespace tflite {
constexpr int32_t DEPTHWISE_WEIGHT_BATCH_DIMENSION = 0;
constexpr int32_t DEPTHWISE_WEIGHT_HEIGHT_DIMENSION = 1;
constexpr int32_t DEPTHWISE_WEIGHT_WIDTH_DIMENSION = 2;
constexpr int32_t DEPTHWISE_WEIGHT_CHANNEL_DIMENSION = 3;
constexpr int32_t DEPTHWISE_WEIGHT_DIMENSION_COUNT = 4;
const std::string NNRT_REFERENCE_DEVICE = "nnrt-reference";

// Bit mask for tensor flags.
enum BIT_MASK {
    NN_TENSOR_FLAG_SCALAR_AS_TENSOR = 1U << 0,
    NN_TENSOR_FLAG_INT8_CONVERSION = 1U << 1,
    NN_TENSOR_FLAG_USE_INT8_ASYMM_SIGNED = 1U << 2,
    NN_TENSOR_FLAG_FORCE_PER_CHANNEL = 1U << 3,
    NN_TENSOR_FLAG_HALF_TO_FLOAT_CONVERSION = 1U << 4,
};

// Returns the enum name corresponding to the given error code if the given
// value corresponds to an of the error codes in the enumeration above or
// an message with the unknown code.
// LINT.IfChange(NnrtErrorDescription)
extern std::string NnrtErrorDescription(int32_t errorCode);

#define RETURN_TFLITE_ERROR_IF_NN_ERROR(code, callDesc)                                            \
    do {                                                                                           \
        if ((code) != OH_NN_SUCCESS) {                                                             \
            const auto errorDesc = NnrtErrorDescription((code));                                   \
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "NN API returned error %s at line %d while %s.\n",   \
                            errorDesc.c_str(), __LINE__, (callDesc));                              \
            return kTfLiteError;                                                                   \
        }                                                                                          \
    } while (0)

#define RETURN_TFLITE_ERROR_IF_NN_ERROR_FOR_TENSOR(code, callDesc, pTensor)                                          \
    do {                                                                                                             \
        if ((code) != OH_NN_SUCCESS) {                                                                               \
            const auto errorDesc = NnrtErrorDescription((code));                                                     \
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR,                                                                        \
                            "NN API returned error %s at line %d while %s for tensor '%s'.\n", errorDesc.c_str(),    \
                             __LINE__, (callDesc), (pTensor)->name ? (pTensor)->name : "no-name");                   \
            return kTfLiteError;                                                                                     \
        }                                                                                                            \
    } while (0)

// Return true if type is kTfLiteFloat32.
extern bool IsFloat(TfLiteType type);

// Return true if type is kTfLiteUInt8 or kTfLiteInt8.
extern bool IsQuantized(TfLiteType type);

// Return true if the operator supports scalar data as input.
extern bool IsScalarInputSupported(int32_t builtinCode);

// Returns true if this delegate is configured to use a specific set of devices.
// If the acceleratorName in the delegate options is equal to "nnrt-reference"
// this method will return true only if the excludeNnrtReference is true.
extern bool IsUseTargetDevice(
    NnrtDelegate::Options delegateOptions, bool excludeNnrtReference = false);

// Fills the given result vector with the list of devices the given delegate
// is referring to.
// There are three possible results,
// - An empty array (not the full list of available accelerators,
//   for efficiency reasons) if no accelerator is chosen and the
//   disallowNnrtCpu delegate option is false.
// - A single element array with the target processor, if an accelerator name
//   is specified in the delegate options.
// - The target available device on device.
extern TfLiteStatus GetTargetDevice(TfLiteContext* context, TfLiteDelegate* delegate,
    const NnrtApi* nnrt, size_t& dev);

// Transpose demension following fixed axis.
// If exist -1  in destAxis, return kTfLiteError.
extern TfLiteStatus TransposeDims(TfLiteContext* context, const int32_t* dims, uint32_t dimCount,
    std::vector<int32_t> destAxis, std::vector<int32_t>& weightDims);

// Get Tensor size by byte.
// Calculate Tesnorsize by mul all dimension in dims.
// Return kTfLiteError if element dimension is less 0.
extern TfLiteStatus GetTensorSize(TfLiteContext* context, const int32_t* dims, int32_t dimCount, int64_t& tensorSize);

// Transpose dimension for Tensor.
// Only change NHWC format tensor to CHWN format tensor, and
// the capacity of result vec must equal to input tensor size.
template <class T>
TfLiteStatus TransposeTensor(TfLiteContext* context, int32_t tensorIndex, const int32_t* dims,
    T* transposeTensor)
{
    TF_LITE_ENSURE_EQ(context, dims != nullptr, true);

    // NHWC -> CHWN
    TfLiteTensor* tensor = &(context->tensors[tensorIndex]);
    const T* tensorData = reinterpret_cast<T*>(tensor->data.data);
    const int32_t batch = dims[DEPTHWISE_WEIGHT_BATCH_DIMENSION];
    const int32_t height = dims[DEPTHWISE_WEIGHT_HEIGHT_DIMENSION];
    const int32_t width = dims[DEPTHWISE_WEIGHT_WIDTH_DIMENSION];
    const int32_t channel = dims[DEPTHWISE_WEIGHT_CHANNEL_DIMENSION];

    for (int32_t c = 0; c < channel; ++c) {
        for (int32_t j = 0; j < height * width; ++j) {
            for (int32_t n = 0; n < batch; ++n) {
                int32_t newPos = c * (height * width) * batch + j * batch + n;
                int32_t orgPos = n * (height * width) * channel + j * channel + c;
                *(transposeTensor + newPos) = *(tensorData + orgPos);
            }
        }
    }

    return kTfLiteOk;
};

namespace delegate {
namespace nnrt {
using unorderedTypeMap = std::unordered_map<int32_t, int32_t>;

extern const std::vector<int32_t> ACTIVATE_FUSE_TYPE_LIST;

extern const unorderedTypeMap TFLITE_TYPE_TO_NNRT_TYPE;

const int32_t INVALID_INDEX = -1;

const int32_t OH_NN_UNSUPPORT_OPS = -1;

const int32_t OH_NN_FUSE_UNSUPPORTED = -1;
} // namespace nnrt
} // namespace delegate
} // namespace tflite

#endif // TENSORFLOW_LITE_DELEGATES_NNRT_UTILS_H
