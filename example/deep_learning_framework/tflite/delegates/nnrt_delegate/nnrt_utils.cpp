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

#include "nnrt_utils.h"

#include <iostream>
#include "tensorflow/lite/util.h"
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#include "neural_network_runtime_type.h"

namespace tflite {
std::string NnrtErrorDescription(int32_t errorCode)
{
    switch (errorCode) {
        case OH_NN_SUCCESS:
            return "OH_NN_SUCCESS";
        case OH_NN_FAILED:
            return "OH_NN_FAILED";
        case OH_NN_INVALID_PARAMETER:
            return "OH_NN_INVALID_PARAMETER";
        case OH_NN_MEMORY_ERROR:
            return "OH_NN_MEMORY_ERROR";
        case OH_NN_OPERATION_FORBIDDEN:
            return "OH_NN_OPERATION_FORBIDDEN";
        case OH_NN_NULL_PTR:
            return "OH_NN_NULL_PTR";
        case OH_NN_INVALID_FILE:
            return "OH_NN_INVALID_FILE";
        case OH_NN_UNAVALIDABLE_DEVICE:
            return "OH_NN_UNAVALIDABLE_DEVICE";
        case OH_NN_INVALID_PATH:
            return "OH_NN_INVALID_PATH";
        default:
            return "Unknown NNRT error code: " + std::to_string(errorCode);
    }
}

bool IsFloat(TfLiteType type)
{
    return type == kTfLiteFloat32;
}

bool IsQuantized(TfLiteType type)
{
    return ((type == kTfLiteUInt8) || (type == kTfLiteInt8));
}

bool IsScalarInputSupported(int32_t builtinCode)
{
    switch (builtinCode) {
        case kTfLiteBuiltinAdd:
        case kTfLiteBuiltinMul:
        case kTfLiteBuiltinSub:
        case kTfLiteBuiltinDiv:
        case kTfLiteBuiltinEqual:
        case kTfLiteBuiltinNotEqual:
        case kTfLiteBuiltinGreater:
        case kTfLiteBuiltinGreaterEqual:
        case kTfLiteBuiltinLess:
        case kTfLiteBuiltinLessEqual:
        case kTfLiteBuiltinPow:
        case kTfLiteBuiltinMaximum:
        case kTfLiteBuiltinMinimum:
        case kTfLiteBuiltinPrelu:
        case kTfLiteBuiltinLeakyRelu:
            return true;
        default:
            return false;
    }
}

bool IsUseTargetDevice(NnrtDelegate::Options delegateOptions, bool excludeNnrtReference)
{
    const std::string& deviceName = delegateOptions.acceleratorName;
    bool hasSelectedAccelerator = !deviceName.empty();
    if (!excludeNnrtReference && hasSelectedAccelerator) {
        if (!deviceName.compare(NNRT_REFERENCE_DEVICE)) {
            hasSelectedAccelerator = false;
        }
    }

    return hasSelectedAccelerator;
}

TfLiteStatus GetTargetDevice(TfLiteContext* context, TfLiteDelegate* delegate, const NnrtApi* nnrt, size_t& dev)
{
    TF_LITE_ENSURE_EQ(context, nnrt != nullptr, true);
    TF_LITE_ENSURE_EQ(context, delegate != nullptr, true);

    NnrtDelegate::Options delegateOptions;
    TF_LITE_ENSURE_STATUS(NnrtDelegate::GetOptions(delegate, delegateOptions));
    const std::string& deviceName = delegateOptions.acceleratorName;

    uint32_t numDevices {0};
    const size_t* alldevicesID {nullptr};
    RETURN_TFLITE_ERROR_IF_NN_ERROR(nnrt->OH_NNDevice_GetAllDevicesID(&alldevicesID, &numDevices),
        "Get available device number and deviceID.");
    if (numDevices == 0) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-UTILS] Have no available device.");
        return kTfLiteError;
    }

    std::vector<OH_NN_DeviceType> deviceTypes;
    for (uint32_t i = 0; i < numDevices; ++i) {
        OH_NN_DeviceType tempDeviceType {OH_NN_ACCELERATOR};
        RETURN_TFLITE_ERROR_IF_NN_ERROR(nnrt->OH_NNDevice_GetType(alldevicesID[i], &tempDeviceType),
            "Get available devicesType.");
        deviceTypes.emplace_back(tempDeviceType);
    }

    OH_NN_DeviceType deviceType {OH_NN_CPU};
    std::vector<OH_NN_DeviceType>::iterator pos = std::find(deviceTypes.begin(), deviceTypes.end(), deviceType);
    if (pos != deviceTypes.end()) {
        int index = distance(deviceTypes.begin(), pos);
        dev = alldevicesID[index];
    } else {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
            "[NNRT-UTILS] Cannot find the %s device, please choose another process unit.",
            deviceName.c_str());
        return kTfLiteError;
    }

    return kTfLiteOk;
}

TfLiteStatus TransposeDims(TfLiteContext* context, const int32_t* dims, uint32_t dimCount,
    std::vector<int32_t> destAxis, std::vector<int32_t>& weightDims)
{
    TF_LITE_ENSURE_EQ(context, dims != nullptr, true);

    if (dimCount != destAxis.size()) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-UTILS] Invalid dimension count %d.", dimCount);
        return kTfLiteError;
    }

    for (auto axis : destAxis) {
        if (axis < 0) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-UTILS] Invalid axis %d.", axis);
            return kTfLiteError;
        }
        weightDims.emplace_back(*(dims + axis));
    }

    return kTfLiteOk;
}

TfLiteStatus GetTensorSize(TfLiteContext* context, const int32_t* dims, int32_t dimCount, int64_t& tensorSize)
{
    TF_LITE_ENSURE_EQ(context, dims != nullptr, true);

    if (dimCount != DEPTHWISE_WEIGHT_DIMENSION_COUNT) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR,
            "[NNRT-UTILS] Dimension count is not equal to destination axis number, should be 4.");
        return kTfLiteError;
    }

    tensorSize = 1;
    for (int32_t i = 0; i < dimCount; ++i) {
        if (*(dims + i) <= 0) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-UTILS] Get invalid dimenision.");
            return kTfLiteError;
        }
        tensorSize *= *(dims + i);
    }

    return kTfLiteOk;
}

namespace delegate {
namespace nnrt {
const std::vector<int32_t> ACTIVATE_FUSE_TYPE_LIST = {
    OH_NN_FUSED_NONE,
    OH_NN_FUSED_RELU,
    OH_NN_FUSE_UNSUPPORTED,
    OH_NN_FUSED_RELU6,
    OH_NN_FUSE_UNSUPPORTED,
    OH_NN_FUSE_UNSUPPORTED,
    OH_NN_FUSE_UNSUPPORTED
};

const unorderedTypeMap TFLITE_TYPE_TO_NNRT_TYPE = {
    {kTfLiteBuiltinAdd,                          OH_NN_OPS_ADD},
    {kTfLiteBuiltinAveragePool2d,                OH_NN_OPS_AVG_POOL},
    {kTfLiteBuiltinConcatenation,                OH_NN_OPS_CONCAT},
    {kTfLiteBuiltinConv2d,                       OH_NN_OPS_CONV2D},
    {kTfLiteBuiltinDepthwiseConv2d,              OH_NN_OPS_DEPTHWISE_CONV2D_NATIVE},
    {kTfLiteBuiltinDepthToSpace,                 OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinDequantize,                   OH_NN_OPS_QUANT_DTYPE_CAST},
    {kTfLiteBuiltinEmbeddingLookup,              OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinFloor,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinFullyConnected,               OH_NN_OPS_FULL_CONNECTION},
    {kTfLiteBuiltinHashtableLookup,              OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinL2Normalization,              OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinL2Pool2d,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLocalResponseNormalization,   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLogistic,                     OH_NN_OPS_SIGMOID},
    {kTfLiteBuiltinLshProjection,                OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLstm,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinMaxPool2d,                    OH_NN_OPS_MAX_POOL},
    {kTfLiteBuiltinMul,                          OH_NN_OPS_MUL},
    {kTfLiteBuiltinRelu,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReluN1To1,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinRelu6,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReshape,                      OH_NN_OPS_RESHAPE},
    {kTfLiteBuiltinResizeBilinear,               OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinRnn,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSoftmax,                      OH_NN_OPS_SOFTMAX},
    {kTfLiteBuiltinSpaceToDepth,                 OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSvdf,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinTanh,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinConcatEmbeddings,             OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSkipGram,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinCall,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinCustom,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinEmbeddingLookupSparse,        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinPad,                          OH_NN_OPS_PAD},
    {kTfLiteBuiltinUnidirectionalSequenceRnn,    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinGather,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinBatchToSpaceNd,               OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSpaceToBatchNd,               OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinTranspose,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinMean,                         OH_NN_OPS_REDUCE_MEAN},
    {kTfLiteBuiltinSub,                          OH_NN_OPS_SUB},
    {kTfLiteBuiltinDiv,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSqueeze,                      OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinUnidirectionalSequenceLstm,   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinStridedSlice,                 OH_NN_OPS_STRIDED_SLICE},
    {kTfLiteBuiltinBidirectionalSequenceRnn,     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinExp,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinTopkV2,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSplit,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLogSoftmax,                   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinDelegate,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinBidirectionalSequenceLstm,    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinCast,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinPrelu,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinMaximum,                      OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinArgMax,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinMinimum,                      OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLess,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinNeg,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinPadv2,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinGreater,                      OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinGreaterEqual,                 OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLessEqual,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSelect,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSlice,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSin,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinTransposeConv,                OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSparseToDense,                OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinTile,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinExpandDims,                   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinEqual,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinNotEqual,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLog,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSum,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSqrt,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinRsqrt,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinShape,                        OH_NN_OPS_SHAPE},
    {kTfLiteBuiltinPow,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinArgMin,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinFakeQuant,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReduceProd,                   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReduceMax,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinPack,                         OH_NN_OPS_STACK},
    {kTfLiteBuiltinLogicalOr,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinOneHot,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLogicalAnd,                   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLogicalNot,                   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinUnpack,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReduceMin,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinFloorDiv,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReduceAny,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSquare,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinZerosLike,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinFill,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinFloorMod,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinRange,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinResizeNearestNeighbor,        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinLeakyRelu,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSquaredDifference,            OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinMirrorPad,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinAbs,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSplitV,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinUnique,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinCeil,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReverseV2,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinAddN,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinGatherNd,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinCos,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinWhere,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinRank,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinElu,                          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReverseSequence,              OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinMatrixDiag,                   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinQuantize,                     OH_NN_OPS_QUANT_DTYPE_CAST},
    {kTfLiteBuiltinMatrixSetDiag,                OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinRound,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinHardSwish,                    OH_NN_OPS_HSWISH},
    {kTfLiteBuiltinIf,                           OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinWhile,                        OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinNonMaxSuppressionV4,          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinNonMaxSuppressionV5,          OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinScatterNd,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSelectV2,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinDensify,                      OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinSegmentSum,                   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinBatchMatmul,                  OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinPlaceholderForGreaterOpCodes, OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinCumsum,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinCallOnce,                     OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinBroadcastTo,                  OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinRfft2d,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinConv3d,                       OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinImag,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReal,                         OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinComplexAbs,                   OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinHashtable,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinHashtableFind,                OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinHashtableImport,              OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinHashtableSize,                OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReduceAll,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinConv3dTranspose,              OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinVarHandle,                    OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinReadVariable,                 OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinAssignVariable,               OH_NN_UNSUPPORT_OPS},
    {kTfLiteBuiltinBroadcastTo,                  OH_NN_UNSUPPORT_OPS},
};
} // nnrt
} // namespace
} // namespace tflite