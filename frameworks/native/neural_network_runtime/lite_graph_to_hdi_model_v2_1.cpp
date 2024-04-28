/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#include "lite_graph_to_hdi_model_v2_1.h"
#include <vector>
#include <algorithm>
#include <sys/mman.h>
#include "common/log.h"
#include "message_parcel.h"
#include "nnrt/v2_1/nnrt_types.h"
#include "nnrt/v2_1/node_attr_types.h"
#include "securec.h"

using namespace OHOS::HDI::Nnrt::V2_1;
typedef void *PrimitivePtr;
typedef void *TensorPtr;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace NNRt_V2_1 {
std::vector<int8_t> ConvertActivation(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertActivation v2_1 failed, primitive is nullptr.");
        return {};
    }

    Activation activation{};
    activation.activationType = static_cast<HDI::Nnrt::V2_1::ActivationType>(
        mindspore::lite::MindIR_Activation_GetActivationType(primitive));
    activation.alpha = mindspore::lite::MindIR_Activation_GetAlpha(primitive);
    activation.minVal = mindspore::lite::MindIR_Activation_GetMinVal(primitive);
    activation.maxVal = mindspore::lite::MindIR_Activation_GetMaxVal(primitive);
    activation.approximate = mindspore::lite::MindIR_Activation_GetApproximate(primitive);

    OHOS::MessageParcel data;
    (void)ActivationBlockMarshalling(data, activation);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertAddFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertAddFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    AddFusion addFusion{};
    addFusion.activationType = static_cast<HDI::Nnrt::V2_1::ActivationType>(
        mindspore::lite::MindIR_Activation_GetActivationType(primitive));

    OHOS::MessageParcel data;
    (void)AddFusionBlockMarshalling(data, addFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertAll(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertAll v2_1 failed, primitive is nullptr.");
        return {};
    }

    All all{};
    all.keepDims = mindspore::lite::MindIR_All_GetKeepDims(primitive);

    OHOS::MessageParcel data;
    (void)AllBlockMarshalling(data, all);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertArgMaxFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertArgMaxFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    ArgMaxFusion argMaxFusion{};
    argMaxFusion.axis = mindspore::lite::MindIR_ArgMaxFusion_GetAxis(primitive);
    argMaxFusion.topK = mindspore::lite::MindIR_ArgMaxFusion_GetTopK(primitive);
    argMaxFusion.keepDims = mindspore::lite::MindIR_ArgMaxFusion_GetKeepDims(primitive);
    argMaxFusion.outMaxValue = mindspore::lite::MindIR_ArgMaxFusion_GetOutMaxValue(primitive);

    OHOS::MessageParcel data;
    (void)ArgMaxFusionBlockMarshalling(data, argMaxFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertAssert(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertAssert v2_1 failed, primitive is nullptr.");
        return {};
    }

    Assert assert{};
    assert.summarize = mindspore::lite::MindIR_Assert_GetSummarize(primitive);

    OHOS::MessageParcel data;
    (void)AssertBlockMarshalling(data, assert);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertAvgPoolFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertAvgPoolFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    AvgPoolFusion avgPoolFusion{};
    avgPoolFusion.kernelSize = mindspore::lite::MindIR_AvgPoolFusion_GetKernelSize(primitive);
    avgPoolFusion.strides = mindspore::lite::MindIR_AvgPoolFusion_GetStrides(primitive);
    avgPoolFusion.pad = mindspore::lite::MindIR_AvgPoolFusion_GetPad(primitive);
    avgPoolFusion.padMode = static_cast<PadMode>(mindspore::lite::MindIR_AvgPoolFusion_GetPadMode(primitive));
    avgPoolFusion.roundMode = static_cast<RoundMode>(mindspore::lite::MindIR_AvgPoolFusion_GetRoundMode(primitive));
    avgPoolFusion.format = static_cast<Format>(mindspore::lite::MindIR_AvgPoolFusion_GetFormat(primitive));
    avgPoolFusion.global = mindspore::lite::MindIR_AvgPoolFusion_GetGlobal(primitive);
    avgPoolFusion.activationType =
        static_cast<ActivationType>(mindspore::lite::MindIR_AvgPoolFusion_GetActivationType(primitive));

    OHOS::MessageParcel data;
    (void)AvgPoolFusionBlockMarshalling(data, avgPoolFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertBatchToSpaceND(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertBatchToSpaceND v2_1 failed, primitive is nullptr.");
        return {};
    }

    BatchToSpaceND batchToSpaceND{};
    batchToSpaceND.blockShape = mindspore::lite::MindIR_BatchToSpaceND_GetBlockShape(primitive);
    batchToSpaceND.crops = mindspore::lite::MindIR_BatchToSpaceND_GetCrops(primitive);

    OHOS::MessageParcel data;
    (void)BatchToSpaceNDBlockMarshalling(data, batchToSpaceND);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertBiasAdd(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertBiasAdd v2_1 failed, primitive is nullptr.");
        return {};
    }

    BiasAdd biasAdd{};
    OHOS::MessageParcel data;
    (void)BiasAddBlockMarshalling(data, biasAdd);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertBroadcastTo(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertBroadcastTo v2_1 failed, primitive is nullptr.");
        return {};
    }

    BroadcastTo broadcastTo{};
    broadcastTo.shape = mindspore::lite::MindIR_BroadcastTo_GetShape(primitive);

    OHOS::MessageParcel data;
    (void)BroadcastToBlockMarshalling(data, broadcastTo);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertCast(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertCast v2_1 failed, primitive is nullptr.");
        return {};
    }

    Cast cast{};
    OHOS::MessageParcel data;
    (void)CastBlockMarshalling(data, cast);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertCeil(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertCeil v2_1 failed, primitive is nullptr.");
        return {};
    }

    Ceil ceil{};
    OHOS::MessageParcel data;
    (void)CeilBlockMarshalling(data, ceil);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertClip(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertClip v2_1 failed, primitive is nullptr.");
        return {};
    }

    Clip clip{};
    clip.max = mindspore::lite::MindIR_Clip_GetMax(primitive);
    clip.min = mindspore::lite::MindIR_Clip_GetMin(primitive);

    OHOS::MessageParcel data;
    (void)ClipBlockMarshalling(data, clip);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertConcat(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertConcat v2_1 failed, primitive is nullptr.");
        return {};
    }

    Concat concat{};
    concat.axis = mindspore::lite::MindIR_Concat_GetAxis(primitive);
    OHOS::MessageParcel data;
    (void)ConcatBlockMarshalling(data, concat);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertConv2DFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertConv2DFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    Conv2DFusion conv2DFusion{};
    conv2DFusion.kernelSize = mindspore::lite::MindIR_Conv2DFusion_GetKernelSize(primitive);
    conv2DFusion.stride = mindspore::lite::MindIR_Conv2DFusion_GetStride(primitive);
    conv2DFusion.dilation = mindspore::lite::MindIR_Conv2DFusion_GetDilation(primitive);
    conv2DFusion.padMode = static_cast<PadMode>(mindspore::lite::MindIR_Conv2DFusion_GetPadMode(primitive));
    conv2DFusion.padList = mindspore::lite::MindIR_Conv2DFusion_GetPadList(primitive);
    conv2DFusion.group = mindspore::lite::MindIR_Conv2DFusion_GetGroup(primitive);
    conv2DFusion.inChannel = mindspore::lite::MindIR_Conv2DFusion_GetInChannel(primitive);
    conv2DFusion.outChannel = mindspore::lite::MindIR_Conv2DFusion_GetOutChannel(primitive);
    conv2DFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_Conv2DFusion_GetActivationType(primitive));

    OHOS::MessageParcel data;
    (void)Conv2DFusionBlockMarshalling(data, conv2DFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertConv2dTransposeFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertConv2dTransposeFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    Conv2dTransposeFusion conv2dTransposeFusion{};
    conv2dTransposeFusion.kernelSize = mindspore::lite::MindIR_Conv2dTransposeFusion_GetKernelSize(primitive);
    conv2dTransposeFusion.stride = mindspore::lite::MindIR_Conv2dTransposeFusion_GetStride(primitive);
    conv2dTransposeFusion.dilation = mindspore::lite::MindIR_Conv2dTransposeFusion_GetDilation(primitive);
    conv2dTransposeFusion.padMode = static_cast<PadMode>(
        mindspore::lite::MindIR_Conv2dTransposeFusion_GetPadMode(primitive));
    conv2dTransposeFusion.padList = mindspore::lite::MindIR_Conv2dTransposeFusion_GetPadList(primitive);
    conv2dTransposeFusion.group = mindspore::lite::MindIR_Conv2dTransposeFusion_GetGroup(primitive);
    conv2dTransposeFusion.inChannel = mindspore::lite::MindIR_Conv2dTransposeFusion_GetInChannel(primitive);
    conv2dTransposeFusion.outChannel = mindspore::lite::MindIR_Conv2dTransposeFusion_GetOutChannel(primitive);
    conv2dTransposeFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_Conv2dTransposeFusion_GetActivationType(primitive));
    conv2dTransposeFusion.outputPaddings = mindspore::lite::MindIR_Conv2dTransposeFusion_GetOutputPaddings(primitive);

    OHOS::MessageParcel data;
    (void)Conv2dTransposeFusionBlockMarshalling(data, conv2dTransposeFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertCos(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertCos v2_1 failed, primitive is nullptr.");
        return {};
    }

    Cos cos{};

    OHOS::MessageParcel data;
    (void)CosBlockMarshalling(data, cos);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertConstantOfShape(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertConstantOfShape v2_1 failed, primitive is nullptr.");
        return {};
    }

    ConstantOfShape constantOfShape{};
    constantOfShape.dataType = mindspore::lite::MindIR_ConstantOfShape_GetDataType(primitive);
    constantOfShape.value = mindspore::lite::MindIR_ConstantOfShape_GetValue(primitive);

    OHOS::MessageParcel data;
    (void)ConstantOfShapeBlockMarshalling(data, constantOfShape);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertCrop(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertCrop v2_1 failed, primitive is nullptr.");
        return {};
    }

    Crop crop{};
    crop.axis = mindspore::lite::MindIR_Crop_GetAxis(primitive);
    crop.offset = mindspore::lite::MindIR_Crop_GetOffsets(primitive);

    OHOS::MessageParcel data;
    (void)CropBlockMarshalling(data, crop);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertDepthToSpace(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertDepthToSpace v2_1 failed, primitive is nullptr.");
        return {};
    }

    DepthToSpace depthToSpace{};
    depthToSpace.blockSize = mindspore::lite::MindIR_DepthToSpace_GetBlockSize(primitive);
    depthToSpace.format = static_cast<Format>(
        mindspore::lite::MindIR_DepthToSpace_GetFormat(primitive));
    depthToSpace.mode = mindspore::lite::MindIR_DepthToSpace_GetMode(primitive);
    
    OHOS::MessageParcel data;
    (void)DepthToSpaceBlockMarshalling(data, depthToSpace);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertDetectionPostProcess(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertDetectionPostProcess v2_1 failed, primitive is nullptr.");
        return {};
    }

    DetectionPostProcess detectionPostProcess{};
    detectionPostProcess.format = static_cast<Format>(
        mindspore::lite::MindIR_DetectionPostProcess_GetFormat(primitive));
    detectionPostProcess.inputSize = mindspore::lite::MindIR_DetectionPostProcess_GetInputSize(primitive);
    detectionPostProcess.scale = mindspore::lite::MindIR_DetectionPostProcess_GetScale(primitive);
    detectionPostProcess.nmsIoUThreshold = mindspore::lite::MindIR_DetectionPostProcess_GetNmsIouThreshold(primitive);
    detectionPostProcess.nmsScoreThreshold =
        mindspore::lite::MindIR_DetectionPostProcess_GetNmsScoreThreshold(primitive);
    detectionPostProcess.maxDetections = mindspore::lite::MindIR_DetectionPostProcess_GetMaxDetections(primitive);
    detectionPostProcess.detectionsPerClass =
        mindspore::lite::MindIR_DetectionPostProcess_GetDetectionsPerClass(primitive);
    detectionPostProcess.maxClassesPerDetection =
        mindspore::lite::MindIR_DetectionPostProcess_GetMaxClassesPerDetection(primitive);
    detectionPostProcess.numClasses = mindspore::lite::MindIR_DetectionPostProcess_GetNumClasses(primitive);
    detectionPostProcess.useRegularNms = mindspore::lite::MindIR_DetectionPostProcess_GetUseRegularNms(primitive);
    detectionPostProcess.outQuantized = mindspore::lite::MindIR_DetectionPostProcess_GetOutQuantized(primitive);
    
    OHOS::MessageParcel data;
    (void)DetectionPostProcessBlockMarshalling(data, detectionPostProcess);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertDivFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertDivFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    DivFusion divFusion{};
    divFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_DivFusion_GetActivationType(primitive));
    OHOS::MessageParcel data;
    (void)DivFusionBlockMarshalling(data, divFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertEltwise(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertEltwise v2_1 failed, primitive is nullptr.");
        return {};
    }

    Eltwise eltwise{};
    eltwise.mode = static_cast<EltwiseMode>(mindspore::lite::MindIR_Eltwise_GetMode(primitive));
    OHOS::MessageParcel data;
    (void)EltwiseBlockMarshalling(data, eltwise);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertEqual(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertEqual v2_1 failed, primitive is nullptr.");
        return {};
    }

    Equal equal{};
    OHOS::MessageParcel data;
    (void)EqualBlockMarshalling(data, equal);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertExpFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertExp v2_1 failed, primitive is nullptr.");
        return {};
    }

    ExpFusion exp{};
    exp.base = mindspore::lite::MindIR_ExpFusion_GetBase(primitive);
    exp.scale = mindspore::lite::MindIR_ExpFusion_GetScale(primitive);
    exp.shift = mindspore::lite::MindIR_ExpFusion_GetShift(primitive);

    OHOS::MessageParcel data;
    (void)ExpFusionBlockMarshalling(data, exp);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertExpandDims(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertExpandDims v2_1 failed, primitive is nullptr.");
        return {};
    }

    ExpandDims expandDims{};
    OHOS::MessageParcel data;
    (void)ExpandDimsBlockMarshalling(data, expandDims);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertFlatten(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertFlatten v2_1 failed, primitive is nullptr.");
        return {};
    }

    Flatten faltten{};
    faltten.axis = mindspore::lite::MindIR_Flatten_GetAxis(primitive);

    OHOS::MessageParcel data;
    (void)FlattenBlockMarshalling(data, faltten);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertFloor(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertFloor v2_1 failed, primitive is nullptr.");
        return {};
    }

    Floor floor{};

    OHOS::MessageParcel data;
    (void)FloorBlockMarshalling(data, floor);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertFill(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertFill v2_1 failed, primitive is nullptr.");
        return {};
    }

    Fill fill{};
    OHOS::MessageParcel data;
    (void)FillBlockMarshalling(data, fill);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertFullConnection(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertFullConnection v2_1 failed, primitive is nullptr.");
        return {};
    }

    FullConnection fullConnection{};
    fullConnection.hasBias = mindspore::lite::MindIR_FullConnection_GetHasBias(primitive);
    fullConnection.useAxis = mindspore::lite::MindIR_FullConnection_GetUseAxis(primitive);
    fullConnection.axis = mindspore::lite::MindIR_FullConnection_GetAxis(primitive);
    fullConnection.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_FullConnection_GetActivationType(primitive));

    OHOS::MessageParcel data;
    (void)FullConnectionBlockMarshalling(data, fullConnection);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertFusedBatchNorm(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertFusedBatchNorm v2_1 failed, primitive is nullptr.");
        return {};
    }

    FusedBatchNorm fusedBatchNorm{};
    fusedBatchNorm.epsilon = mindspore::lite::MindIR_FusedBatchNorm_GetEpsilon(primitive);
    OHOS::MessageParcel data;
    (void)FusedBatchNormBlockMarshalling(data, fusedBatchNorm);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertGather(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertGather v2_1 failed, primitive is nullptr.");
        return {};
    }

    Gather gather{};
    OHOS::MessageParcel data;
    (void)GatherBlockMarshalling(data, gather);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertGatherNd(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertGatherNd v2_1 failed, primitive is nullptr.");
        return {};
    }

    GatherNd gatherNd{};
    OHOS::MessageParcel data;
    (void)GatherNdBlockMarshalling(data, gatherNd);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertGreater(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertGreater v2_1 failed, primitive is nullptr.");
        return {};
    }

    Greater greater{};
    OHOS::MessageParcel data;
    (void)GreaterBlockMarshalling(data, greater);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertGreaterEqual(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertGreaterEqual v2_1 failed, primitive is nullptr.");
        return {};
    }

    GreaterEqual greaterEqual{};
    OHOS::MessageParcel data;
    (void)GreaterEqualBlockMarshalling(data, greaterEqual);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertInstanceNorm(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertInstanceNorm v2_1 failed, primitive is nullptr.");
        return {};
    }

    InstanceNorm instanceNorm{};
    instanceNorm.epsilon = mindspore::lite::MindIR_InstanceNorm_GetEpsilon(primitive);

    OHOS::MessageParcel data;
    (void)InstanceNormBlockMarshalling(data, instanceNorm);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLayerNormFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLayerNorm v2_1 failed, primitive is nullptr.");
        return {};
    }

    LayerNormFusion layerNormFusion{};
    layerNormFusion.beginNormAxis = mindspore::lite::MindIR_LayerNormFusion_GetBeginNormAxis(primitive);
    layerNormFusion.epsilon = mindspore::lite::MindIR_LayerNormFusion_GetEpsilon(primitive);
    layerNormFusion.elementwiseAffine = mindspore::lite::MindIR_LayerNormFusion_GetElementwiseAffine(primitive);
    layerNormFusion.beginParamsAxis = mindspore::lite::MindIR_LayerNormFusion_GetBeginParamsAxis(primitive);

    OHOS::MessageParcel data;
    (void)LayerNormFusionBlockMarshalling(data, layerNormFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLess(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLess v2_1 failed, primitive is nullptr.");
        return {};
    }

    Less less{};
    OHOS::MessageParcel data;
    (void)LessBlockMarshalling(data, less);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLessEqual(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLessEqual v2_1 failed, primitive is nullptr.");
        return {};
    }

    LessEqual lessEqual{};
    OHOS::MessageParcel data;
    (void)LessEqualBlockMarshalling(data, lessEqual);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLog(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLog v2_1 failed, primitive is nullptr.");
        return {};
    }

    Log log{};

    OHOS::MessageParcel data;
    (void)LogBlockMarshalling(data, log);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLogicalAnd(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLogicalAnd v2_1 failed, primitive is nullptr.");
        return {};
    }

    LogicalAnd logicalAnd{};

    OHOS::MessageParcel data;
    (void)LogicalAndBlockMarshalling(data, logicalAnd);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLogicalNot(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLogicalNot v2_1 failed, primitive is nullptr.");
        return {};
    }

    LogicalNot logicalNot{};

    OHOS::MessageParcel data;
    (void)LogicalNotBlockMarshalling(data, logicalNot);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLogicalOr(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLogicalOr v2_1 failed, primitive is nullptr.");
        return {};
    }

    LogicalOr logicalOr{};

    OHOS::MessageParcel data;
    (void)LogicalOrBlockMarshalling(data, logicalOr);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLRN(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLRN v2_1 failed, primitive is nullptr.");
        return {};
    }

    LRN lrn{};
    lrn.depthRadius = mindspore::lite::MindIR_LRN_GetDepthRadius(primitive);
    lrn.bias = mindspore::lite::MindIR_LRN_GetBias(primitive);
    lrn.alpha = mindspore::lite::MindIR_LRN_GetAlpha(primitive);
    lrn.beta = mindspore::lite::MindIR_LRN_GetBeta(primitive);
    lrn.normRegion = mindspore::lite::MindIR_LRN_GetNormRegion(primitive);

    OHOS::MessageParcel data;
    (void)LRNBlockMarshalling(data, lrn);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLSTM(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLSTM v2_1 failed, primitive is nullptr.");
        return {};
    }

    LSTM lstm{};
    lstm.bidirectional = mindspore::lite::MindIR_LSTM_GetBidirectional(primitive);
    lstm.hasBias = mindspore::lite::MindIR_LSTM_GetHasBias(primitive);
    lstm.inputSize = mindspore::lite::MindIR_LSTM_GetInputSize(primitive);
    lstm.hiddenSize = mindspore::lite::MindIR_LSTM_GetHiddenSize(primitive);
    lstm.numLayers = mindspore::lite::MindIR_LSTM_GetNumLayers(primitive);
    lstm.numDirections = mindspore::lite::MindIR_LSTM_GetNumDirections(primitive);
    lstm.dropout = mindspore::lite::MindIR_LSTM_GetDropout(primitive);
    lstm.zoneoutCell = mindspore::lite::MindIR_LSTM_GetZoneoutCell(primitive);
    lstm.zoneoutHidden = mindspore::lite::MindIR_LSTM_GetZoneoutHidden(primitive);
    lstm.projSize = mindspore::lite::MindIR_LSTM_GetProjSize(primitive);

    OHOS::MessageParcel data;
    (void)LSTMBlockMarshalling(data, lstm);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertL2NormalizeFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertL2NormalizeFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    L2NormalizeFusion l2NormalizeFusion{};
    l2NormalizeFusion.axis = mindspore::lite::MindIR_L2NormalizeFusion_GetAxis(primitive);
    l2NormalizeFusion.epsilon = mindspore::lite::MindIR_L2NormalizeFusion_GetEpsilon(primitive);
    l2NormalizeFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_L2NormalizeFusion_GetActivationType(primitive));

    OHOS::MessageParcel data;
    (void)L2NormalizeFusionBlockMarshalling(data, l2NormalizeFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertMatMulFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMatMulFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    MatMulFusion matMulFusion{};
    matMulFusion.transposeA = mindspore::lite::MindIR_MatMulFusion_GetTransposeA(primitive);
    matMulFusion.transposeB = mindspore::lite::MindIR_MatMulFusion_GetTransposeB(primitive);
    matMulFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_MatMulFusion_GetActivationType(primitive));

    OHOS::MessageParcel data;
    (void)MatMulFusionBlockMarshalling(data, matMulFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertMaximum(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMaximum v2_1 failed, primitive is nullptr.");
        return {};
    }

    Maximum maximum{};
    OHOS::MessageParcel data;
    (void)MaximumBlockMarshalling(data, maximum);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertMaxPoolFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMaxPoolFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    MaxPoolFusion maxPoolFusion{};
    maxPoolFusion.kernelSize = mindspore::lite::MindIR_MaxPoolFusion_GetKernelSize(primitive);
    maxPoolFusion.strides = mindspore::lite::MindIR_MaxPoolFusion_GetStrides(primitive);
    maxPoolFusion.pad = mindspore::lite::MindIR_MaxPoolFusion_GetPad(primitive);
    maxPoolFusion.padMode = static_cast<PadMode>(mindspore::lite::MindIR_MaxPoolFusion_GetPadMode(primitive));
    maxPoolFusion.format = static_cast<Format>(mindspore::lite::MindIR_MaxPoolFusion_GetFormat(primitive));
    maxPoolFusion.roundMode = static_cast<RoundMode>(mindspore::lite::MindIR_MaxPoolFusion_GetRoundMode(primitive));
    maxPoolFusion.global = mindspore::lite::MindIR_MaxPoolFusion_GetGlobal(primitive);
    maxPoolFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_MaxPoolFusion_GetActivationType(primitive));

    OHOS::MessageParcel data;
    (void)MaxPoolFusionBlockMarshalling(data, maxPoolFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertMinimum(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMinimum v2_1 failed, primitive is nullptr.");
        return {};
    }

    Minimum minimum{};

    OHOS::MessageParcel data;
    (void)MinimumBlockMarshalling(data, minimum);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertMod(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMod v2_1 failed, primitive is nullptr.");
        return {};
    }

    Mod mod{};

    OHOS::MessageParcel data;
    (void)ModBlockMarshalling(data, mod);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertMulFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMulFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    MulFusion mulFusion{};
    mulFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_MulFusion_GetActivationType(primitive));
    OHOS::MessageParcel data;
    (void)MulFusionBlockMarshalling(data, mulFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertNeg(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertNeg v2_1 failed, primitive is nullptr.");
        return {};
    }

    Neg neg{};

    OHOS::MessageParcel data;
    (void)NegBlockMarshalling(data, neg);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertNotEqual(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertNotEqual v2_1 failed, primitive is nullptr.");
        return {};
    }

    NotEqual notEqual{};

    OHOS::MessageParcel data;
    (void)NotEqualBlockMarshalling(data, notEqual);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertOneHot(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertOneHot v2_1 failed, primitive is nullptr.");
        return {};
    }

    OneHot oneHot{};
    oneHot.axis = mindspore::lite::MindIR_OneHot_GetAxis(primitive);
    OHOS::MessageParcel data;
    (void)OneHotBlockMarshalling(data, oneHot);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertPadFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertPadFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    PadFusion padFusion{};
    padFusion.paddings = mindspore::lite::MindIR_PadFusion_GetPaddings(primitive);
    padFusion.paddingMode = static_cast<PaddingMode>(mindspore::lite::MindIR_PadFusion_GetPaddingMode(primitive));
    padFusion.constantValue = mindspore::lite::MindIR_PadFusion_GetConstantValue(primitive);
    OHOS::MessageParcel data;
    (void)PadFusionBlockMarshalling(data, padFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertPowFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertPowFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    PowFusion powFusion{};
    powFusion.scale = mindspore::lite::MindIR_PowFusion_GetScale(primitive);
    powFusion.shift = mindspore::lite::MindIR_PowFusion_GetShift(primitive);
    OHOS::MessageParcel data;
    (void)PowFusionBlockMarshalling(data, powFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertPReLUFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertPReLUFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    PReLUFusion pReLUFusion{};
    pReLUFusion.channelShared = mindspore::lite::MindIR_PReLUFusion_GetChannelShared(primitive);
    OHOS::MessageParcel data;
    (void)PReLUFusionBlockMarshalling(data, pReLUFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertQuantDTypeCast(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertQuantDTypeCast v2_1 failed, primitive is nullptr.");
        return {};
    }

    QuantDTypeCastV2 quantDTypeCast{};
    quantDTypeCast.srcT = mindspore::lite::MindIR_QuantDTypeCast_GetSrcT(primitive);
    quantDTypeCast.dstT = mindspore::lite::MindIR_QuantDTypeCast_GetDstT(primitive);
    quantDTypeCast.axis = mindspore::lite::MindIR_QuantDTypeCast_GetAxis(primitive);

    OHOS::MessageParcel data;
    (void)QuantDTypeCastV2BlockMarshalling(data, quantDTypeCast);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertRank(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertRank v2_1 failed, primitive is nullptr.");
        return {};
    }

    Rank rank{};

    OHOS::MessageParcel data;
    (void)RankBlockMarshalling(data, rank);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertRange(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertRange v2_1 failed, primitive is nullptr.");
        return {};
    }

    Range range{};
    range.dataType = mindspore::lite::MindIR_Range_GetDType(primitive);
    range.start = mindspore::lite::MindIR_Range_GetStart(primitive);
    range.limit = mindspore::lite::MindIR_Range_GetLimit(primitive);
    range.delta = mindspore::lite::MindIR_Range_GetDelta(primitive);

    OHOS::MessageParcel data;
    (void)RangeBlockMarshalling(data, range);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertReciprocal(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertReciprocal v2_1 failed, primitive is nullptr.");
        return {};
    }

    Reciprocal reciprocal{};

    OHOS::MessageParcel data;
    (void)ReciprocalBlockMarshalling(data, reciprocal);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertReduceFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertReduceFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    ReduceFusion reduceFusion{};
    reduceFusion.keepDims = mindspore::lite::MindIR_ReduceFusion_GetKeepDims(primitive);
    reduceFusion.mode = static_cast<ReduceMode>(mindspore::lite::MindIR_ReduceFusion_GetMode(primitive));
    reduceFusion.reduceToEnd = mindspore::lite::MindIR_ReduceFusion_GetReduceToEnd(primitive);
    reduceFusion.coeff = mindspore::lite::MindIR_ReduceFusion_GetCoeff(primitive);
    OHOS::MessageParcel data;
    (void)ReduceFusionBlockMarshalling(data, reduceFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertReshape(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertReshape v2_1 failed, primitive is nullptr.");
        return {};
    }

    Reshape reshape{};
    OHOS::MessageParcel data;
    (void)ReshapeBlockMarshalling(data, reshape);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertResize(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertResize v2_1 failed, primitive is nullptr.");
        return {};
    }
 
    Resize resize{};
    resize.method = static_cast<ResizeMethod>(mindspore::lite::MindIR_Resize_GetMethod(primitive));
    resize.newHeight = mindspore::lite::MindIR_Resize_GetNewHeight(primitive);
    resize.newWidth = mindspore::lite::MindIR_Resize_GetNewWidth(primitive);
    resize.preserveAspectRatio = mindspore::lite::MindIR_Resize_GetPreserveAspectRatio(primitive);
    resize.coordinateTransformMode =
      static_cast<CoordinateTransformMode>(mindspore::lite::MindIR_Resize_GetCoordinateTransformMode(primitive));
    resize.cubicCoeff = mindspore::lite::MindIR_Resize_GetCubicCoeff(primitive);
    resize.excludeOutside = mindspore::lite::MindIR_Resize_GetExcludeOutside(primitive);
    resize.extrapolationValue = mindspore::lite::MindIR_Resize_GetExtrapolationValue(primitive);
    resize.nearestMode = static_cast<NearestMode>(mindspore::lite::MindIR_Resize_GetNearestMode(primitive));
    OHOS::MessageParcel data;
    (void)ResizeBlockMarshalling(data, resize);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertRound(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertRound v2_1 failed, primitive is nullptr.");
        return {};
    }
 
    Round round{};

    OHOS::MessageParcel data;
    (void)RoundBlockMarshalling(data, round);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertRsqrt(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertRsqrt v2_1 failed, primitive is nullptr.");
        return {};
    }

    Rsqrt rsqrt{};
    OHOS::MessageParcel data;
    (void)RsqrtBlockMarshalling(data, rsqrt);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertScaleFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertScaleFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    ScaleFusion scaleFusion{};
    scaleFusion.axis = mindspore::lite::MindIR_ScaleFusion_GetAxis(primitive);
    scaleFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_ScaleFusion_GetActivationType(primitive));
    OHOS::MessageParcel data;
    (void)ScaleFusionBlockMarshalling(data, scaleFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertScatterNd(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertScatterNd v2_1 failed, primitive is nullptr.");
        return {};
    }

    ScatterNd scatterNd{};

    OHOS::MessageParcel data;
    (void)ScatterNdBlockMarshalling(data, scatterNd);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertShape(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertShape v2_1 failed, primitive is nullptr.");
        return {};
    }

    Shape shape{};
    OHOS::MessageParcel data;
    (void)ShapeBlockMarshalling(data, shape);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSin(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSin v2_1 failed, primitive is nullptr.");
        return {};
    }

    Sin sin{};

    OHOS::MessageParcel data;
    (void)SinBlockMarshalling(data, sin);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSliceFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSliceFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    SliceFusion sliceFusion{};
    sliceFusion.axes = mindspore::lite::MindIR_SliceFusion_GetAxes(primitive);
    OHOS::MessageParcel data;
    (void)SliceFusionBlockMarshalling(data, sliceFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSoftmax(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSoftmax v2_1 failed, primitive is nullptr.");
        return {};
    }

    Softmax softmax{};
    softmax.axis = mindspore::lite::MindIR_Softmax_GetAxis(primitive);
    OHOS::MessageParcel data;
    (void)SoftmaxBlockMarshalling(data, softmax);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSpaceToBatchND(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSpaceToBatchND v2_1 failed, primitive is nullptr.");
        return {};
    }

    SpaceToBatchND spaceToBatchND{};
    spaceToBatchND.blockShape = mindspore::lite::MindIR_SpaceToBatchND_GetBlockShape(primitive);
    spaceToBatchND.paddings = mindspore::lite::MindIR_SpaceToBatchND_GetPaddings(primitive);
    OHOS::MessageParcel data;
    (void)SpaceToBatchNDBlockMarshalling(data, spaceToBatchND);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSpaceToDepth(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSpaceToDepth v2_1 failed, primitive is nullptr.");
        return {};
    }

    SpaceToDepth spaceToDepth{};
    spaceToDepth.format = static_cast<Format>(mindspore::lite::MindIR_SpaceToDepth_GetFormat(primitive));
    spaceToDepth.blockSize = mindspore::lite::MindIR_SpaceToDepth_GetBlockSize(primitive);

    OHOS::MessageParcel data;
    (void)SpaceToDepthBlockMarshalling(data, spaceToDepth);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSparseToDense(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSparseToDense v2_1 failed, primitive is nullptr.");
        return {};
    }

    SparseToDense sparseToDense{};
    OHOS::MessageParcel data;
    (void)SparseToDenseBlockMarshalling(data, sparseToDense);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSplit(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSplit v2_1 failed, primitive is nullptr.");
        return {};
    }

    Split split{};
    split.outputNum = mindspore::lite::MindIR_Split_GetOutputNum(primitive);
    split.sizeSplits = mindspore::lite::MindIR_Split_GetSizeSplits(primitive);
    split.axis = mindspore::lite::MindIR_Split_GetAxis(primitive);
    OHOS::MessageParcel data;
    (void)SplitBlockMarshalling(data, split);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSqrt(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSqrt v2_1 failed, primitive is nullptr.");
        return {};
    }

    Sqrt sqrt{};
    OHOS::MessageParcel data;
    (void)SqrtBlockMarshalling(data, sqrt);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSquare(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSquare v2_1 failed, primitive is nullptr.");
        return {};
    }

    Square square{};
    OHOS::MessageParcel data;
    (void)SquareBlockMarshalling(data, square);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSquaredDifference(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSquaredDifference v2_1 failed, primitive is nullptr.");
        return {};
    }

    SquaredDifference squaredDifference{};
    OHOS::MessageParcel data;
    (void)SquaredDifferenceBlockMarshalling(data, squaredDifference);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSqueeze(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSqueeze v2_1 failed, primitive is nullptr.");
        return {};
    }

    Squeeze squeeze{};
    squeeze.axis = mindspore::lite::MindIR_Squeeze_GetAxis(primitive);
    OHOS::MessageParcel data;
    (void)SqueezeBlockMarshalling(data, squeeze);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertStack(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertStack v2_1 failed, primitive is nullptr.");
        return {};
    }

    Stack stack{};
    stack.axis = mindspore::lite::MindIR_Stack_GetAxis(primitive);
    OHOS::MessageParcel data;
    (void)StackBlockMarshalling(data, stack);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertStridedSlice(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertStridedSlice v2_1 failed, primitive is nullptr.");
        return {};
    }

    StridedSlice stridedSlice{};
    stridedSlice.beginMask = mindspore::lite::MindIR_StridedSlice_GetBeginMask(primitive);
    stridedSlice.endMask = mindspore::lite::MindIR_StridedSlice_GetEndMask(primitive);
    stridedSlice.ellipsisMask = mindspore::lite::MindIR_StridedSlice_GetEllipsisMask(primitive);
    stridedSlice.newAxisMask = mindspore::lite::MindIR_StridedSlice_GetNewAxisMask(primitive);
    stridedSlice.shrinkAxisMask = mindspore::lite::MindIR_StridedSlice_GetShrinkAxisMask(primitive);
    OHOS::MessageParcel data;
    (void)StridedSliceBlockMarshalling(data, stridedSlice);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSubFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSubFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    SubFusion subFusion{};
    subFusion.activationType = static_cast<ActivationType>(
        mindspore::lite::MindIR_SubFusion_GetActivationType(primitive));
    OHOS::MessageParcel data;
    (void)SubFusionBlockMarshalling(data, subFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertTileFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertTileFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    TileFusion tileFusion{};
    tileFusion.dims = mindspore::lite::MindIR_TileFusion_GetDims(primitive);
    OHOS::MessageParcel data;
    (void)TileFusionBlockMarshalling(data, tileFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertTopKFusion(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertTopKFusion v2_1 failed, primitive is nullptr.");
        return {};
    }

    TopKFusion topKFusion{};
    topKFusion.sorted = mindspore::lite::MindIR_TopKFusion_GetSorted(primitive);
    topKFusion.axis = mindspore::lite::MindIR_TopKFusion_GetAxis(primitive);
    OHOS::MessageParcel data;
    (void)TopKFusionBlockMarshalling(data, topKFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertTranspose(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertTranspose v2_1 failed, primitive is nullptr.");
        return {};
    }

    Transpose transpose{};
    OHOS::MessageParcel data;
    (void)TransposeBlockMarshalling(data, transpose);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertUnsqueeze(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertUnsqueeze v2_1 failed, primitive is nullptr.");
        return {};
    }

    Unsqueeze unsqueeze{};
    unsqueeze.axis = mindspore::lite::MindIR_Unsqueeze_GetAxis(primitive);
    OHOS::MessageParcel data;
    (void)UnsqueezeBlockMarshalling(data, unsqueeze);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertUnstack(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertUnstack v2_1 failed, primitive is nullptr.");
        return {};
    }

    Unstack unstack{};
    unstack.axis = mindspore::lite::MindIR_Unstack_GetAxis(primitive);

    OHOS::MessageParcel data;
    (void)UnstackBlockMarshalling(data, unstack);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertWhere(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertWhere v2_1 failed, primitive is nullptr.");
        return {};
    }

    Where where{};

    OHOS::MessageParcel data;
    (void)WhereBlockMarshalling(data, where);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSelect(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSelect v2_1 failed, primitive is nullptr.");
        return {};
    }

    Select select{};

    OHOS::MessageParcel data;
    (void)SelectBlockMarshalling(data, select);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertErf(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertErf v2_1 failed, primitive is nullptr.");
        return {};
    }

    Erf erf{};

    OHOS::MessageParcel data;
    (void)ErfBlockMarshalling(data, erf);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLogSoftmax(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLogSoftmax v2_1 failed, primitive is nullptr.");
        return {};
    }

    LogSoftmax logSoftmax{};
    logSoftmax.axis = mindspore::lite::MindIR_LogSoftmax_GetAxis(primitive);

    OHOS::MessageParcel data;
    (void)LogSoftmaxBlockMarshalling(data, logSoftmax);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::unordered_map<NodeType, std::vector<int8_t>(*)(const PrimitivePtr)> convertOpMap = {
    {NODE_TYPE_ACTIVATION, &ConvertActivation},
    {NODE_TYPE_ADD_FUSION, &ConvertAddFusion},
    {NODE_TYPE_ALL, &ConvertAll},
    {NODE_TYPE_ARGMAX_FUSION, &ConvertArgMaxFusion},
    {NODE_TYPE_ASSERT, &ConvertAssert},
    {NODE_TYPE_AVGPOOL_FUSION, &ConvertAvgPoolFusion},
    {NODE_TYPE_BATCH_TO_SPACE_ND, &ConvertBatchToSpaceND},
    {NODE_TYPE_BIAS_ADD, &ConvertBiasAdd},
    {NODE_TYPE_BROADCAST_TO, &ConvertBroadcastTo},
    {NODE_TYPE_CAST, &ConvertCast},
    {NODE_TYPE_CEIL, &ConvertCeil},
    {NODE_TYPE_CLIP, &ConvertClip},
    {NODE_TYPE_CONCAT, &ConvertConcat},
    {NODE_TYPE_CONV2D_FUSION, &ConvertConv2DFusion},
    {NODE_TYPE_CONV2D_TRANSPOSE_FUSION, &ConvertConv2dTransposeFusion},
    {NODE_TYPE_COS, &ConvertCos},
    {NODE_TYPE_CONSTANT_OF_SHAPE, &ConvertConstantOfShape},
    {NODE_TYPE_CROP, &ConvertCrop},
    {NODE_TYPE_DEPTH_TO_SPACE, &ConvertDepthToSpace},
    {NODE_TYPE_DETECTION_POST_PROCESS, &ConvertDetectionPostProcess},
    {NODE_TYPE_DIV_FUSION, &ConvertDivFusion},
    {NODE_TYPE_ELTWISE, &ConvertEltwise},
    {NODE_TYPE_EQUAL, &ConvertEqual},
    {NODE_TYPE_EXPFUSION, &ConvertExpFusion},
    {NODE_TYPE_EXPAND_DIMS, &ConvertExpandDims},
    {NODE_TYPE_FLATTEN, &ConvertFlatten},
    {NODE_TYPE_FLOOR, &ConvertFloor},
    {NODE_TYPE_FILL, &ConvertFill},
    {NODE_TYPE_FULL_CONNECTION, &ConvertFullConnection},
    {NODE_TYPE_FUSED_BATCH_NORM, &ConvertFusedBatchNorm},
    {NODE_TYPE_GATHER, &ConvertGather},
    {NODE_TYPE_GATHER_ND, &ConvertGatherNd},
    {NODE_TYPE_GREATER, &ConvertGreater},
    {NODE_TYPE_GREATER_EQUAL, &ConvertGreaterEqual},
    {NODE_TYPE_INSTANCE_NORM, &ConvertInstanceNorm},
    {NODE_TYPE_LAYER_NORM_FUSION, &ConvertLayerNormFusion},
    {NODE_TYPE_LESS, &ConvertLess},
    {NODE_TYPE_LESS_EQUAL, &ConvertLessEqual},
    {NODE_TYPE_LOG, &ConvertLog},
    {NODE_TYPE_LOGICAL_AND, &ConvertLogicalAnd},
    {NODE_TYPE_LOGICAL_NOT, &ConvertLogicalNot},
    {NODE_TYPE_LOGICAL_OR, &ConvertLogicalOr},
    {NODE_TYPE_LRN, &ConvertLRN},
    {NODE_TYPE_LSTM, &ConvertLSTM},
    {NODE_TYPE_L2_NORMALIZE_FUSION, &ConvertL2NormalizeFusion},
    {NODE_TYPE_MATMUL_FUSION, &ConvertMatMulFusion},
    {NODE_TYPE_MAXIMUM, &ConvertMaximum},
    {NODE_TYPE_MAX_POOL_FUSION, &ConvertMaxPoolFusion},
    {NODE_TYPE_MINIMUM, &ConvertMinimum},
    {NODE_TYPE_MOD, &ConvertMod},
    {NODE_TYPE_MUL_FUSION, &ConvertMulFusion},
    {NODE_TYPE_NEG, &ConvertNeg},
    {NODE_TYPE_NOT_EQUAL, &ConvertNotEqual},
    {NODE_TYPE_ONE_HOT, &ConvertOneHot},
    {NODE_TYPE_PAD_FUSION, &ConvertPadFusion},
    {NODE_TYPE_POW_FUSION, &ConvertPowFusion},
    {NODE_TYPE_PRELU_FUSION, &ConvertPReLUFusion},
    {NODE_TYPE_QUANT_DTYPE_CAST, &ConvertQuantDTypeCast},
    {NODE_TYPE_RANK, &ConvertRank},
    {NODE_TYPE_RANGE, &ConvertRange},
    {NODE_TYPE_RECIPROCAL, &ConvertReciprocal},
    {NODE_TYPE_REDUCE_FUSION, &ConvertReduceFusion},
    {NODE_TYPE_RESHAPE, &ConvertReshape},
    {NODE_TYPE_RESIZE, &ConvertResize},
    {NODE_TYPE_ROUND, &ConvertRound},
    {NODE_TYPE_RSQRT, &ConvertRsqrt},
    {NODE_TYPE_SCALE_FUSION, &ConvertScaleFusion},
    {NODE_TYPE_SCATTER_ND, &ConvertScatterNd},
    {NODE_TYPE_SHAPE, &ConvertShape},
    {NODE_TYPE_SIN, &ConvertSin},
    {NODE_TYPE_SLICE_FUSION, &ConvertSliceFusion},
    {NODE_TYPE_SOFTMAX, &ConvertSoftmax},
    {NODE_TYPE_SPACE_TO_BATCH_ND, &ConvertSpaceToBatchND},
    {NODE_TYPE_SPACE_TO_DEPTH, &ConvertSpaceToDepth},
    {NODE_TYPE_SPARSE_TO_DENSE, &ConvertSparseToDense},
    {NODE_TYPE_SPLIT, &ConvertSplit},
    {NODE_TYPE_SQRT, &ConvertSqrt},
    {NODE_TYPE_SQUARED_DIFFERENCE, &ConvertSquaredDifference},
    {NODE_TYPE_SQUEEZE, &ConvertSqueeze},
    {NODE_TYPE_SQUARE, &ConvertSquare},
    {NODE_TYPE_STACK, &ConvertStack},
    {NODE_TYPE_STRIDED_SLICE, &ConvertStridedSlice},
    {NODE_TYPE_SUB_FUSION, &ConvertSubFusion},
    {NODE_TYPE_TILE_FUSION, &ConvertTileFusion},
    {NODE_TYPE_TOPK_FUSION, &ConvertTopKFusion},
    {NODE_TYPE_TRANSPOSE, &ConvertTranspose},
    {NODE_TYPE_UNSQUEEZE, &ConvertUnsqueeze},
    {NODE_TYPE_UNSTACK, &ConvertUnstack},
    {NODE_TYPE_WHERE, &ConvertWhere},
    {NODE_TYPE_SELECT, &ConvertSelect},
    {NODE_TYPE_ERF, &ConvertErf},
    {NODE_TYPE_LOG_SOFTMAX, &ConvertLogSoftmax}};

std::vector<int8_t> Convert(OHOS::HDI::Nnrt::V2_1::NodeType type, const PrimitivePtr primitive)
{
    if (convertOpMap.find(type) != convertOpMap.end()) {
        return convertOpMap[type](primitive);
    }
    LOGE("MindIR_LiteGraph_To_Model v2_1 failed, nodeType invalid, type =%d", type);
    return {};
}

inline std::vector<OHOS::HDI::Nnrt::V2_1::QuantParam> MindIR_Tensor_GetQuantParams_OHOS(TensorPtr tensor)
{
    if (tensor != nullptr) {
        std::vector<OHOS::HDI::Nnrt::V2_1::QuantParam> result;
        auto src = mindspore::lite::MindIR_Tensor_GetQuantParams(tensor);
        if (src.empty()) {
            return {};
        }
        size_t size = src.size();
        for (size_t i = 0; i < size; i++) {
            OHOS::HDI::Nnrt::V2_1::QuantParam quantParam{src[i].numBits, src[i].zeroPoint, src[i].scale};
            result.emplace_back(quantParam);
        }
        return result;
    } else {
        return {};
    }
}

void HDIModel_Destroy(OHOS::HDI::Nnrt::V2_1::Model **model)
{
    if (model != nullptr && *model != nullptr) {
        auto modelData = *model;
        delete (modelData);
        *model = nullptr;
    }
}

OHOS::HDI::Nnrt::V2_1::SharedBuffer Copy_MindIR_Tensor_Data_To_HDIBuffer(const TensorPtr tensor,
    const OHOS::HDI::Nnrt::V2_1::SharedBuffer &bufferTemplete, uint8_t *mmapPtr, unsigned int offset)
{
    if (tensor == nullptr) {
        LOGE("");
        return {-1, 0, offset, 0};
    }
    if (mmapPtr == nullptr) {
        LOGE("Tensor GetData failed, mmap pointer should not be nullptr");
        return {-1, 0, offset, 0};
    }

    OHOS::HDI::Nnrt::V2_1::SharedBuffer result{};
    std::vector<uint8_t> data = mindspore::lite::MindIR_Tensor_GetData(tensor);
    if (data.empty()) {
        result.fd = -1;
        result.bufferSize = bufferTemplete.bufferSize;
        result.offset = offset;
        result.dataSize = 0;
        return result;
    }
    result.fd = bufferTemplete.fd;
    result.bufferSize = bufferTemplete.bufferSize;
    auto ret = memcpy_s(mmapPtr + offset, data.size(), data.data(), data.size());
    if (ret != EOK) {
        LOGE("Tensor memcpy failed.");
        return {-1, 0, offset, 0};
    }
    result.offset = offset;
    result.dataSize = data.size();
    return result;
}

OHOS::HDI::Nnrt::V2_1::Model *LiteGraph_To_HDIModel(const mindspore::lite::LiteGraph *liteGraph,
    const OHOS::HDI::Nnrt::V2_1::SharedBuffer &buffer)
{
    if (liteGraph == nullptr) {
        LOGE("MindIR_LiteGraph_To_Model v2_1 failed, lite graph is nullptr.");
        return nullptr;
    }

    LOGI("MindIR_LiteGraph_To_Model begin");

    std::vector<OHOS::HDI::Nnrt::V2_1::Node> nodes;
    std::vector<OHOS::HDI::Nnrt::V2_1::Tensor> allTensors;
    std::vector<OHOS::HDI::Nnrt::V2_1::SubGraph> subGraph;

    // nodes
    for (auto node : liteGraph->all_nodes_) {
        if (node == nullptr) {
            LOGE("MindIR_LiteGraph_To_Model v2_1 failed, node is nullptr.");
            return nullptr;
        }
        OHOS::HDI::Nnrt::V2_1::Node tmp;
        tmp.name = node->name_;
        if (node->primitive_ == nullptr) {
            LOGE("MindIR_LiteGraph_To_Model v2_1 failed, node primitive is nullptr.");
            return nullptr;
        }
        tmp.nodeType = static_cast<OHOS::HDI::Nnrt::V2_1::NodeType>(
            mindspore::lite::MindIR_Primitive_GetType(node->primitive_));
        tmp.nodeAttr = Convert(tmp.nodeType, node->primitive_);
        tmp.inputIndex = node->input_indices_;
        tmp.outputIndex = node->output_indices_;
        tmp.quantType = static_cast<QuantType>(node->quant_type_);
        nodes.emplace_back(tmp);
    }

    // Tensor
    unsigned int tensorBufferOffset = 0;
    uint8_t *mmapPtr = nullptr;
    if (buffer.fd != -1) {
        mmapPtr =
          static_cast<uint8_t *>(mmap(nullptr, buffer.bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, buffer.fd, 0));
        if (mmapPtr == MAP_FAILED) {
            LOGE("MindIR_LiteGraph_To_Model v2_1 failed, mmap failed.");
            return nullptr;
        }
    }
    for (auto tensor : liteGraph->all_tensors_) {
        OHOS::HDI::Nnrt::V2_1::Tensor tmp;
        tmp.name = mindspore::lite::MindIR_Tensor_GetName(tensor);
        tmp.dataType = static_cast<DataType>(mindspore::lite::MindIR_Tensor_GetDataType(tensor));
        tmp.dims = mindspore::lite::MindIR_Tensor_GetDims(tensor);
        tmp.format = static_cast<Format>(mindspore::lite::MindIR_Tensor_GetFormat(tensor));
        tmp.data = Copy_MindIR_Tensor_Data_To_HDIBuffer(tensor, buffer, mmapPtr, tensorBufferOffset);
        tmp.quantParams = MindIR_Tensor_GetQuantParams_OHOS(tensor);
        allTensors.emplace_back(tmp);
        tensorBufferOffset = tmp.data.offset + tmp.data.dataSize;
    }
    if (buffer.fd != -1) {
        auto munmapRes = munmap(mmapPtr, buffer.bufferSize);
        if (munmapRes != 0) {
            LOGE("MindIR_LiteGraph_To_Model v2_1 failed, unmap failed.");
            return nullptr;
        }
    }

    // SubGraph
    for (auto graph : liteGraph->sub_graphs_) {
        OHOS::HDI::Nnrt::V2_1::SubGraph tmp;
        tmp.name = graph->name_;
        tmp.inputIndices = std::vector<uint32_t>(graph->input_indices_);
        tmp.outputIndices = std::vector<uint32_t>(graph->output_indices_);
        tmp.nodeIndices = std::vector<uint32_t>(graph->node_indices_);
        subGraph.emplace_back(tmp);
    }

    auto *retModel = new (std::nothrow) OHOS::HDI::Nnrt::V2_1::Model();
    if (retModel == nullptr) {
        LOGE("MindIR_LiteGraph_To_Model v2_1 failed, new Model failed.");
        return nullptr;
    }
    retModel->name = liteGraph->name_;
    retModel->inputIndex = liteGraph->input_indices_;
    retModel->outputIndex = liteGraph->output_indices_;
    retModel->nodes = nodes;
    retModel->allTensors = allTensors;
    retModel->subGraph = subGraph;
    return retModel;
}
} // NNRt_V2_1
} // NeuralNetworkRuntime
} // OHOS