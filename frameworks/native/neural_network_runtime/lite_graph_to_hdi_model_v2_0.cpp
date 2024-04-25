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

#include "lite_graph_to_hdi_model_v2_0.h"
#include <vector>
#include <algorithm>
#include <sys/mman.h>
#include "common/log.h"
#include "message_parcel.h"
#include "nnrt/v2_0/nnrt_types.h"
#include "nnrt/v2_0/node_attr_types.h"
#include "securec.h"

using namespace OHOS::HDI::Nnrt::V2_0;
typedef void *PrimitivePtr;
typedef void *TensorPtr;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace V2 {
std::vector<int8_t> ConvertActivation(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertActivation v2 failed, primitive is nullptr.");
        return {};
    }

    Activation activation{};
    activation.activationType = static_cast<HDI::Nnrt::V2_0::ActivationType>(
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

std::vector<int8_t> ConvertAddFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertAddFusion v2 failed, primitive is nullptr.");
        return {};
    }

    AddFusion addFusion{};
    addFusion.activationType = static_cast<HDI::Nnrt::V2_0::ActivationType>(
        mindspore::lite::MindIR_Activation_GetActivationType(primitive));

    OHOS::MessageParcel data;
    (void)AddFusionBlockMarshalling(data, addFusion);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertArgMaxFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertArgMaxFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertAvgPoolFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertAvgPoolFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertBatchToSpaceND(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertBatchToSpaceND v2 failed, primitive is nullptr.");
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
        LOGE("ConvertBiasAdd v2 failed, primitive is nullptr.");
        return {};
    }

    BiasAdd biasAdd{};
    OHOS::MessageParcel data;
    (void)BiasAddBlockMarshalling(data, biasAdd);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertCast(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertCast v2 failed, primitive is nullptr.");
        return {};
    }

    Cast cast{};
    OHOS::MessageParcel data;
    (void)CastBlockMarshalling(data, cast);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertConcat(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertConcat v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertConv2DFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertConv2DFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertConv2dTransposeFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertConv2dTransposeFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertDivFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertDivFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertEltwise(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertEltwise v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertExpandDims(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertExpandDims v2 failed, primitive is nullptr.");
        return {};
    }

    ExpandDims expandDims{};
    OHOS::MessageParcel data;
    (void)ExpandDimsBlockMarshalling(data, expandDims);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertFill(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertFill v2 failed, primitive is nullptr.");
        return {};
    }

    Fill fill{};
    OHOS::MessageParcel data;
    (void)FillBlockMarshalling(data, fill);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertFullConnection(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertFullConnection v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertFusedBatchNorm(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertFusedBatchNorm v2 failed, primitive is nullptr.");
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
        LOGE("ConvertGather v2 failed, primitive is nullptr.");
        return {};
    }

    Gather gather{};
    OHOS::MessageParcel data;
    (void)GatherBlockMarshalling(data, gather);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertLayerNormFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertGather v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertLessEqual(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertLessEqual v2 failed, primitive is nullptr.");
        return {};
    }

    LessEqual lessEqual{};
    OHOS::MessageParcel data;
    (void)LessEqualBlockMarshalling(data, lessEqual);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertMatMulFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMatMulFusion v2 failed, primitive is nullptr.");
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
        LOGE("ConvertMaximum v2 failed, primitive is nullptr.");
        return {};
    }

    Maximum maximum{};
    OHOS::MessageParcel data;
    (void)MaximumBlockMarshalling(data, maximum);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertMaxPoolFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMaxPoolFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertMulFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertMulFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertOneHot(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertOneHot v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertPadFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertPadFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertPowFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertPowFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertPReLUFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertPReLUFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertQuantDTypeCast(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertPReLUFusion v2 failed, primitive is nullptr.");
        return {};
    }

    QuantDTypeCast quantDTypeCast{};
    quantDTypeCast.srcT = mindspore::lite::MindIR_QuantDTypeCast_GetSrcT(primitive);
    quantDTypeCast.dstT = mindspore::lite::MindIR_QuantDTypeCast_GetDstT(primitive);
    OHOS::MessageParcel data;
    (void)QuantDTypeCastBlockMarshalling(data, quantDTypeCast);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertReduceFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertReduceFusion v2 failed, primitive is nullptr.");
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
        LOGE("ConvertReshape v2 failed, primitive is nullptr.");
        return {};
    }

    Reshape reshape{};
    OHOS::MessageParcel data;
    (void)ReshapeBlockMarshalling(data, reshape);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertResize(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertResize v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertRsqrt(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertRsqrt v2 failed, primitive is nullptr.");
        return {};
    }

    Rsqrt rsqrt{};
    OHOS::MessageParcel data;
    (void)RsqrtBlockMarshalling(data, rsqrt);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertScaleFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertScaleFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertShape(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertShape v2 failed, primitive is nullptr.");
        return {};
    }

    Shape shape{};
    OHOS::MessageParcel data;
    (void)ShapeBlockMarshalling(data, shape);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}
std::vector<int8_t> ConvertSliceFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSliceFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertSoftmax(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSoftmax v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertSpaceToBatchND(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSpaceToBatchND v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertSplit(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSplit v2 failed, primitive is nullptr.");
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
        LOGE("ConvertSqrt v2 failed, primitive is nullptr.");
        return {};
    }

    Sqrt sqrt{};
    OHOS::MessageParcel data;
    (void)SqrtBlockMarshalling(data, sqrt);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}
std::vector<int8_t> ConvertSquaredDifference(const PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSquaredDifference v2 failed, primitive is nullptr.");
        return {};
    }

    SquaredDifference squaredDifference{};
    OHOS::MessageParcel data;
    (void)SquaredDifferenceBlockMarshalling(data, squaredDifference);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertSqueeze(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSqueeze v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertStack(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertStack v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertStridedSlice(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertStridedSlice v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertSubFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertSubFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertTileFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertTileFusion v2 failed, primitive is nullptr.");
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

std::vector<int8_t> ConvertTopKFusion(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertTopKFusion v2 failed, primitive is nullptr.");
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
        LOGE("ConvertTranspose v2 failed, primitive is nullptr.");
        return {};
    }

    Transpose transpose{};
    OHOS::MessageParcel data;
    (void)TransposeBlockMarshalling(data, transpose);
    std::vector<int8_t> ret(reinterpret_cast<const int8_t *>(data.GetData()),
                            reinterpret_cast<const int8_t *>(data.GetData()) + data.GetDataSize());
    return ret;
}

std::vector<int8_t> ConvertUnsqueeze(PrimitivePtr primitive)
{
    if (primitive == nullptr) {
        LOGE("ConvertUnsqueeze v2 failed, primitive is nullptr.");
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

std::unordered_map<NodeType, std::vector<int8_t>(*)(PrimitivePtr)> convertOpMap = {
    {NODE_TYPE_ACTIVATION, &ConvertActivation},
    {NODE_TYPE_ADD_FUSION, &ConvertAddFusion},
    {NODE_TYPE_ARGMAX_FUSION, &ConvertArgMaxFusion},
    {NODE_TYPE_AVGPOOL_FUSION, &ConvertAvgPoolFusion},
    {NODE_TYPE_BATCH_TO_SPACE_ND, &ConvertBatchToSpaceND},
    {NODE_TYPE_BIAS_ADD, &ConvertBiasAdd},
    {NODE_TYPE_CAST, &ConvertCast},
    {NODE_TYPE_CONCAT, &ConvertConcat},
    {NODE_TYPE_CONV2D_FUSION, &ConvertConv2DFusion},
    {NODE_TYPE_CONV2D_TRANSPOSE_FUSION, &ConvertConv2dTransposeFusion},
    {NODE_TYPE_DIV_FUSION, &ConvertDivFusion},
    {NODE_TYPE_ELTWISE, &ConvertEltwise},
    {NODE_TYPE_EXPAND_DIMS, &ConvertExpandDims},
    {NODE_TYPE_FILL, &ConvertFill},
    {NODE_TYPE_FULL_CONNECTION, &ConvertFullConnection},
    {NODE_TYPE_FUSED_BATCH_NORM, &ConvertFusedBatchNorm},
    {NODE_TYPE_GATHER, &ConvertGather},
    {NODE_TYPE_LAYER_NORM_FUSION, &ConvertLayerNormFusion},
    {NODE_TYPE_LESS_EQUAL, &ConvertLessEqual},
    {NODE_TYPE_MATMUL_FUSION, &ConvertMatMulFusion},
    {NODE_TYPE_MAXIMUM, &ConvertMaximum},
    {NODE_TYPE_MAX_POOL_FUSION, &ConvertMaxPoolFusion},
    {NODE_TYPE_MUL_FUSION, &ConvertMulFusion},
    {NODE_TYPE_ONE_HOT, &ConvertOneHot},
    {NODE_TYPE_PAD_FUSION, &ConvertPadFusion},
    {NODE_TYPE_POW_FUSION, &ConvertPowFusion},
    {NODE_TYPE_PRELU_FUSION, &ConvertPReLUFusion},
    {NODE_TYPE_QUANT_DTYPE_CAST, &ConvertQuantDTypeCast},
    {NODE_TYPE_REDUCE_FUSION, &ConvertReduceFusion},
    {NODE_TYPE_RESHAPE, &ConvertReshape},
    {NODE_TYPE_RESIZE, &ConvertResize},
    {NODE_TYPE_RSQRT, &ConvertRsqrt},
    {NODE_TYPE_SCALE_FUSION, &ConvertScaleFusion},
    {NODE_TYPE_SHAPE, &ConvertShape},
    {NODE_TYPE_SLICE_FUSION, &ConvertSliceFusion},
    {NODE_TYPE_SOFTMAX, &ConvertSoftmax},
    {NODE_TYPE_SPACE_TO_BATCH_ND, &ConvertSpaceToBatchND},
    {NODE_TYPE_SPLIT, &ConvertSplit},
    {NODE_TYPE_SQRT, &ConvertSqrt},
    {NODE_TYPE_SQUARED_DIFFERENCE, &ConvertSquaredDifference},
    {NODE_TYPE_SQUEEZE, &ConvertSqueeze},
    {NODE_TYPE_STACK, &ConvertStack},
    {NODE_TYPE_STRIDED_SLICE, &ConvertStridedSlice},
    {NODE_TYPE_SUB_FUSION, &ConvertSubFusion},
    {NODE_TYPE_TILE_FUSION, &ConvertTileFusion},
    {NODE_TYPE_TOPK_FUSION, &ConvertTopKFusion},
    {NODE_TYPE_TRANSPOSE, &ConvertTranspose},
    {NODE_TYPE_UNSQUEEZE, &ConvertUnsqueeze}};

std::vector<int8_t> Convert(NodeType type, PrimitivePtr primitive)
{
    if (convertOpMap.find(type) != convertOpMap.end()) {
        return convertOpMap[type](primitive);
    }
    LOGE("MindIR_LiteGraph_To_Model v2_0 failed, nodeType invalid, type =%d", type);
    return {};
}

inline std::vector<OHOS::HDI::Nnrt::V2_0::QuantParam> MindIR_Tensor_GetQuantParams_OHOS(TensorPtr tensor)
{
    if (tensor != nullptr) {
        std::vector<OHOS::HDI::Nnrt::V2_0::QuantParam> result;
        auto src = mindspore::lite::MindIR_Tensor_GetQuantParams(tensor);
        if (src.empty()) {
            return {};
        }
        size_t size = src.size();
        for (size_t i = 0; i < size; i++) {
            OHOS::HDI::Nnrt::V2_0::QuantParam quantParam{src[i].numBits, src[i].zeroPoint, src[i].scale};
            result.emplace_back(quantParam);
        }
        return result;
    } else {
        return {};
    }
}

void HDIModel_Destroy(OHOS::HDI::Nnrt::V2_0::Model **model)
{
    if (model != nullptr && *model != nullptr) {
        auto modelData = *model;
        delete (modelData);
        *model = nullptr;
    }
}

OHOS::HDI::Nnrt::V2_0::SharedBuffer Copy_MindIR_Tensor_Data_To_HDIBuffer(const TensorPtr tensor,
    const OHOS::HDI::Nnrt::V2_0::SharedBuffer &bufferTemplete, uint8_t *mmapPtr, unsigned int offset)
{
    if (tensor == nullptr) {
        LOGE("");
        return {-1, 0, offset, 0};
    }
    if (mmapPtr == nullptr) {
        LOGE("Tensor GetData failed, mmap pointer should not be nullptr");
        return {-1, 0, offset, 0};
    }

    OHOS::HDI::Nnrt::V2_0::SharedBuffer result{};
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

OHOS::HDI::Nnrt::V2_0::Model *LiteGraph_To_HDIModel(const mindspore::lite::LiteGraph *liteGraph,
    const OHOS::HDI::Nnrt::V2_0::SharedBuffer &buffer)
{
    if (liteGraph == nullptr) {
        LOGE("MindIR_LiteGraph_To_Model v2 failed, lite graph is nullptr.");
        return nullptr;
    }

    LOGI("MindIR_LiteGraph_To_Model begin");

    std::vector<OHOS::HDI::Nnrt::V2_0::Node> nodes;
    std::vector<OHOS::HDI::Nnrt::V2_0::Tensor> allTensors;
    std::vector<OHOS::HDI::Nnrt::V2_0::SubGraph> subGraph;

    // nodes
    for (auto node : liteGraph->all_nodes_) {
        if (node == nullptr) {
            LOGE("MindIR_LiteGraph_To_Model v2 failed, node is nullptr.");
            return nullptr;
        }
        OHOS::HDI::Nnrt::V2_0::Node tmp;
        tmp.name = node->name_;
        if (node->primitive_ == nullptr) {
            LOGE("MindIR_LiteGraph_To_Model v2 failed, node primitive is nullptr.");
            return nullptr;
        }
        tmp.nodeType = static_cast<NodeType>(mindspore::lite::MindIR_Primitive_GetType(node->primitive_));
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
            LOGE("MindIR_LiteGraph_To_Model v2 failed, mmap failed.");
            return nullptr;
        }
    }
    for (auto tensor : liteGraph->all_tensors_) {
        OHOS::HDI::Nnrt::V2_0::Tensor tmp;
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
            LOGE("MindIR_LiteGraph_To_Model v2 failed, unmap failed.");
            return nullptr;
        }
    }

    // SubGraph
    for (auto graph : liteGraph->sub_graphs_) {
        OHOS::HDI::Nnrt::V2_0::SubGraph tmp;
        tmp.name = graph->name_;
        tmp.inputIndices = std::vector<uint32_t>(graph->input_indices_);
        tmp.outputIndices = std::vector<uint32_t>(graph->output_indices_);
        tmp.nodeIndices = std::vector<uint32_t>(graph->node_indices_);
        subGraph.emplace_back(tmp);
    }

    auto *retModel = new (std::nothrow) Model();
    if (retModel == nullptr) {
        LOGE("MindIR_LiteGraph_To_Model v2 failed, new Model failed.");
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

} // V2
} // NeuralNetworkRuntime
} // OHOS