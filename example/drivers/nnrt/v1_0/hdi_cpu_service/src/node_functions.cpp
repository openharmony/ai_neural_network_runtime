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

#include "node_functions.h"

#include "node_registry.h"
#include <message_parcel.h>
#include <v1_0/node_attr_types.h>

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
PrimUniquePtr GetAddPrimitive(const std::vector<int8_t>& primitive)
{
    AddFusion addAttr;
    auto ret = ParsePrimitive<AddFusion>(primitive, addAttr, AddFusionBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of AddFusion operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_AddFusion;
    auto attr = new (std::nothrow) mindspore::schema::AddFusionT;
    if (attr == nullptr) {
        HDF_LOGE("Create AddFusion primitive failed.");
        return nullptr;
    }
    attr->activation_type = static_cast<mindspore::schema::ActivationType>(addAttr.activationType);
    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetAvgPoolPrimitive(const std::vector<int8_t>& primitive)
{
    AvgPoolFusion avgPoolAttr;
    auto ret = ParsePrimitive<AvgPoolFusion>(primitive, avgPoolAttr, AvgPoolFusionBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of AvgPoolFusion operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_AvgPoolFusion;

    auto attr = new (std::nothrow) mindspore::schema::AvgPoolFusionT;
    if (attr == nullptr) {
        HDF_LOGE("Create AvgPoolFusion primitive failed.");
        return nullptr;
    }
    attr->kernel_size = avgPoolAttr.kernelSize;
    attr->strides = avgPoolAttr.strides;
    attr->pad = avgPoolAttr.pad;
    attr->pad_mode = static_cast<mindspore::schema::PadMode>(avgPoolAttr.padMode);
    attr->round_mode = static_cast<mindspore::schema::RoundMode>(avgPoolAttr.roundMode);
    attr->format = static_cast<mindspore::schema::Format>(avgPoolAttr.format);
    attr->global = avgPoolAttr.global;
    attr->activation_type = static_cast<mindspore::schema::ActivationType>(avgPoolAttr.activationType);
    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetConcatPrimitive(const std::vector<int8_t>& primitive)
{
    Concat concatAttr;
    auto ret = ParsePrimitive<Concat>(primitive, concatAttr, ConcatBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of Concat operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_Concat;

    auto attr = new (std::nothrow) mindspore::schema::ConcatT;
    if (attr == nullptr) {
        HDF_LOGE("Create concat primitive failed.");
        return nullptr;
    }
    attr->axis = concatAttr.axis;
    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetConv2dPrimitive(const std::vector<int8_t>& primitive)
{
    Conv2DFusion conv2dAttr;
    auto ret = ParsePrimitive<Conv2DFusion>(primitive, conv2dAttr, Conv2DFusionBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of Conv2DFusion operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_Conv2DFusion;

    auto attr = new (std::nothrow) mindspore::schema::Conv2DFusionT;
    if (attr == nullptr) {
        HDF_LOGE("Create Conv2DFusion primitive failed.");
        return nullptr;
    }

    attr->kernel_size = conv2dAttr.kernelSize;
    attr->stride = conv2dAttr.stride;
    attr->dilation = conv2dAttr.dilation;
    attr->pad_mode = static_cast<mindspore::schema::PadMode>(conv2dAttr.padMode);
    attr->pad_list = conv2dAttr.padList;
    attr->group = conv2dAttr.group;
    attr->in_channel = conv2dAttr.inChannel;
    attr->out_channel = conv2dAttr.outChannel;
    attr->activation_type = static_cast<mindspore::schema::ActivationType>(conv2dAttr.activationType);

    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetFullConnectionPrimitive(const std::vector<int8_t>& primitive)
{
    FullConnection fullConnAttr;
    auto ret = ParsePrimitive<FullConnection>(primitive, fullConnAttr, FullConnectionBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of FullConnection operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_FullConnection;

    auto attr = new (std::nothrow) mindspore::schema::FullConnectionT;
    if (attr == nullptr) {
        HDF_LOGE("Create FullConnection primitive failed.");
        return nullptr;
    }

    attr->has_bias = fullConnAttr.hasBias;
    attr->use_axis = fullConnAttr.useAxis;
    attr->axis = fullConnAttr.axis;
    attr->activation_type = static_cast<mindspore::schema::ActivationType>(fullConnAttr.activationType);

    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetMaxPoolFusionPrimitive(const std::vector<int8_t>& primitive)
{
    MaxPoolFusion maxPoolAttr;
    auto ret = ParsePrimitive<MaxPoolFusion>(primitive, maxPoolAttr, MaxPoolFusionBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of MaxPoolFusion operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_MaxPoolFusion;

    auto attr = new (std::nothrow) mindspore::schema::MaxPoolFusionT;
    if (attr == nullptr) {
        HDF_LOGE("Create MaxPoolFusion primitive failed.");
        return nullptr;
    }

    attr->kernel_size = maxPoolAttr.kernelSize;
    attr->strides = maxPoolAttr.strides;
    attr->pad = maxPoolAttr.pad;
    attr->pad_mode = static_cast<mindspore::schema::PadMode>(maxPoolAttr.padMode);
    attr->format = static_cast<mindspore::schema::Format>(maxPoolAttr.format);
    attr->global = maxPoolAttr.global;
    attr->activation_type = static_cast<mindspore::schema::ActivationType>(maxPoolAttr.activationType);

    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetMatMulFusionPrimitive(const std::vector<int8_t>& primitive)
{
    MatMulFusion matmulAttr;
    auto ret = ParsePrimitive<MatMulFusion>(primitive, matmulAttr, MatMulFusionBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of MatMulFusion operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_MatMulFusion;

    auto attr = new (std::nothrow) mindspore::schema::MatMulFusionT;
    if (attr == nullptr) {
        HDF_LOGE("Create MatMulFusion primitive failed.");
        return nullptr;
    }

    attr->transpose_a = matmulAttr.transposeA;
    attr->transpose_b = matmulAttr.transposeB;
    attr->activation_type = static_cast<mindspore::schema::ActivationType>(matmulAttr.activationType);

    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetSoftmaxPrimitive(const std::vector<int8_t>& primitive)
{
    Softmax softmaxAttr;
    auto ret = ParsePrimitive<Softmax>(primitive, softmaxAttr, SoftmaxBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of Softmax operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_Softmax;

    auto attr = new (std::nothrow) mindspore::schema::SoftmaxT;
    if (attr == nullptr) {
        HDF_LOGE("Create Softmax primitive failed.");
        return nullptr;
    }

    attr->axis = softmaxAttr.axis;
    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetReshapePrimitive(const std::vector<int8_t>& primitive)
{
    Reshape reshapeAttr;
    auto ret = ParsePrimitive<Reshape>(primitive, reshapeAttr, ReshapeBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of Reshape operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_Reshape;

    auto attr = new (std::nothrow) mindspore::schema::ReshapeT;
    if (attr == nullptr) {
        HDF_LOGE("Create Reshape primitive failed.");
        return nullptr;
    }

    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetScaleFusionPrimitive(const std::vector<int8_t>& primitive)
{
    ScaleFusion scaleAttr;
    auto ret = ParsePrimitive<ScaleFusion>(primitive, scaleAttr, ScaleFusionBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of ScaleFusion operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_ScaleFusion;

    auto attr = new (std::nothrow) mindspore::schema::ScaleFusionT;
    if (attr == nullptr) {
        HDF_LOGE("Create ScaleFusion primitive failed.");
        return nullptr;
    }

    attr->axis = scaleAttr.axis;
    attr->activation_type = static_cast<mindspore::schema::ActivationType>(scaleAttr.activationType);
    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetActivationPrimitive(const std::vector<int8_t>& primitive)
{
    Activation actAttr;
    auto ret = ParsePrimitive<Activation>(primitive, actAttr, ActivationBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of Activation operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_Activation;

    auto attr = new (std::nothrow) mindspore::schema::ActivationT;
    if (attr == nullptr) {
        HDF_LOGE("Create Activation primitive failed.");
        return nullptr;
    }

    attr->alpha = actAttr.alpha;
    attr->min_val = actAttr.minVal;
    attr->max_val = actAttr.maxVal;
    attr->approximate = actAttr.approximate;
    attr->activation_type = static_cast<mindspore::schema::ActivationType>(actAttr.activationType);

    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetQuantDTypeCastPrimitive(const std::vector<int8_t>& primitive)
{
    QuantDTypeCast quantAttr;
    auto ret = ParsePrimitive<QuantDTypeCast>(primitive, quantAttr, QuantDTypeCastBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of QuantDTypeCast operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_QuantDTypeCast;

    auto attr = new (std::nothrow) mindspore::schema::QuantDTypeCastT;
    if (attr == nullptr) {
        HDF_LOGE("Create QuantDTypeCast primitive failed.");
        return nullptr;
    }

    attr->src_t = quantAttr.srcT;
    attr->dst_t = quantAttr.dstT;
    prim->value.value = attr;
    return prim;
}

PrimUniquePtr GetMulFusionPrimitive(const std::vector<int8_t>& primitive)
{
    MulFusion mulAttr;
    auto ret = ParsePrimitive<MulFusion>(primitive, mulAttr, MulFusionBlockUnmarshalling);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse primitive data of MulFusion operator failed.");
        return nullptr;
    }

    auto prim = std::make_unique<mindspore::schema::PrimitiveT>();
    prim->value.type = mindspore::schema::PrimitiveType_MulFusion;

    auto attr = new (std::nothrow) mindspore::schema::MulFusionT;
    if (attr == nullptr) {
        HDF_LOGE("Create MulFusion primitive failed.");
        return nullptr;
    }

    attr->activation_type = static_cast<mindspore::schema::ActivationType>(mulAttr.activationType);
    prim->value.value = attr;
    return prim;
}

REGISTER_NODE(Activation, NodeType::NODE_TYPE_ACTIVATION, GetActivationPrimitive);
REGISTER_NODE(AddFusion, NodeType::NODE_TYPE_ADD_FUSION, GetAddPrimitive);
REGISTER_NODE(AvgPoolFusion, NodeType::NODE_TYPE_AVGPOOL_FUSION, GetAvgPoolPrimitive);
REGISTER_NODE(Concat, NodeType::NODE_TYPE_CONCAT, GetConcatPrimitive);
REGISTER_NODE(Conv2DFusion, NodeType::NODE_TYPE_CONV2D_FUSION, GetConv2dPrimitive);
REGISTER_NODE(FullConnection, NodeType::NODE_TYPE_FULL_CONNECTION, GetFullConnectionPrimitive);
REGISTER_NODE(MaxPoolFusion, NodeType::NODE_TYPE_MAX_POOL_FUSION, GetMaxPoolFusionPrimitive);
REGISTER_NODE(MatMulFusion, NodeType::NODE_TYPE_MATMUL_FUSION, GetMatMulFusionPrimitive);
REGISTER_NODE(Reshape, NodeType::NODE_TYPE_RESHAPE, GetReshapePrimitive);
REGISTER_NODE(Softmax, NodeType::NODE_TYPE_SOFTMAX, GetSoftmaxPrimitive);
REGISTER_NODE(ScaleFusion, NodeType::NODE_TYPE_SCALE_FUSION, GetScaleFusionPrimitive);
REGISTER_NODE(QuantDTypeCast, NodeType::NODE_TYPE_QUANT_DTYPE_CAST, GetQuantDTypeCastPrimitive);
REGISTER_NODE(MulFusion, NodeType::NODE_TYPE_MUL_FUSION, GetMulFusionPrimitive);
} // namespace V1_0
} // namespace Nnrt
} // namespace HDI
} // namespace OHOS