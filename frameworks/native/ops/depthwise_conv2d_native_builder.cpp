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

#include "depthwise_conv2d_native_builder.h"

#include "frameworks/native/transform.h"
#include "frameworks/native/validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 3;
static const int OUTPUT_NUM = 1;
static const int PAD_MODE_SIZE = 1;
static const int PAD_LIST_SIZE = 4;
static const int IN_CHANNEL_IN_INPUT = 3;
static const int OUT_CHANNEL_IN_WEIGHT = 0;
static const int HEIGHT_IN_WEIGHT = 1;
static const int WIDTH_IN_WEIGHT = 2;
static const int INPUT_RANK = 4;
static const int INPUT_X = 0;
static const int INPUT_WEIGHT = 1;
static const int SCALE_LENGTH = 1;
static const std::string OP_NAME = "DepthwiseConv2DNative";

DepthwiseConv2DNativeBuilder::DepthwiseConv2DNativeBuilder() {}

DepthwiseConv2DNativeBuilder::~DepthwiseConv2DNativeBuilder() {}

OH_NN_ReturnCode DepthwiseConv2DNativeBuilder::SetIsPadMode(std::shared_ptr<NNTensor> tensor,
    bool &isPadMode)
{
    if (tensor->GetElementCount() == PAD_MODE_SIZE) {
        isPadMode = true;
    } else if (tensor->GetElementCount() != PAD_LIST_SIZE) {
        LOGE("[DepthwiseConv2DNative] The element size of padMode should be 1 or "
            "the element size of padList should be 4.");
        return OH_NN_INVALID_PARAMETER;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthwiseConv2DNativeBuilder::SetActivation(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set ActivationType
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[DepthwiseConv2DNative] SetActivation failed, the Activation should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[DepthwiseConv2DNative] SetActivation failed, the activationType should have type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DepthwiseConv2DNative] SetActivation GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    int8_t* pFuseData = static_cast<int8_t*>(buffer);
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*pFuseData))) {
        LOGE("[DepthwiseConv2DNative] SetActivation failed, activation input is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_activationType = NNToMS::TransfromFusionType((OH_NN_FuseType)(*pFuseData));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthwiseConv2DNativeBuilder::SetKernelSize(const std::vector<uint32_t>& inputsIndex,
    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    // Set kernleSize and outChannel
    auto weightShape = allTensors[inputsIndex[INPUT_WEIGHT]]->GetDimensions();
    if (weightShape.size() != INPUT_RANK) {
        LOGE("[DepthwiseConv2DNative] SetKernelSize failed, invalid rank of shape of weight, should be 4 dimensions.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_outChannel = weightShape[OUT_CHANNEL_IN_WEIGHT];
    m_kernelSize.clear();
    m_kernelSize.emplace_back(weightShape[HEIGHT_IN_WEIGHT]);
    m_kernelSize.emplace_back(weightShape[WIDTH_IN_WEIGHT]);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthwiseConv2DNativeBuilder::SetStrides(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[DepthwiseConv2DNative] SetStrides failed, the stride should have type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DepthwiseConv2DNative] SetStrides GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    const int64_t* pStrides = reinterpret_cast<const int64_t*>(buffer);
    int stridesSize = tensor->GetElementCount();
    m_strides.assign(pStrides, pStrides + stridesSize);

    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode DepthwiseConv2DNativeBuilder::SetDilation(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[DepthwiseConv2DNative] SetDilation failed, the dilation should have type OH_NN_INT64");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DepthwiseConv2DNative] SetDilation GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    const int64_t* pDilation = reinterpret_cast<const int64_t*>(buffer);
    int dilationSize = tensor->GetElementCount();
    m_dilation.assign(pDilation, pDilation + dilationSize);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthwiseConv2DNativeBuilder::SetPadModeOrPaddings(
    std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    bool isPadMode = false;
    OH_NN_ReturnCode ret = SetIsPadMode(tensor, isPadMode);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DepthwiseConv2DNative] SetPad GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    if (isPadMode) {
        if (tensor->GetDataType() != OH_NN_INT8) {
            LOGE("[DepthwiseConv2DNative] SetPadModeOrPaddings failed, the padMode should have type OH_NN_INT8.");
            return OH_NN_INVALID_PARAMETER;
        }

        int8_t* pPad = static_cast<int8_t*>(buffer);
        if (!OHOS::NeuralNetworkRuntime::Validation::ValidatePadMode(*pPad)) {
            LOGE("[DepthwiseConv2DNative] SetPadModeOrPaddings failed, invalid pad mode.");
            return OH_NN_INVALID_PARAMETER;
        }
        m_padMode = NNToMS::TransformPadModeValue(*pPad);
    } else {
        if (tensor->GetDataType() != OH_NN_INT64) {
            LOGE("[DepthwiseConv2DNative] SetPadModeOrPaddings failed, the padList should have type OH_NN_INT64.");
            return OH_NN_INVALID_PARAMETER;
        }

        const int64_t* pPadList = reinterpret_cast<const int64_t*>(buffer);
        int padListSize = tensor->GetElementCount();
        m_pad.assign(pPadList, pPadList + padListSize);
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthwiseConv2DNativeBuilder::SetInputAndOutput(
    const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex,
    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[DepthwiseConv2DNative] SetInputAndOutput failed, passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DepthwiseConv2DNativeBuilder::Build(const std::vector<uint32_t>& paramsIndex,
    const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex,
    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[DepthwiseConv2DNative] Build failed, operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode ret = SetInputAndOutput(inputsIndex, outputsIndex, allTensors);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    auto inputShape = allTensors[inputsIndex[INPUT_X]]->GetDimensions();
    if (inputShape.size() != INPUT_RANK) {
        LOGE("[DepthwiseConv2DNative] Build failed, invalid rank of shape of input, should be 4 dimensions.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_inChannel = inputShape[IN_CHANNEL_IN_INPUT];
    // Set Kernel Size
    ret = SetKernelSize(inputsIndex, allTensors);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[DepthwiseConv2DNative] Build failed, SetKernelSize failed.");
        return ret;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];  // 参数 tensor
        switch (tensor->GetType()) {
            case OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES:
                ret = SetStrides(tensor);
                break;
            case OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION:
                ret = SetDilation(tensor);
                break;
            case OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE:
            case OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD:
                ret = SetPadModeOrPaddings(tensor);
                break;
            case OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE:
                ret = SetActivation(tensor);
                break;
            default:
                LOGE("[DepthwiseConv2DNative] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }
        if (ret != OH_NN_SUCCESS) {
            LOGE("[DepthwiseConv2DNative] Build failed, passed invalid param.");
            return ret;
        }
    }

    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr DepthwiseConv2DNativeBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[DepthwiseConv2DNative] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = MindIR_Conv2DFusion_CreatePrimitive(m_kernelSize, m_strides,
        m_dilation, m_padMode, m_pad, m_inChannel, m_inChannel, m_outChannel, m_activationType);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(DepthwiseConv2DNativeBuilder, OH_NN_OPS_DEPTHWISE_CONV2D_NATIVE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
