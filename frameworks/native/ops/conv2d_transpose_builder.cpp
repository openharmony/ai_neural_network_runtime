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

#include "conv2d_transpose_builder.h"

#include "frameworks/native/transform.h"
#include "frameworks/native/validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static constexpr int INPUT_NUM = 3;
static constexpr int OUTPUT_NUM = 1;
static constexpr int INPUT_WEIGHT = 1;
static constexpr int WEIGHT_SIZE = 4;
static constexpr int OUT_CHANNEL_INDEX = 0;
static constexpr int IN_CHANNEL_INDEX = 3;
static constexpr int KERNEL_HEIGHT_INDEX = 1;
static constexpr int KERNEL_WEIGHT_INDEX = 2;
static constexpr int PAD_MODE_PARAM_NUM = 1;
static constexpr int PAD_LIST_PARAM_NUM = 4;
static constexpr int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "Conv2DTranspose";

Conv2DTransposeBuilder::Conv2DTransposeBuilder() {}

Conv2DTransposeBuilder::~Conv2DTransposeBuilder() {}

OH_NN_ReturnCode Conv2DTransposeBuilder::SetInput(const std::vector<uint32_t>& inputsIndex,
                                                  const std::vector<uint32_t>& outputsIndex,
                                                  const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Conv2dTranspose] SetInput failed, Passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    // set inChannel, outChannel, kernelSize
    auto weightShape = allTensors[inputsIndex[INPUT_WEIGHT]]->GetDimensions();
    if (weightShape.size() != WEIGHT_SIZE) {
        LOGE("[Conv2dTranspose] SetInput failed, the dimension of weight should be %d", WEIGHT_SIZE);
        return OH_NN_INVALID_PARAMETER;
    }

    m_inChannel = weightShape[IN_CHANNEL_INDEX];
    m_outChannel = weightShape[OUT_CHANNEL_INDEX];

    return OH_NN_SUCCESS;
}

void Conv2DTransposeBuilder::SetKernelSize(const std::vector<uint32_t>& inputsIndex,
                                           const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    auto weightShape = allTensors[inputsIndex[INPUT_WEIGHT]]->GetDimensions();

    m_kernelSize.clear();
    m_kernelSize.emplace_back(weightShape[KERNEL_HEIGHT_INDEX]);
    m_kernelSize.emplace_back(weightShape[KERNEL_WEIGHT_INDEX]);
}

OH_NN_ReturnCode Conv2DTransposeBuilder::SetStrides(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set Strides
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Conv2DTranspose] SetStrides failed, the Strides should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2DTranspose] SetStrides GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    const int64_t* pStrides = reinterpret_cast<const int64_t*>(buffer);
    int elementSize = tensor->GetElementCount();
    m_strides.assign(pStrides, pStrides + elementSize);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DTransposeBuilder::SetDilation(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set Dilation
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Conv2DTranspose] SetDilation failed, the Dilation should be type OH_NN_INT64");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2DTranspose] SetDilation GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    const int64_t* pDilation = reinterpret_cast<const int64_t*>(buffer);
    int dilationSize = tensor->GetElementCount();
    m_dilation.assign(pDilation, pDilation + dilationSize);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DTransposeBuilder::SetPad(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    bool isPadMode = false;
    if (tensor->GetElementCount() == PAD_MODE_PARAM_NUM) {
        isPadMode = true;
    } else if (tensor->GetElementCount() != PAD_LIST_PARAM_NUM) {
        LOGE("[Conv2DTranspose] SetPad failed, the inputs should be 1 if using padMode or 4 if using padList.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2DTranspose] SetPadMode GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    // Set PadMode or PadList
    if (isPadMode) {
        if (tensor->GetDataType() != OH_NN_INT8) {
            LOGE("[Conv2DTranspose] SetPad failed, the PadMode should have type OH_NN_INT8.");
            return OH_NN_INVALID_PARAMETER;
        }

        int8_t* pPad = static_cast<int8_t*>(buffer);
        if (!OHOS::NeuralNetworkRuntime::Validation::ValidatePadMode(*pPad)) {
            LOGE("[Conv2DTranspose] SetPad failed, invalid pad mode.");
            return OH_NN_INVALID_PARAMETER;
        }
        m_padMode = NNToMS::TransformPadModeValue(*pPad);
    } else {
        if (tensor->GetDataType() != OH_NN_INT64) {
            LOGE("[Conv2DTranspose] SetPad failed, the PadList should have type OH_NN_INT64.");
            return OH_NN_INVALID_PARAMETER;
        }

        const int64_t* pPadList = reinterpret_cast<const int64_t*>(buffer);
        int padListPadSize = tensor->GetElementCount();
        m_padList.assign(pPadList, pPadList + padListPadSize);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DTransposeBuilder::SetGroup(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set Group
    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Conv2dTranspose] SetGroup failed, the Group shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Conv2dTranspose] SetGroup failed, the Group should have type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2DTranspose] SetGroup GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    m_group = *reinterpret_cast<const int64_t*>(buffer);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DTransposeBuilder::SetOutPadding(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set outputPadding
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Conv2DTranspose] SetOutPadding failed, the outputPadding should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2DTranspose] SetOutPadding GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    const int64_t* pOutputPadding = reinterpret_cast<const int64_t*>(buffer);
    int outputPadSize = tensor->GetElementCount();
    m_outputPaddings.assign(pOutputPadding, pOutputPadding + outputPadSize);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DTransposeBuilder::SetActivation(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Conv2DTranspose] SetActivation failed, the ActivationType shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[Conv2DTranspose] SetActivation failed, the ActivationType should have type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2DTranspose] SetOutPadding GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    int8_t* pFuseData = static_cast<int8_t*>(buffer);
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*pFuseData))) {
        LOGE("[Conv2DTranspose] SetActivation failed, activation input is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_activationType = NNToMS::TransfromFusionType((OH_NN_FuseType)(*pFuseData));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DTransposeBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                               const std::vector<uint32_t>& inputsIndex,
                                               const std::vector<uint32_t>& outputsIndex,
                                               const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Conv2DTranspose] Build failed, conv2DTranspose operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = SetInput(inputsIndex, outputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        return returnCode;
    }

    SetKernelSize(inputsIndex, allTensors);

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor =  allTensors[i]; // 参数 tensor
        switch (tensor->GetType()) {
            case OH_NN_CONV2D_TRANSPOSE_STRIDES:
                returnCode = SetStrides(tensor);
                break;
            case OH_NN_CONV2D_TRANSPOSE_DILATION:
                returnCode = SetDilation(tensor);
                break;
            case OH_NN_CONV2D_TRANSPOSE_PAD_MODE:
            case OH_NN_CONV2D_TRANSPOSE_PAD:
                returnCode = SetPad(tensor);
                break;
            case OH_NN_CONV2D_TRANSPOSE_GROUP:
                returnCode = SetGroup(tensor);
                break;
            case OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS:
                returnCode = SetOutPadding(tensor);
                break;
            case OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE:
                returnCode = SetActivation(tensor);
                break;
            default:
                LOGE("[Conv2DTranspose] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Conv2DTranspose] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr Conv2DTransposeBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Conv2DTranspose] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    void* primitive = MindIR_Conv2dTransposeFusion_CreatePrimitive(m_kernelSize,
        m_strides, m_dilation, m_padMode, m_padList, m_group, m_inChannel, m_outChannel,
        m_activationType, m_outputPaddings);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(Conv2DTransposeBuilder, OH_NN_OPS_CONV2D_TRANSPOSE);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
