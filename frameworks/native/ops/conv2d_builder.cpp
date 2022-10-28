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

#include "conv2d_builder.h"

#include "frameworks/native/transform.h"
#include "frameworks/native/validation.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static constexpr int INPUT_NUM = 3;
static constexpr int OUTPUT_NUM = 1;
static constexpr int CONV2D_INPUT_WEIGHT = 1;
static constexpr int WEIGHT_SIZE = 4;
static constexpr int OUT_CHANNEL_INDEX = 0;
static constexpr int IN_CHANNEL_INDEX = 3;
static constexpr int KERNEL_HEIGHT_INDEX = 1;
static constexpr int KERNEL_WEIGHT_INDEX = 2;
static constexpr int PAD_MODE_GET = 1;
static constexpr int PAD_LIST_GET = 4;
static constexpr int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "Conv2D";

Conv2DBuilder::Conv2DBuilder() {}

Conv2DBuilder::~Conv2DBuilder() {}

OH_NN_ReturnCode Conv2DBuilder::SetInputAndOutput(const std::vector<uint32_t>& inputsIndex,
                                                  const std::vector<uint32_t>& outputsIndex,
                                                  const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[Conv2d] SetInputAndOutput failed, passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DBuilder::SetChannel(const std::vector<uint32_t>& inputsIndex,
                                           const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    // set inChannel, outChannel, kernelSize
    auto weightShape = allTensors[inputsIndex[CONV2D_INPUT_WEIGHT]]->GetDimensions();
    if (weightShape.size() != WEIGHT_SIZE) {
        LOGE("[Conv2d] SetChannel failed, the dimension of weight should be %d", WEIGHT_SIZE);
        return OH_NN_INVALID_PARAMETER;
    }

    m_inChannel = weightShape[IN_CHANNEL_INDEX];
    m_outChannel = weightShape[OUT_CHANNEL_INDEX];

    return OH_NN_SUCCESS;
}

void Conv2DBuilder::SetKernelSize(const std::vector<uint32_t>& inputsIndex,
                                  const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    // set inChannel, outChannel, kernelSize
    auto weightShape = allTensors[inputsIndex[CONV2D_INPUT_WEIGHT]]->GetDimensions();

    m_kernelSize.clear();
    m_kernelSize.emplace_back(weightShape[KERNEL_HEIGHT_INDEX]);
    m_kernelSize.emplace_back(weightShape[KERNEL_WEIGHT_INDEX]);
}

OH_NN_ReturnCode Conv2DBuilder::SetStrides(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set Strides
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Conv2d] SetStrides failed, the Strides should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2d] SetStrides GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    const int64_t* pStrides = reinterpret_cast<const int64_t*>(buffer);
    int stridesSize = tensor->GetElementCount();
    m_strides.assign(pStrides, pStrides + stridesSize);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DBuilder::SetDilation(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set Dilation
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Conv2d] SetDilation failed, the Dilation should have type OH_NN_INT64");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2d] SetDilation GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    const int64_t* pDilation = reinterpret_cast<const int64_t*>(buffer);
    int dilationSize = tensor->GetElementCount();
    m_dilation.assign(pDilation, pDilation + dilationSize);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DBuilder::SetPad(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    bool isPadMode = false;
    if (tensor->GetElementCount() == PAD_MODE_GET) {
        isPadMode = true;
    } else if (tensor->GetElementCount() != PAD_LIST_GET) {
        LOGE("[Conv2d] SetPad failed, inputs should be 1 for padMode and 4 for padList.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2d] SetPadList GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }

    // Set PadMode or PadList
    if (isPadMode) {
        if (tensor->GetDataType() != OH_NN_INT8) {
            LOGE("[Conv2d] SetPad failed, the PadMode should have type OH_NN_INT8.");
            return OH_NN_INVALID_PARAMETER;
        }

        int8_t* pPad = static_cast<int8_t*>(buffer);
        if (!OHOS::NeuralNetworkRuntime::Validation::ValidatePadMode(*pPad)) {
            LOGE("[Conv2d] SetPad failed, invalid pad mode.");
            return OH_NN_INVALID_PARAMETER;
        }
        m_padMode = NNToMS::TransformPadModeValue(*pPad);
    } else {
        if (tensor->GetDataType() != OH_NN_INT64) {
            LOGE("[Conv2d] SetPad failed, the PadList should have type OH_NN_INT64.");
            return OH_NN_INVALID_PARAMETER;
        }

        int64_t* pPadList = static_cast<int64_t*>(buffer);
        int padListSize = tensor->GetElementCount();
        m_pad.assign(pPadList, pPadList + padListSize);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DBuilder::SetGroup(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    // Set Group
    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Conv2d] SetGroup failed, The Group shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[Conv2d] SetGroup failed, The Group should have type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2d] SetGroup GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    m_group = *static_cast<int64_t*>(buffer);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DBuilder::SetActavitation(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[Conv2d] SetActavitation failed, the ActivationType shoule be a scalar");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[Conv2d] SetActavitation failed, the ActivationType should have type OH_NN_INT8.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[Conv2d] SetGroup GetBuffer return nullptr");
        return OH_NN_INVALID_PARAMETER;
    }
    int8_t* pFuseData = static_cast<int8_t*>(buffer);
    if (!OHOS::NeuralNetworkRuntime::Validation::ValidateFuseType(static_cast<OH_NN_FuseType>(*pFuseData))) {
        LOGE("[Conv2d] SetActavitation failed, activation input is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_activationType = NNToMS::TransfromFusionType((OH_NN_FuseType)(*pFuseData));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Conv2DBuilder::Build(const std::vector<uint32_t>& paramsIndex,
    const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex,
    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[Conv2d] Build failed, Conv2D operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = SetInputAndOutput(inputsIndex, outputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        return returnCode;
    }

    returnCode = SetChannel(inputsIndex, allTensors);
    if (returnCode != OH_NN_SUCCESS) {
        return returnCode;
    }

    SetKernelSize(inputsIndex, allTensors);

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_CONV2D_STRIDES:
                returnCode = SetStrides(tensor);
                break;
            case OH_NN_CONV2D_DILATION:
                returnCode = SetDilation(tensor);
                break;
            case OH_NN_CONV2D_PAD_MODE:
            case OH_NN_CONV2D_PAD:
                returnCode = SetPad(tensor);
                break;
            case OH_NN_CONV2D_GROUP:
                returnCode = SetGroup(tensor);
                break;
            case OH_NN_CONV2D_ACTIVATION_TYPE:
                returnCode = SetActavitation(tensor);
                break;
            default:
                LOGE("[Conv2D] Build failed, param invalid, type = %d.", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[Conv2D] Build failed, Passed invalid param.");
            return returnCode;
        }
    }

    // The quantization type of the first output determinies that of the operator.
    SetQuantType(outputsIndex, allTensors);

    m_isBuild = true;
    m_name = OP_NAME;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr Conv2DBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[Conv2d] GetPrimitive failed, Cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    auto primitive = MindIR_Conv2DFusion_CreatePrimitive(m_kernelSize, m_strides,
        m_dilation, m_padMode, m_pad, m_group, m_inChannel, m_outChannel, m_activationType);
    LiteGraphPrimitvePtr  graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(Conv2DBuilder, OH_NN_OPS_CONV2D);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
