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

#include "resize_bilinear_builder.h"

#include "frameworks/native/ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 1;
static const int OUTPUT_NUM = 1;
static const int SCALE_LENGTH = 1;
static const std::string OP_NAME = "ResizeBilinear";

ResizeBilinearBuilder::ResizeBilinearBuilder() {}

ResizeBilinearBuilder::~ResizeBilinearBuilder() {}

OH_NN_ReturnCode ResizeBilinearBuilder::SetNewHeight(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[ResizeBilinear] SetNewHeight failed, the new_height dimensions should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[ResizeBilinear] SetNewHeight failed, the new_height should be type OH_NN_INT64");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ResizeBilinear] ResizeBilinear failed, the new_height passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_newHeight = *(static_cast<uint64_t *>(buffer));
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ResizeBilinearBuilder::SetNewWidth(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[ResizeBilinear] SetNewWidth failed, the new_width dimensions should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[ResizeBilinear] SetNewWidth failed, the new_width should be type OH_NN_INT64");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ResizeBilinear] SetNewWidth failed, the new_width passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_newWidth = *(static_cast<uint64_t *>(buffer));
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ResizeBilinearBuilder::SetPreserveAspectRatio(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[ResizeBilinear] SetPreserveAspectRatio failed, the preserve_aspect_ratio dimensions should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[ResizeBilinear] SetPreserveAspectRatio failed, the preserve_aspect_ratio should be type OH_NN_BOOL");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ResizeBilinear] SetPreserveAspectRatio failed, the preserve_aspect_ratio passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_preserveAspectRatio = *(static_cast<bool *>(buffer));
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ResizeBilinearBuilder::SetCoordinateTransformMode(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[ResizeBilinear] SetCoordinateTransformMode failed,"
            "the coordinate_transform_mode dimensions should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT8) {
        LOGE("[ResizeBilinear] SetCoordinateTransformMode failed,"
            "the coordinate_transform_mode should be type OH_NN_INT32");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ResizeBilinear] SetCoordinateTransformMode failed,"
            "the coordinate_transform_mode passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_coordinateTransformMode = *(static_cast<mindspore::lite::CoordinateTransformMode *>(buffer));
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ResizeBilinearBuilder::SetExcludeOutside(std::shared_ptr<NNTensor> tensor)
{
    tensor->IdentifyOpParameter();
    if (tensor->GetElementCount() != SCALE_LENGTH) {
        LOGE("[ResizeBilinear] SetExcludeOutside failed, the exclude_outside dimensions should be scaler.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[ResizeBilinear] SetExcludeOutside failed, the exclude_outside should be type OH_NN_INT64");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[ResizeBilinear] SetExcludeOutside failed, the exclude_outside passed buffer is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_excludeOutside = *(static_cast<uint64_t *>(buffer));
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ResizeBilinearBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                              const std::vector<uint32_t>& inputsIndex,
                                              const std::vector<uint32_t>& outputsIndex,
                                              const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[ResizeBilinear] Build failed, the Resize operation has been build, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode returnCode = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("[ResizeBilinear] Build failed, passed invalid input or output index.");
        return returnCode;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    for (uint32_t i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        switch (tensor->GetType()) {
            case OH_NN_RESIZE_BILINEAR_NEW_HEIGHT:
                returnCode = SetNewHeight(tensor);
                break;
            case OH_NN_RESIZE_BILINEAR_NEW_WIDTH:
                returnCode = SetNewWidth(tensor);
                break;
            case OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO:
                returnCode = SetPreserveAspectRatio(tensor);
                break;
            case OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE:
                returnCode = SetCoordinateTransformMode(tensor);
                break;
            case OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE:
                returnCode = SetExcludeOutside(tensor);
                break;
            default:
                LOGE("[ResizeBilinear] Build failed, parameter type is invalid. type=%d", tensor->GetType());
                return OH_NN_INVALID_PARAMETER;
        }

        if (returnCode != OH_NN_SUCCESS) {
            LOGE("[ResizeBilinear] Build failed, passed invalid param.");
            return returnCode;
        }
    }

    SetQuantType(outputsIndex, allTensors);

    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr ResizeBilinearBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[ResizeBilinear] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    float cubicCoeff{0.0f};
    float extrapolationValue{0.0f};
    mindspore::lite::NearestMode nearestMode{mindspore::lite::NEAREST_MODE_NORMAL};

    void* primitive = mindspore::lite::MindIR_Resize_CreatePrimitive(m_method, m_newHeight, m_newWidth,
        m_preserveAspectRatio, m_coordinateTransformMode, cubicCoeff, m_excludeOutside,
        extrapolationValue, nearestMode);

    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive);
    return graphPrimitivePtr;
}

REGISTER_OPS(ResizeBilinearBuilder, OH_NN_OPS_RESIZE_BILINEAR);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS