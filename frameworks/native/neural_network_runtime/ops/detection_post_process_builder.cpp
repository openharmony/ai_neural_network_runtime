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

#include "detection_post_process_builder.h"

#include "transform.h"
#include "validation.h"
#include "mindir.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
static const int INPUT_NUM = 3;
static const int OUTPUT_NUM = 4;
static const int PARAM_MAX_NUM = 10;
static const int SCALAR_LENGTH = 1;
static const std::string OP_NAME = "DetectionPostProcess";

DetectionPostProcessBuilder::DetectionPostProcessBuilder() {}

DetectionPostProcessBuilder::~DetectionPostProcessBuilder() {}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetInputSize(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[DetectionPostProcess] The inputSize should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The inputSize should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_inputSize = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetScale(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[DetectionPostProcess] The scale should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_scale.clear();

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    float* pScale = static_cast<float*>(buffer);

    uint32_t elementCount = tensor->GetElementCount();
    for (uint32_t i = 0; i < elementCount; ++i) {
        m_scale.emplace_back(*pScale);
        ++pScale;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetNmsIoUThreshold(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[DetectionPostProcess] The nmsIoUThreshold should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The nmsIoUThreshold should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_nmsIoUThreshold = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetNmsScoreThreshold(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_FLOAT32) {
        LOGE("[DetectionPostProcess] The scoreThreshold should be type OH_NN_FLOAT32.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The scoreThreshold should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_nmsScoreThreshold = *(static_cast<const float*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetMaxDetections(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[DetectionPostProcess] The maxDetections should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The maxDetections should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_maxDetections = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetDetectionsPerClass(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[DetectionPostProcess] The detectionsPerClass should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The detectionsPerClass should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_detectionsPerClass = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetMaxClassesPerDetection(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[DetectionPostProcess] The maxClassesPerDetection should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The maxClassesPerDetection should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_maxClassesPerDetection = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetNumClasses(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_INT64) {
        LOGE("[DetectionPostProcess] The numClasses should be type OH_NN_INT64.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The numClasses should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_numClasses = *(static_cast<const int64_t*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetUseRegularNms(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[DetectionPostProcess] The useRegularNms should be type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The useRegularNms should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_useRegularNms = *(static_cast<bool*>(buffer));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DetectionPostProcessBuilder::SetOutQuantized(const std::shared_ptr<NNTensor>& tensor)
{
    if (tensor->GetDataType() != OH_NN_BOOL) {
        LOGE("[DetectionPostProcess] The outQuantized should be type OH_NN_BOOL.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor->GetElementCount() != SCALAR_LENGTH) {
        LOGE("[DetectionPostProcess] The outQuantized should be scalar.");
        return OH_NN_INVALID_PARAMETER;
    }

    void* buffer = tensor->GetBuffer();
    if (buffer == nullptr) {
        LOGE("[DetectionPostProcess] Tensor buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_outQuantized = *(static_cast<bool*>(buffer));

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode DetectionPostProcessBuilder::Build(const std::vector<uint32_t>& paramsIndex,
                                                    const std::vector<uint32_t>& inputsIndex,
                                                    const std::vector<uint32_t>& outputsIndex,
                                                    const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (m_isBuild) {
        LOGE("[DetectionPostProcess] Build failed, the detectionPostProcess operation has been build. \
             cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto ret = CheckIOIndex(inputsIndex, outputsIndex, allTensors, INPUT_NUM, OUTPUT_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[DetectionPostProcess] Build failed, passed invalid input or output index.");
        return ret;
    }

    m_inputsIndex = inputsIndex;
    m_outputsIndex = outputsIndex;

    ret = CheckParamIndex(paramsIndex, allTensors, PARAM_MAX_NUM);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[DetectionPostProcess] Build failed, passed invalid param index.");
        return ret;
    }

    for (int i : paramsIndex) {
        std::shared_ptr<NNTensor> tensor = allTensors[i];
        tensor->IdentifyOpParameter();
        if (m_paramMap.find(tensor->GetType()) != m_paramMap.end()) {
            ret = (this->*(m_paramMap[tensor->GetType()]))(tensor);
        } else {
            LOGE("[DetectionPostProcess] Build failed, param invalid, type=%d", tensor->GetType());
            return OH_NN_INVALID_PARAMETER;
        }

        if (ret != OH_NN_SUCCESS) {
            LOGE("[DetectionPostProcess] Build failed, passed invalid param.");
            return ret;
        }
    }
    
    m_name = OP_NAME;
    m_isBuild = true;
    return OH_NN_SUCCESS;
}

LiteGraphPrimitvePtr DetectionPostProcessBuilder::GetPrimitive()
{
    if (!m_isBuild) {
        LOGE("[DetectionPostProcess] GetPrimitive failed, cannot get primitive before call build.");
        return {nullptr, DestroyLiteGraphPrimitive};
    }

    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};

    void* primitive = mindspore::lite::MindIR_DetectionPostProcess_CreatePrimitive(format, m_inputSize, m_scale,
        m_nmsIoUThreshold, m_nmsScoreThreshold, m_maxDetections, m_detectionsPerClass, m_maxClassesPerDetection,
        m_numClasses, m_useRegularNms, m_outQuantized);
    LiteGraphPrimitvePtr graphPrimitivePtr(primitive, DestroyLiteGraphPrimitive) ;
    return graphPrimitivePtr;
}

REGISTER_OPS(DetectionPostProcessBuilder, OH_NN_OPS_DETECTION_POST_PROCESS);
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS