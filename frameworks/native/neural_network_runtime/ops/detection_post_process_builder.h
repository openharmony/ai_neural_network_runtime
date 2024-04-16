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

#ifndef NEURAL_NETWORK_RUNTIME_DETECTION_POST_PROCESS_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_DETECTION_POST_PROCESS_BUILDER_H

#include "ops_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class DetectionPostProcessBuilder : public OpsBuilder {
public:
    typedef DetectionPostProcessBuilder DPPBuilder;
    typedef OH_NN_ReturnCode (DPPBuilder::*FuncPtr)(const std::shared_ptr<NNTensor>&);

    DetectionPostProcessBuilder();
    ~DetectionPostProcessBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;

    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetInputSize(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetScale(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetNmsIoUThreshold(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetNmsScoreThreshold(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetMaxDetections(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetDetectionsPerClass(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetMaxClassesPerDetection(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetNumClasses(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetUseRegularNms(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetOutQuantized(const std::shared_ptr<NNTensor>& tensor);

private:
    int64_t m_inputSize {0};
    std::vector<float> m_scale;
    float m_nmsIoUThreshold {0.0f};
    float m_nmsScoreThreshold {0.0f};
    int64_t m_maxDetections {0};
    int64_t m_detectionsPerClass {0};
    int64_t m_maxClassesPerDetection {0};
    int64_t m_numClasses {0};
    bool m_useRegularNms {false};
    bool m_outQuantized {false};
    std::unordered_map<OH_NN_TensorType, FuncPtr> m_paramMap = {
        {OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE, &DPPBuilder::SetInputSize},
        {OH_NN_DETECTION_POST_PROCESS_SCALE, &DPPBuilder::SetScale},
        {OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD, &DPPBuilder::SetNmsIoUThreshold},
        {OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD, &DPPBuilder::SetNmsScoreThreshold},
        {OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS, &DPPBuilder::SetMaxDetections},
        {OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS, &DPPBuilder::SetDetectionsPerClass},
        {OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION, &DPPBuilder::SetMaxClassesPerDetection},
        {OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES, &DPPBuilder::SetNumClasses},
        {OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS, &DPPBuilder::SetUseRegularNms},
        {OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED, &DPPBuilder::SetOutQuantized}
    };
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_DETECTION_POST_PROCESS_BUILDER_H
