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

#include "ops/detection_post_process_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class DetectionPostProcessBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SetInputTensor();
    void SetInputSize(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetScale(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetNmsIoUThreshold(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetNmsScoreThreshold(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetMaxDetections(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetDetectionsPerClass(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetMaxClassesPerDetection(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetNumClasses(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetUseRegularNms(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetOutQuantized(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    DetectionPostProcessBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1, 2};
    std::vector<uint32_t> m_outputs {3, 4, 5, 6};
    std::vector<uint32_t> m_params {7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    std::vector<int32_t> m_inputBboxDim {1, 16};
    std::vector<int32_t> m_inputScoresDim {1, 4};
    std::vector<int32_t> m_inputAnchorsDim {1, 2, 8};
    std::vector<int32_t> m_outputDim {2, 3};
    std::vector<int32_t> m_paramDim {};
    std::vector<int32_t> m_scaleDim {4};
};

void DetectionPostProcessBuilderTest::SetUp() {}

void DetectionPostProcessBuilderTest::TearDown() {}

void DetectionPostProcessBuilderTest::SetInputSize(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> inputSizeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* inputSizeValue = new (std::nothrow) int64_t[1] {300};
    EXPECT_NE(nullptr, inputSizeValue);
    inputSizeTensor->SetBuffer(inputSizeValue, sizeof(int64_t));
    m_allTensors.emplace_back(inputSizeTensor);
}

void DetectionPostProcessBuilderTest::SetScale(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* scaleValue = new (std::nothrow) float[4] {10.0, 10.0, 5.0, 5.0};
    int32_t scaleSize = 4;
    EXPECT_NE(nullptr, scaleValue);
    scaleTensor->SetBuffer(scaleValue, sizeof(float) * scaleSize);
    m_allTensors.emplace_back(scaleTensor);
}

void DetectionPostProcessBuilderTest::SetNmsIoUThreshold(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> nmsIouThresholdTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* nmsIouThresholdValue = new (std::nothrow) float[1] {0.5};
    EXPECT_NE(nullptr, nmsIouThresholdValue);
    nmsIouThresholdTensor->SetBuffer(nmsIouThresholdValue, sizeof(float));
    m_allTensors.emplace_back(nmsIouThresholdTensor);
}

void DetectionPostProcessBuilderTest::SetNmsScoreThreshold(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> nmsScoreThresholdTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* nmsScoreThresholdValue = new (std::nothrow) float[1] {0.5};
    EXPECT_NE(nullptr, nmsScoreThresholdValue);
    nmsScoreThresholdTensor->SetBuffer(nmsScoreThresholdValue, sizeof(float));
    m_allTensors.emplace_back(nmsScoreThresholdTensor);
}

void DetectionPostProcessBuilderTest::SetMaxDetections(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> maxDetectionsTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* maxDetectionsValue = new (std::nothrow) int64_t[1] {5};
    EXPECT_NE(nullptr, maxDetectionsValue);
    maxDetectionsTensor->SetBuffer(maxDetectionsValue, sizeof(int64_t));
    m_allTensors.emplace_back(maxDetectionsTensor);
}

void DetectionPostProcessBuilderTest::SetDetectionsPerClass(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> detectionsPerClassTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* detectionsPerClassValue = new (std::nothrow) int64_t[1] {2};
    EXPECT_NE(nullptr, detectionsPerClassValue);
    detectionsPerClassTensor->SetBuffer(detectionsPerClassValue, sizeof(int64_t));
    m_allTensors.emplace_back(detectionsPerClassTensor);
}

void DetectionPostProcessBuilderTest::SetMaxClassesPerDetection(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> maxClassesPerDetectionTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* maxClassesPerDetectionValue = new (std::nothrow) int64_t[1] {1};
    EXPECT_NE(nullptr, maxClassesPerDetectionValue);
    maxClassesPerDetectionTensor->SetBuffer(maxClassesPerDetectionValue, sizeof(int64_t));
    m_allTensors.emplace_back(maxClassesPerDetectionTensor);
}

void DetectionPostProcessBuilderTest::SetNumClasses(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> numClassesTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* numClassesValue = new (std::nothrow) int64_t[1] {5};
    EXPECT_NE(nullptr, numClassesValue);
    numClassesTensor->SetBuffer(numClassesValue, sizeof(int64_t));
    m_allTensors.emplace_back(numClassesTensor);
}

void DetectionPostProcessBuilderTest::SetUseRegularNms(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> useRegularNmsTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool* useRegularNmsValue = new (std::nothrow) bool(false);
    EXPECT_NE(nullptr, useRegularNmsValue);
    useRegularNmsTensor->SetBuffer(useRegularNmsValue, sizeof(bool));
    m_allTensors.emplace_back(useRegularNmsTensor);
}

void DetectionPostProcessBuilderTest::SetOutQuantized(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> outQuantizedTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool* outQuantizedValue = new (std::nothrow) bool(false);
    EXPECT_NE(nullptr, outQuantizedValue);
    outQuantizedTensor->SetBuffer(outQuantizedValue, sizeof(bool));
    m_allTensors.emplace_back(outQuantizedTensor);
}

void DetectionPostProcessBuilderTest::SetInputTensor()
{
    m_inputsIndex = m_inputs;
    std::shared_ptr<NNTensor> bboxTensor;
    bboxTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputBboxDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(bboxTensor);

    std::shared_ptr<NNTensor> scoresTensor;
    scoresTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputScoresDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(scoresTensor);

    std::shared_ptr<NNTensor> anchorsTensor;
    anchorsTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputAnchorsDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(anchorsTensor);
}

/**
 * @tc.name: detection_post_process_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_001, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: detection_post_process_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_002, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: detection_post_process_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_003, TestSize.Level1)
{
    m_inputs = {0, 1, 2, 3, 4};
    m_outputs = {5, 6, 7, 8};
    m_params = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    SaveInputTensor({1}, OH_NN_FLOAT32, m_inputBboxDim, nullptr);
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_004, TestSize.Level1)
{
    m_outputs = {4, 5, 6, 7, 8};
    m_params = {9, 10, 11, 12, 13, 14, 15, 16, 17, 18};

    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_006, TestSize.Level1)
{
    SetInputTensor();

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid inputSize's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_007, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> inputSizeTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    float* inputSizeValue = new (std::nothrow) float[1] {300.0f};
    inputSizeTensor->SetBuffer(inputSizeValue, sizeof(float));
    m_allTensors.emplace_back(inputSizeTensor);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    inputSizeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid scale's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_008, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    int64_t* scaleValue = new (std::nothrow) int64_t[4] {10.0f, 10.0f, 5.0f, 5.0f};
    int32_t scaleSize = 4;
    scaleTensor->SetBuffer(scaleValue, sizeof(int64_t) * scaleSize);
    m_allTensors.emplace_back(scaleTensor);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    scaleTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid nmsIoUThreshold's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_009, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    std::shared_ptr<NNTensor> nmsIoUThresholdTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    int64_t* nmsIoUThresholdValue = new (std::nothrow) int64_t[1] {0};
    nmsIoUThresholdTensor->SetBuffer(nmsIoUThresholdValue, sizeof(int64_t));
    m_allTensors.emplace_back(nmsIoUThresholdTensor);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    nmsIoUThresholdTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_010
 * @tc.desc: Verify that the build function returns a failed message with invalid nmsScoreThreshold's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_010, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    std::shared_ptr<NNTensor> nmsScoreThresholdTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    int64_t* nmsScoreThresholdValue = new (std::nothrow) int64_t[1] {0};
    nmsScoreThresholdTensor->SetBuffer(nmsScoreThresholdValue, sizeof(int64_t));
    m_allTensors.emplace_back(nmsScoreThresholdTensor);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    nmsScoreThresholdTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_011
 * @tc.desc: Verify that the build function returns a failed message with invalid maxDetections's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_011, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    std::shared_ptr<NNTensor> maxDetectionsTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    float* maxDetectionsValue = new (std::nothrow) float[1] {5.0f};
    maxDetectionsTensor->SetBuffer(maxDetectionsValue, sizeof(float));
    m_allTensors.emplace_back(maxDetectionsTensor);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    maxDetectionsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_012
 * @tc.desc: Verify that the build function returns a failed message with invalid detectionsPerClass's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_012, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    std::shared_ptr<NNTensor> detectionsPerClassTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    float* detectionsPerClassValue = new (std::nothrow) float[1] {2.0f};
    detectionsPerClassTensor->SetBuffer(detectionsPerClassValue, sizeof(float));
    m_allTensors.emplace_back(detectionsPerClassTensor);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    detectionsPerClassTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_013
 * @tc.desc: Verify that the build function returns a failed message with invalid maxClassesPerDetection's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_013, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    std::shared_ptr<NNTensor> maxClassesPerDetectionTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    float* maxClassesPerDetectionValue = new (std::nothrow) float[2] {1.0f};
    maxClassesPerDetectionTensor->SetBuffer(maxClassesPerDetectionValue, sizeof(float));
    m_allTensors.emplace_back(maxClassesPerDetectionTensor);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    maxClassesPerDetectionTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_014
 * @tc.desc: Verify that the build function returns a failed message with invalid numClasses's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_014, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    std::shared_ptr<NNTensor> numClassesTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    float* numClassesValue = new (std::nothrow) float[1] {5.0f};
    numClassesTensor->SetBuffer(numClassesValue, sizeof(float));
    m_allTensors.emplace_back(numClassesTensor);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    numClassesTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_015
 * @tc.desc: Verify that the build function returns a failed message with invalid useRegularNms's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_015, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    std::shared_ptr<NNTensor> useRegularNmsTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    int64_t* useRegularNmsValue = new (std::nothrow) int64_t[1] {0};
    useRegularNmsTensor->SetBuffer(useRegularNmsValue, sizeof(int64_t));
    m_allTensors.emplace_back(useRegularNmsTensor);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    useRegularNmsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_016
 * @tc.desc: Verify that the build function returns a failed message with invalid outQuantized's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_016, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    std::shared_ptr<NNTensor> outQuantizedTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);
    int64_t* outQuantizedValue = new (std::nothrow) int64_t[1] {0};
    outQuantizedTensor->SetBuffer(outQuantizedValue, sizeof(int64_t));
    m_allTensors.emplace_back(outQuantizedTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    outQuantizedTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: detection_post_process_build_017
 * @tc.desc: Verify that the build function returns a failed message with passing invalid inputSize.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_017, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_018
 * @tc.desc: Verify that the build function returns a failed message with passing invalid scale.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_018, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_019
 * @tc.desc: Verify that the build function returns a failed message with passing invalid nmsIoUThreshold.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_019, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_020
 * @tc.desc: Verify that the build function returns a failed message with passing invalid nmsScoreThreshold.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_020, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_021
 * @tc.desc: Verify that the build function returns a failed message with passing invalid maxDetections.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_021, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_022
 * @tc.desc: Verify that the build function returns a failed message with passing invalid detectionsPerClass.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_022, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_023
 * @tc.desc: Verify that the build function returns a failed message with passing invalid maxClassesPerDetection.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_023, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_024
 * @tc.desc: Verify that the build function returns a failed message with passing invalid numClasses.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_024, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_025
 * @tc.desc: Verify that the build function returns a failed message with passing invalid useRegularNms.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_025, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_026
 * @tc.desc: Verify that the build function returns a failed message with passing invalid outQuantized.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_026, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_027
 * @tc.desc: Verify that the build function returns a failed message without set buffer for InputSize.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_027, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> inputSizeTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    m_allTensors.emplace_back(inputSizeTensor);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_028
 * @tc.desc: Verify that the build function returns a failed message without set buffer for scale.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_028, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    m_allTensors.emplace_back(scaleTensor);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_029
 * @tc.desc: Verify that the build function returns a failed message without set buffer for nmsIoUThreshold.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_029, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    std::shared_ptr<NNTensor> nmsIoUThresholdTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    m_allTensors.emplace_back(nmsIoUThresholdTensor);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_030
 * @tc.desc: Verify that the build function returns a failed message without set buffer for nmsScoreThreshold.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_030, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    std::shared_ptr<NNTensor> nmsScoreThresholdTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    m_allTensors.emplace_back(nmsScoreThresholdTensor);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_031
 * @tc.desc: Verify that the build function returns a failed message without set buffer for maxDetections.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_031, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    std::shared_ptr<NNTensor> maxDetectionsTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    m_allTensors.emplace_back(maxDetectionsTensor);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_032
 * @tc.desc: Verify that the build function returns a failed message without set buffer for detectionsPerClass.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_032, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    std::shared_ptr<NNTensor> detectionsPerClassTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    m_allTensors.emplace_back(detectionsPerClassTensor);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_033
 * @tc.desc: Verify that the build function returns a failed message without set buffer for maxClassesPerDetection.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_033, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    std::shared_ptr<NNTensor> maxClassesPerDetectionTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    m_allTensors.emplace_back(maxClassesPerDetectionTensor);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_034
 * @tc.desc: Verify that the build function returns a failed message without set buffer for numClasses.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_034, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    std::shared_ptr<NNTensor> numClassesTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    m_allTensors.emplace_back(numClassesTensor);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_035
 * @tc.desc: Verify that the build function returns a failed message without set buffer for useRegularNms.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_035, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    std::shared_ptr<NNTensor> useRegularNmsTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    m_allTensors.emplace_back(useRegularNmsTensor);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_build_036
 * @tc.desc: Verify that the build function returns a failed message without set buffer for outQuantized.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_build_036, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    std::shared_ptr<NNTensor> outQuantizedTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);
    m_allTensors.emplace_back(outQuantizedTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: detection_post_process_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_getprimitive_001, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_INPUT_SIZE);
    SetScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_SCALE);
    SetNmsIoUThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_IOU_THRESHOLD);
    SetNmsScoreThreshold(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NMS_SCORE_THRESHOLD);
    SetMaxDetections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_DETECTIONS);
    SetDetectionsPerClass(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_DETECTIONS_PER_CLASS);
    SetMaxClassesPerDetection(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_MAX_CLASSES_PER_DETECTION);
    SetNumClasses(OH_NN_INT64, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_NUM_CLASSES);
    SetUseRegularNms(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_USE_REGULAR_NMS);
    SetOutQuantized(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_DETECTION_POST_PROCESS_OUT_QUANTIZED);

    int64_t inputSizeValue = 300;
    std::vector<float> scaleValue = {10.0f, 10.0f, 5.0f, 5.0f};
    float nmsIoUThresholdValue = 0.5f;
    float nmsScoreThresholdValue = 0.5f;
    int64_t maxDetectionsValue = 5;
    int64_t detectionsPerClassValue = 2;
    int64_t maxClassesPerDetectionValue = 1;
    int64_t numClassesValue = 5;
    bool useRegularNmsValue = false;
    bool outQuantizedValue = false;

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnInputSize = mindspore::lite::MindIR_DetectionPostProcess_GetInputSize(primitive.get());
    EXPECT_EQ(returnInputSize, inputSizeValue);
    auto returnScale = mindspore::lite::MindIR_DetectionPostProcess_GetScale(primitive.get());
    auto returnScaleSize = returnScale.size();
    for (size_t i = 0; i < returnScaleSize; ++i) {
        EXPECT_EQ(returnScale[i], scaleValue[i]);
    }
    auto returnNmsIoUThresholdValue = mindspore::lite::MindIR_DetectionPostProcess_GetNmsIouThreshold(primitive.get());
    EXPECT_EQ(returnNmsIoUThresholdValue, nmsIoUThresholdValue);
    auto returnNmsScoreThreshold = mindspore::lite::MindIR_DetectionPostProcess_GetNmsScoreThreshold(primitive.get());
    EXPECT_EQ(returnNmsScoreThreshold, nmsScoreThresholdValue);
    auto returnMaxDetections = mindspore::lite::MindIR_DetectionPostProcess_GetMaxDetections(primitive.get());
    EXPECT_EQ(returnMaxDetections, maxDetectionsValue);
    auto returnDetectionsPerClass =
        mindspore::lite::MindIR_DetectionPostProcess_GetDetectionsPerClass(primitive.get());
    EXPECT_EQ(returnDetectionsPerClass, detectionsPerClassValue);
    auto returnMaxClassesPerDetection =
        mindspore::lite::MindIR_DetectionPostProcess_GetMaxClassesPerDetection(primitive.get());
    EXPECT_EQ(returnMaxClassesPerDetection, maxClassesPerDetectionValue);
    auto returnNumClasses = mindspore::lite::MindIR_DetectionPostProcess_GetNumClasses(primitive.get());
    EXPECT_EQ(returnNumClasses, numClassesValue);
    auto returnUseRegularNms = mindspore::lite::MindIR_DetectionPostProcess_GetUseRegularNms(primitive.get());
    EXPECT_EQ(returnUseRegularNms, useRegularNmsValue);
    auto returnOutQuantized = mindspore::lite::MindIR_DetectionPostProcess_GetOutQuantized(primitive.get());
    EXPECT_EQ(returnOutQuantized, outQuantizedValue);
}

/**
 * @tc.name: detection_post_process_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(DetectionPostProcessBuilderTest, detection_post_process_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}