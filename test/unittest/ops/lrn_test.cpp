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

#include "ops/lrn_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class LRNBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveDepthRadius(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveAlpha(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveBeta(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveBias(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveNormRegion(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    LRNBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2, 3, 4, 5, 6};
    std::vector<int32_t> m_inputDim {1, 3, 2, 2};
    std::vector<int32_t> m_outputDim {1, 3, 2, 2};
    std::vector<int32_t> m_paramDim {};
};

void LRNBuilderTest::SetUp() {}

void LRNBuilderTest::TearDown() {}

void LRNBuilderTest::SaveDepthRadius(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> depthRadiusTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* depthRadiusValue = new (std::nothrow) int64_t[1] {1};
    EXPECT_NE(nullptr, depthRadiusValue);
    depthRadiusTensor->SetBuffer(depthRadiusValue, sizeof(int64_t));
    m_allTensors.emplace_back(depthRadiusTensor);
}

void LRNBuilderTest::SaveAlpha(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> alphaTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* alphaValue = new (std::nothrow) float[1] {0.0001};
    EXPECT_NE(nullptr, alphaValue);
    alphaTensor->SetBuffer(alphaValue, sizeof(float));
    m_allTensors.emplace_back(alphaTensor);
}

void LRNBuilderTest::SaveBeta(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> betaTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* betaValue = new (std::nothrow) float[1] {0.75};
    EXPECT_NE(nullptr, betaValue);
    betaTensor->SetBuffer(betaValue, sizeof(float));
    m_allTensors.emplace_back(betaTensor);
}

void LRNBuilderTest::SaveBias(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> biasTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* biasValue = new (std::nothrow) float[1] {2.0f};
    EXPECT_NE(nullptr, biasValue);
    biasTensor->SetBuffer(biasValue, sizeof(float));
    m_allTensors.emplace_back(biasTensor);
}

void LRNBuilderTest::SaveNormRegion(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> normRegionTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int32_t* normRegionValue = new (std::nothrow) int32_t[1] {0};
    EXPECT_NE(nullptr, normRegionValue);
    normRegionTensor->SetBuffer(normRegionValue, sizeof(int32_t));
    m_allTensors.emplace_back(normRegionTensor);
}

/**
 * @tc.name: lrn_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: lrn_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: lrn_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_003, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3, 4, 5, 6, 7};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_004, TestSize.Level1)
{
    m_outputs = {1, 2};
    m_params = {3, 4, 5, 6, 7};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid depthRadius's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> depthRadiusTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LRN_DEPTH_RADIUS);
    float* depthRadiusValue = new (std::nothrow) float[1] {1.5};
    depthRadiusTensor->SetBuffer(depthRadiusValue, sizeof(float));
    m_allTensors.emplace_back(depthRadiusTensor);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    depthRadiusTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: lrn_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid alpha's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    std::shared_ptr<NNTensor> alphaTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LRN_ALPHA);
    int64_t* alphaValue = new (std::nothrow) int64_t[1] {0};
    alphaTensor->SetBuffer(alphaValue, sizeof(int64_t));
    m_allTensors.emplace_back(alphaTensor);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    alphaTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: lrn_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid beta's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    std::shared_ptr<NNTensor> betaTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LRN_BETA);
    int64_t* betaValue = new (std::nothrow) int64_t[1] {1};
    betaTensor->SetBuffer(betaValue, sizeof(int64_t));
    m_allTensors.emplace_back(betaTensor);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    betaTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: lrn_build_010
 * @tc.desc: Verify that the build function returns a failed message with invalid bias's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_010, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    std::shared_ptr<NNTensor> biasTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LRN_BIAS);
    int64_t* biasValue = new (std::nothrow) int64_t[2] {2};
    biasTensor->SetBuffer(biasValue, sizeof(int64_t));
    m_allTensors.emplace_back(biasTensor);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    biasTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: lrn_build_011
 * @tc.desc: Verify that the build function returns a failed message with invalid normRegion's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    std::shared_ptr<NNTensor> normRegionTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LRN_NORM_REGION);
    int64_t* normRegionValue = new (std::nothrow) int64_t[1] {0};
    normRegionTensor->SetBuffer(normRegionValue, sizeof(int64_t));
    m_allTensors.emplace_back(normRegionTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    normRegionTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: lrn_build_012
 * @tc.desc: Verify that the build function returns a failed message with passing invalid depthRadius.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_012, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_013
 * @tc.desc: Verify that the build function returns a failed message with passing invalid alpha.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_013, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_014
 * @tc.desc: Verify that the build function returns a failed message with passing invalid beta.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_014, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_015
 * @tc.desc: Verify that the build function returns a failed message with passing invalid bias.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_015, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_016
 * @tc.desc: Verify that the build function returns a failed message with passing invalid normRegion.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_016, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_017
 * @tc.desc: Verify that the build function returns a failed message without set buffer for depthRadius.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_017, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> depthRadiusTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LRN_DEPTH_RADIUS);
    m_allTensors.emplace_back(depthRadiusTensor);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_018
 * @tc.desc: Verify that the build function returns a failed message without set buffer for alpha.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_018, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    std::shared_ptr<NNTensor> alphaTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LRN_ALPHA);
    m_allTensors.emplace_back(alphaTensor);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_019
 * @tc.desc: Verify that the build function returns a failed message without set buffer for beta.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_019, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    std::shared_ptr<NNTensor> betaTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LRN_BETA);
    m_allTensors.emplace_back(betaTensor);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_020
 * @tc.desc: Verify that the build function returns a failed message without set buffer for bias.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_020, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    std::shared_ptr<NNTensor> biasTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LRN_BIAS);
    m_allTensors.emplace_back(biasTensor);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_build_021
 * @tc.desc: Verify that the build function returns a failed message without set buffer for normRegion.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_build_021, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    std::shared_ptr<NNTensor> normRegionTensor = TransToNNTensor(OH_NN_INT32, m_paramDim,
        nullptr, OH_NN_LRN_NORM_REGION);
    m_allTensors.emplace_back(normRegionTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: lrn_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveDepthRadius(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LRN_DEPTH_RADIUS);
    SaveAlpha(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_ALPHA);
    SaveBeta(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BETA);
    SaveBias(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LRN_BIAS);
    SaveNormRegion(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LRN_NORM_REGION);

    int64_t depthRadiusValue {1};
    float alphaValue {0.0001};
    float betaValue {0.75};
    float biasValue {2.0f};
    std::string normRegionValue = "ACROSS_CHANNELS";

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnDepthRadius = mindspore::lite::MindIR_LRN_GetDepthRadius(primitive.get());
    EXPECT_EQ(returnDepthRadius, depthRadiusValue);
    auto returnAlpha = mindspore::lite::MindIR_LRN_GetAlpha(primitive.get());
    EXPECT_EQ(returnAlpha, alphaValue);
    auto returnBeta = mindspore::lite::MindIR_LRN_GetBeta(primitive.get());
    EXPECT_EQ(returnBeta, betaValue);
    auto returnBias = mindspore::lite::MindIR_LRN_GetBias(primitive.get());
    EXPECT_EQ(returnBias, biasValue);
    auto returnNormRegion = mindspore::lite::MindIR_LRN_GetNormRegion(primitive.get());
    EXPECT_EQ(returnNormRegion, normRegionValue);
}

/**
 * @tc.name: lrn_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(LRNBuilderTest, lrn_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}