/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#include "ops/l2_normalize_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class L2NormalizeBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SetAxis(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetEpsilon(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    L2NormalizeBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2, 3, 4};
    std::vector<int32_t> m_inputDim {2, 3};
    std::vector<int32_t> m_outputDim {2, 3};
    std::vector<int32_t> m_paramDim {1};
};

void L2NormalizeBuilderTest::SetUp() {}

void L2NormalizeBuilderTest::TearDown() {}

void L2NormalizeBuilderTest::SetAxis(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axisValue = new (std::nothrow) int64_t[1] {1};
    EXPECT_NE(nullptr, axisValue);
    axisTensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(axisTensor);
}

void L2NormalizeBuilderTest::SetEpsilon(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* epsilonValue = new (std::nothrow) float[1] {0.0f};
    EXPECT_NE(nullptr, epsilonValue);
    epsilonTensor->SetBuffer(epsilonValue, sizeof(float));
    m_allTensors.emplace_back(epsilonTensor);
}

/**
 * @tc.name: l2_normalize_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: l2_normalize_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: l2_normalize_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_003, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3, 4, 5};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_004, TestSize.Level1)
{
    m_outputs = {1, 2};
    m_params = {3, 4, 5};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid axis's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_L2_NORMALIZE_AXIS);
    float* axisValue = new (std::nothrow) float[1] {1.0f};
    axisTensor->SetBuffer(axisValue, sizeof(float));
    m_allTensors.emplace_back(axisTensor);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    axisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: l2_normalize_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid epsilon's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    int64_t* epsilonValue = new (std::nothrow) int64_t[1] {0};
    epsilonTensor->SetBuffer(epsilonValue, sizeof(int64_t));
    m_allTensors.emplace_back(epsilonTensor);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    epsilonTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: l2_normalize_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid activationType's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    std::shared_ptr<NNTensor> activationTypeTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);
    float* activationTypeValue = new (std::nothrow) float[1] {0.0f};
    activationTypeTensor->SetBuffer(activationTypeValue, sizeof(float));
    m_allTensors.emplace_back(activationTypeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    activationTypeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: l2_normalize_build_010
 * @tc.desc: Verify that the build function returns a failed message with passing invalid axis.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_010, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_011
 * @tc.desc: Verify that the build function returns a failed message with passing invalid epsilon.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_012
 * @tc.desc: Verify that the build function returns a failed message with passing invalid activationType.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_012, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_013
 * @tc.desc: Verify that the build function returns a failed message without set buffer for axis.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_013, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_L2_NORMALIZE_AXIS);
    m_allTensors.emplace_back(axisTensor);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_014
 * @tc.desc: Verify that the build function returns a failed message without set buffer for epsilon.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_014, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    m_allTensors.emplace_back(epsilonTensor);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_build_015
 * @tc.desc: Verify that the build function returns a failed message without set buffer for activationType.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_build_015, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    std::shared_ptr<NNTensor> shapeTensor = TransToNNTensor(OH_NN_INT8, m_paramDim,
        nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);
    m_allTensors.emplace_back(shapeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: l2_normalize_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetAxis(OH_NN_INT64, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_AXIS);
    SetEpsilon(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_EPSILON);
    SetActivation(OH_NN_INT8, m_paramDim, nullptr, OH_NN_L2_NORMALIZE_ACTIVATION_TYPE);

    std::vector<int64_t> axisValue {1};
    float epsilonValue {0.0f};
    mindspore::lite::ActivationType activationTypeValue {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnAxis = mindspore::lite::MindIR_L2NormalizeFusion_GetAxis(primitive.get());
    auto returnAxisSize = returnAxis.size();
    for (size_t i = 0; i < returnAxisSize; ++i) {
        EXPECT_EQ(returnAxis[i], axisValue[i]);
    }
    auto returnEpsilon = mindspore::lite::MindIR_L2NormalizeFusion_GetEpsilon(primitive.get());
    EXPECT_EQ(returnEpsilon, epsilonValue);
    mindspore::lite::ActivationType returnActivationValue =
        mindspore::lite::MindIR_L2NormalizeFusion_GetActivationType(primitive.get());
    EXPECT_EQ(returnActivationValue, activationTypeValue);
}

/**
 * @tc.name: l2_normalize_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(L2NormalizeBuilderTest, l2_normalize_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}