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

#include "ops/leaky_relu_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class LeakyReluBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveNegativeSlope(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    LeakyReluBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2};
    std::vector<int32_t> m_dim {1, 2, 2, 1};
    std::vector<int32_t> m_paramDim {};
};

void LeakyReluBuilderTest::SetUp() {}

void LeakyReluBuilderTest::TearDown() {}

void LeakyReluBuilderTest::SaveNegativeSlope(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> negativeSlopeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* negativeSlopeValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, negativeSlopeValue);
    negativeSlopeTensor->SetBuffer(negativeSlopeValue, sizeof(float));
    m_allTensors.emplace_back(negativeSlopeTensor);
}

/**
 * @tc.name: leaky_relu_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveNegativeSlope(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LEAKY_RELU_NEGATIVE_SLOPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: leaky_relu_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveNegativeSlope(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LEAKY_RELU_NEGATIVE_SLOPE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: leaky_relu_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_003, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveNegativeSlope(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LEAKY_RELU_NEGATIVE_SLOPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: leaky_relu_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_004, TestSize.Level1)
{
    m_outputs = {1, 2};
    m_params = {3};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveNegativeSlope(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LEAKY_RELU_NEGATIVE_SLOPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: leaky_relu_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: leaky_relu_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: leaky_relu_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid negative_slope's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> negativeSlopeTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LEAKY_RELU_NEGATIVE_SLOPE);
    int64_t* negativeSlopeValue = new (std::nothrow) int64_t [1]{0};
    EXPECT_NE(nullptr, negativeSlopeValue);
    negativeSlopeTensor->SetBuffer(negativeSlopeValue, sizeof(int64_t));
    m_allTensors.emplace_back(negativeSlopeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    negativeSlopeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: leaky_relu_build_008
 * @tc.desc: Verify that the build function returns a failed message with passing invalid negative_slope param.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveNegativeSlope(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: leaky_relu_build_009
 * @tc.desc: Verify that the build function returns a failed message without set buffer for negative_slope.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> negativeSlopeTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LEAKY_RELU_NEGATIVE_SLOPE);
    m_allTensors.emplace_back(negativeSlopeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: leaky_relu_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveNegativeSlope(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LEAKY_RELU_NEGATIVE_SLOPE);

    float negativeSlopeValue = 0.0f;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnNegativeSlopeValue = mindspore::lite::MindIR_LeakyRelu_GetNegativeSlope(primitive.get());
    EXPECT_EQ(returnNegativeSlopeValue, negativeSlopeValue);
}

/**
 * @tc.name: leaky_relu_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(LeakyReluBuilderTest, leaky_relu_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}