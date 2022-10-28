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

#include "frameworks/native/ops/scale_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ScaleBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveAxisTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveActivationTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    ScaleBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1, 2};
    std::vector<uint32_t> m_outputs {3};
    std::vector<uint32_t> m_params {4, 5};
    std::vector<int32_t> m_dim {1, 4, 1, 1};
    std::vector<int32_t> m_paramDim {};
};

void ScaleBuilderTest::SetUp() {}

void ScaleBuilderTest::TearDown() {}

void ScaleBuilderTest::SaveAxisTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t *axisValue = new (std::nothrow) int64_t(1);
    EXPECT_NE(nullptr, axisValue);
    axisTensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(axisTensor);
}

void ScaleBuilderTest::SaveActivationTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> activationTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t *activationValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValue);
    activationTensor->SetBuffer(activationValue, sizeof(int64_t));
    m_allTensors.emplace_back(activationTensor);
}

/**
 * @tc.name: scale_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: scale_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: scale_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2, 3};
    m_outputs = {4};
    m_params = {5, 6};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: scale_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_004, TestSize.Level0)
{
    m_outputs = {3, 4};
    m_params = {5, 6};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: scale_build_005
 * @tc.desc: Verify that the build function return a failed message with null allTensor
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: scale_build_006
 * @tc.desc: Verify that the build function return a failed message without output tensor
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: scale_build_007
 * @tc.desc: Verify that the build function return a failed message with invalided axis's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_007, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);

    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_SCALE_AXIS);
    int32_t axisValue = 1;
    axisTensor->SetBuffer(&axisValue, sizeof(axisValue));
    m_allTensors.emplace_back(axisTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    axisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: scale_build_008
 * @tc.desc: Verify that the build function return a failed message with invalided activation's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_008, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);

    std::shared_ptr<NNTensor> activationTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_SCALE_ACTIVATIONTYPE);
    int64_t activationValue = 0;
    activationTensor->SetBuffer(&activationValue, sizeof(activationValue));
    m_allTensors.emplace_back(activationTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    activationTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: scale_build_009
 * @tc.desc: Verify that the build function return a failed message with invalided axis's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_009, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    std::vector<int32_t> axistDim = {2};
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_INT64, axistDim, nullptr, OH_NN_SCALE_AXIS);
    int64_t axisValue[2] = {1, 1};
    axisTensor->SetBuffer(axisValue, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(axisTensor);

    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    axisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: scale_build_010
 * @tc.desc: Verify that the build function return a failed message with invalided activation's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_010, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);

    std::vector<int32_t> activationDim = {2};
    std::shared_ptr<NNTensor> activationTensor = TransToNNTensor(OH_NN_INT8,
        activationDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);
    int64_t activationValue[2] = {1, 1};
    activationTensor->SetBuffer(activationValue, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(activationTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    activationTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: scale_build_011
 * @tc.desc: Verify that the build function return a failed message with invalided activation's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_011, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);

    std::shared_ptr<NNTensor> activationTensor = TransToNNTensor(OH_NN_INT8,
        m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);
    int8_t activationValue = -1;
    activationTensor->SetBuffer(&activationValue, sizeof(activationValue));
    m_allTensors.emplace_back(activationTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    activationTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: scale_build_012
 * @tc.desc: Verify that the build function return a failed message with invalided parameter
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_012, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: scale_build_013
 * @tc.desc: Verify that the build function return a failed message with empty axis's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_013, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);

    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);
    m_allTensors.emplace_back(axisTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    axisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: scale_build_014
 * @tc.desc: Verify that the build function return a failed message with empty activation's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_build_014, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);

    std::shared_ptr<NNTensor> activationTensor = TransToNNTensor(OH_NN_INT8, m_paramDim,
        nullptr, OH_NN_SCALE_ACTIVATIONTYPE);
    m_allTensors.emplace_back(activationTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    activationTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: scale_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: scale_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ScaleBuilderTest, scale_get_primitive_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveAxisTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SCALE_AXIS);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_SCALE_ACTIVATIONTYPE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(primitive, expectPrimitive);

    int64_t axisValue = 1;
    int8_t activationValue = 0;
    auto axisReturn = mindspore::lite::MindIR_ScaleFusion_GetAxis(primitive.get());
    EXPECT_EQ(axisReturn, axisValue);
    auto activationReturn = mindspore::lite::MindIR_ScaleFusion_GetActivationType(primitive.get());
    EXPECT_EQ(activationReturn, activationValue);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS