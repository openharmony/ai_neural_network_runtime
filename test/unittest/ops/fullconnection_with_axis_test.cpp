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

#include "frameworks/native/ops/fullconnection_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class FullConnectionAxisBuilderTest : public OpsTest {
public:
    void SetUp();
    void TearDown();

    void SetInputToAlltensor();
    void SetActivation(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SeAxis(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

public:
    FullConnectionBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1, 2};
    std::vector<uint32_t> m_outputs {3};
    std::vector<uint32_t> m_params {4, 5};
    std::vector<int32_t> m_output_dim {2, 2};
    std::vector<int32_t> m_param_dim {};
};

void FullConnectionAxisBuilderTest::SetUp() {}

void FullConnectionAxisBuilderTest::TearDown() {}

void FullConnectionAxisBuilderTest::SetInputToAlltensor()
{
    std::vector<int32_t> m_input_dim{2, 2};
    std::vector<int32_t> biasDim{2};
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(tensor);
    int32_t weightNum = 4;
    int32_t biasNum = 2;
    tensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    float* valueWeight = new (std::nothrow) float[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, valueWeight);
    tensor->SetBuffer(valueWeight, weightNum * sizeof(float));
    m_allTensors.emplace_back(tensor);

    tensor = TransToNNTensor(OH_NN_FLOAT32, biasDim, nullptr, OH_NN_TENSOR);
    float* valueBias = new (std::nothrow) float[2]{0, 0};
    EXPECT_NE(nullptr, valueBias);
    tensor->SetBuffer(valueBias, biasNum * sizeof(float));
    m_allTensors.emplace_back(tensor);
}

void FullConnectionAxisBuilderTest::SetActivation(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* activationValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
}

void FullConnectionAxisBuilderTest::SeAxis(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axisValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, axisValue);

    tensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
}

/**
 * @tc.name: fullconnection_build_axis_001
 * @tc.desc: Verify the behavior of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_001, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_002
 * @tc.desc: Verify the behavior of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_002, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_003
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_003, TestSize.Level1)
{
    m_outputs = {};
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_004
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_004, TestSize.Level1)
{
    m_inputs = {0, 1, 6};
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_005
 * @tc.desc: Verify the invalid axis of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_005, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    int32_t *axisValueTest = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, axisValueTest);

    tensor->SetBuffer(axisValueTest, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_006
 * @tc.desc: Verify the behavior of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_006, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr,
        OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    int32_t *activationValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_007
 * @tc.desc: Verify the behavior of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_007, TestSize.Level1)
{
    std::vector<int32_t> paramDimTest = {2};
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;

    SetInputToAlltensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, paramDimTest, nullptr,
        OH_NN_FULL_CONNECTION_AXIS);
    int64_t *axisValueTest = new (std::nothrow) int64_t[2]{0, 0};
    EXPECT_NE(nullptr, axisValueTest);

    tensor->SetBuffer(axisValueTest, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_008
 * @tc.desc: Verify the behavior of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_008, TestSize.Level1)
{
    std::vector<int32_t> paramDimTest = {2};
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, paramDimTest, nullptr,
        OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    int8_t *activationValue = new (std::nothrow) int8_t[2]{0, 0};
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_009
 * @tc.desc: Verify the fullconnection without set axis of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_009, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_axis_010
 * @tc.desc: Verify the fullconnection without set activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_build_axis_010, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_getprimitive_axis_001
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_getprimitive_axis_001, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);

    int returnValue = mindspore::lite::MindIR_FullConnection_GetAxis(primitive.get());
    EXPECT_EQ(returnValue, 0);
    bool activationReturn = mindspore::lite::MindIR_FullConnection_GetActivationType(primitive.get());
    EXPECT_EQ(activationReturn, 0);
}

/**
 * @tc.name: fullconnection_getprimitive_axis_002
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionAxisBuilderTest, fullconnection_getprimitive_axis_002, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SeAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_AXIS);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS