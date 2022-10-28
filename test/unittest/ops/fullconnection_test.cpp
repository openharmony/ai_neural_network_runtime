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
class FullConnectionBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

    void SetInputToAlltensor();
    void SetActivation(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

public:
    FullConnectionBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1, 2};
    std::vector<uint32_t> m_outputs {3};
    std::vector<uint32_t> m_params {4};
    std::vector<int32_t> m_output_dim {2, 2};
    std::vector<int32_t> m_param_dim {};
};

void FullConnectionBuilderTest::SetUp() {}

void FullConnectionBuilderTest::TearDown() {}

void FullConnectionBuilderTest::SetInputToAlltensor()
{
    std::vector<int32_t> m_input_dim{2, 2};
    std::vector<int32_t> biasDim = {2};
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(tensor);

    int32_t numWeight = 4;
    int32_t numBias = 2;
    tensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    float* valueWeight = new (std::nothrow) float[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, valueWeight);

    tensor->SetBuffer(valueWeight, numWeight * sizeof(float));
    m_allTensors.emplace_back(tensor);

    tensor = TransToNNTensor(OH_NN_FLOAT32, biasDim, nullptr, OH_NN_TENSOR);
    float* valueBias = new (std::nothrow) float[2]{0, 0};
    EXPECT_NE(nullptr, valueBias);
    tensor->SetBuffer(valueBias, numBias * sizeof(float));
    m_allTensors.emplace_back(tensor);
}

void FullConnectionBuilderTest::SetActivation(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* activationValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
}

/**
 * @tc.name: fullconnection_build_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionBuilderTest, fullconnection_build_001, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionBuilderTest, fullconnection_build_002, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_003
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionBuilderTest, fullconnection_build_003, TestSize.Level1)
{
    m_outputs = {};
    m_params = {3};
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_004
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionBuilderTest, fullconnection_build_004, TestSize.Level1)
{
    m_inputs = {0, 1, 6};
    m_outputs = {3};
    m_params = {4};

    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_005
 * @tc.desc: Verify the behavior of the build function
 * @tc.type: FUNC
 */

HWTEST_F(FullConnectionBuilderTest, fullconnection_build_005, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr,
        OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    int32_t *activationValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_006
 * @tc.desc: Verify the behavior of the build function
 * @tc.type: FUNC
 */

HWTEST_F(FullConnectionBuilderTest, fullconnection_build_006, TestSize.Level1)
{
    m_param_dim = {2};
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    int8_t *activationValue = new (std::nothrow) int8_t[2]{0, 0};
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_007
 * @tc.desc: Verify the invalid avtivation value of the build function
 * @tc.type: FUNC
 */

HWTEST_F(FullConnectionBuilderTest, fullconnection_build_007, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    int8_t *activationValue = new (std::nothrow) int8_t(10);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_build_008
 * @tc.desc: Verify the invalid param to fullconnection of the build function
 * @tc.type: FUNC
 */

HWTEST_F(FullConnectionBuilderTest, fullconnection_build_008, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_DIV_ACTIVATIONTYPE);
    int8_t *activationValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValue);
    tensor->SetBuffer(activationValue, sizeof(int8_t));

    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: fullconnection_getprimitive_001
 * @tc.desc: Verify the success of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionBuilderTest, fullconnection_getprimitive_001, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);

    int8_t activationReturn = mindspore::lite::MindIR_FullConnection_GetActivationType(primitive.get());
    EXPECT_EQ(activationReturn, 0);
}

/**
 * @tc.name: fullconnection_getprimitive_002
 * @tc.desc: Verify the nullptr return of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(FullConnectionBuilderTest, fullconnection_getprimitive_002, TestSize.Level1)
{
    m_inputsIndex = m_inputs;
    m_paramsIndex = m_params;
    SetInputToAlltensor();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_FULL_CONNECTION_ACTIVATIONTYPE);
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS