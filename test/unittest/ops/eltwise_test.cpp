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

#include "frameworks/native/ops/eltwise_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class EltwiseBuilderTest : public OpsTest {
public:
    void SetUp();
    void TearDown();

    void SetEltwiseMode(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
public:
    EltwiseBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {3};
    std::vector<int32_t> m_input_dim {3, 3};
    std::vector<int32_t> m_output_dim {3, 3};
    std::vector<int32_t> m_param_dim {};
};

void EltwiseBuilderTest::SetUp() {}

void EltwiseBuilderTest::TearDown() {}

void EltwiseBuilderTest::SetEltwiseMode(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* modeValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, modeValue);
    tensor->SetBuffer(modeValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
}

/**
 * @tc.name: eltwise_build_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_build_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetEltwiseMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_build_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetEltwiseMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_build_003, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {1};
    m_params = {2};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetEltwiseMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_build_004, TestSize.Level1)
{
    m_outputs = {};
    m_params = {2};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetEltwiseMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_build_005, TestSize.Level1)
{
    m_inputs = {0, 6};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetEltwiseMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_build_006, TestSize.Level1)
{
    m_outputs = {6};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetEltwiseMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_007
 * @tc.desc: Verify the invalid eltwiseMode of the build function
 * @tc.type: FUNC
 */

HWTEST_F(EltwiseBuilderTest, eltwise_build_007, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);
    int32_t* modeValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, modeValue);

    tensor->SetBuffer(modeValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_008
 * @tc.desc: Verify the scalar length of the build function
 * @tc.type: FUNC
 */

HWTEST_F(EltwiseBuilderTest, eltwise_build_008, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_param_dim = {2};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);
    int8_t* modeValue = new (std::nothrow) int8_t[2]{0, 0};
    EXPECT_NE(nullptr, modeValue);

    tensor->SetBuffer(modeValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_008
 * @tc.desc: Verify the invalid mode value of the build function
 * @tc.type: FUNC
 */

HWTEST_F(EltwiseBuilderTest, eltwise_build_009, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);
    int8_t* modeValue = new (std::nothrow) int8_t(10);
    EXPECT_NE(nullptr, modeValue);

    tensor->SetBuffer(modeValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_010
 * @tc.desc: Verify the invalid param to eltwise of the build function
 * @tc.type: FUNC
 */

HWTEST_F(EltwiseBuilderTest, eltwise_build_010, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    int8_t* modeValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, modeValue);

    tensor->SetBuffer(modeValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_build_011
 * @tc.desc: Verify the eltwise without set mode of the build function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: eltwise_getprimitive_001
 * @tc.desc: Verify the success of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_getprimitive_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetEltwiseMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);
    bool eltwiseModeReturn = mindspore::lite::MindIR_Eltwise_GetMode(primitive.get());
    EXPECT_EQ(eltwiseModeReturn, eltwiseModeReturn);
}

/**
 * @tc.name: eltwise_getprimitive_002
 * @tc.desc: Verify the nullptr return of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(EltwiseBuilderTest, eltwise_getprimitive_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetEltwiseMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ELTWISE_MODE);

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS