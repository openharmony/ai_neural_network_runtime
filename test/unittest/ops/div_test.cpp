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

#include "frameworks/native/ops/div_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class DivFusionTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

    void SaveParamsTensor(const std::vector<uint32_t>& m_params, OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

public:
    DivBuilder m_builder;
    std::vector<uint32_t> m_inputs{0, 1};
    std::vector<uint32_t> m_outputs{2};
    std::vector<uint32_t> m_params{3};
    std::vector<int32_t> m_input_dim{3, 3};
    std::vector<int32_t> m_output_dim{3, 3};
    std::vector<int32_t> m_param_dim{};
};

void DivFusionTest::SetUp() {}

void DivFusionTest::TearDown() {}

void DivFusionTest::SaveParamsTensor(const std::vector<uint32_t>& m_params, OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    m_paramsIndex = m_params;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* activationValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValue);
    tensor->SetBuffer(activationValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
}

/**
 * @tc.name: div_build_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SaveParamsTensor(m_params, OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SaveParamsTensor(m_params, OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_003, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {1};
    m_params = {2};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SaveParamsTensor(m_params, OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_004, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {};
    m_params = {2};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SaveParamsTensor(m_params, OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_005, TestSize.Level1)
{
    m_inputs = {0, 6};
    m_outputs = {2};
    m_params = {3};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SaveParamsTensor(m_params, OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_006, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {6};
    m_params = {3};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SaveParamsTensor(m_params, OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_007
 * @tc.desc: Verify the param invalid of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    m_paramsIndex = m_params;

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    int32_t* activationValueTest = new (std::nothrow) int32_t[0];
    EXPECT_NE(nullptr, activationValueTest);

    tensor->SetBuffer(activationValueTest, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_008
 * @tc.desc: Verify the scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_008, TestSize.Level1)
{
    m_param_dim = {2};
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    m_paramsIndex = m_params;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    int8_t* activationValueTest = new (std::nothrow) int8_t[2]{0, 0};
    EXPECT_NE(nullptr, activationValueTest);

    tensor->SetBuffer(activationValueTest, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_009
 * @tc.desc: Verify the invalid activation value of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    m_paramsIndex = m_params;

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    int8_t* activationValueTest = new (std::nothrow) int8_t(10);
    EXPECT_NE(nullptr, activationValueTest);

    tensor->SetBuffer(activationValueTest, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_010
 * @tc.desc: Verify the invalid param to div of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_010, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    m_paramsIndex = m_params;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr, OH_NN_ADD_ACTIVATIONTYPE);
    int8_t* activationValueTest = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValueTest);

    tensor->SetBuffer(activationValueTest, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_build_011
 * @tc.desc: Verify the div without set activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_DIV_ACTIVATIONTYPE);
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: div_getprimitive_001
 * @tc.desc: Verify the success of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SaveParamsTensor(m_params, OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);

    int8_t activationValueTest = 0;
    int8_t returnValue = mindspore::lite::MindIR_DivFusion_GetActivationType(primitive.get());
    EXPECT_EQ(returnValue, activationValueTest);
}

/**
 * @tc.name: div_getprimitive_002
 * @tc.desc: Verify the nullptr return of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(DivFusionTest, div_getprimitive_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SaveParamsTensor(m_params, OH_NN_INT8, m_param_dim, nullptr, OH_NN_DIV_ACTIVATIONTYPE);

    LiteGraphTensorPtr primitive = {nullptr, DestroyLiteGraphPrimitive};
    LiteGraphTensorPtr expectPrimitive = m_builder.GetPrimitive();
    EXPECT_EQ(primitive, expectPrimitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS