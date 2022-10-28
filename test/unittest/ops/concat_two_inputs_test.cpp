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

#include "frameworks/native/ops/concat_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ConcatTwoInputBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

    void SetAxis(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

public:
    ConcatBuilder m_builder;
    std::vector<uint32_t> m_inputs{0, 1};
    std::vector<uint32_t> m_outputs{2};
    std::vector<uint32_t> m_params{3};
    std::vector<int32_t> m_input_dim{3, 3};
    std::vector<int32_t> m_output_dim{3, 3};
    std::vector<int32_t> m_param_dim{};
};

void ConcatTwoInputBuilderTest::SetUp() {}

void ConcatTwoInputBuilderTest::TearDown() {}

void ConcatTwoInputBuilderTest::SetAxis(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axisValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, axisValue);

    tensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
}

/**
 * @tc.name: concat_build_two_input_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_build_two_input_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_build_two_input_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_003, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {2};
    m_params = {3};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_build_two_input_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_004, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {};
    m_params = {3};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_build_two_input_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_005, TestSize.Level1)
{
    m_inputs = {0, 1, 6};
    m_outputs = {3};
    m_params = {4};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_build_two_input_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_006, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {6};
    m_params = {3};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_build_two_input_007
 * @tc.desc: Verify the invalid axis of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_007, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    int32_t* axisValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, axisValue);

    tensor->SetBuffer(axisValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_build_two_input_008
 * @tc.desc: Verify the scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_008, TestSize.Level1)
{
    m_param_dim = {2};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    int64_t* axisValue = new (std::nothrow) int64_t[2]{0, 0};
    EXPECT_NE(nullptr, axisValue);

    tensor->SetBuffer(axisValue, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_build_two_input_009
 * @tc.desc: This is OH_NN_INVALID_PARAMETER case, course the value of axis is nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_build_two_input_010, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    m_allTensors.emplace_back(tensor);

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: concat_getprimitive_two_input_001
 * @tc.desc: Verify the success of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_getprimitive_two_input_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);

    int64_t returnValue = mindspore::lite::MindIR_Concat_GetAxis(primitive.get());
    EXPECT_EQ(returnValue, 0);
}

/**
 * @tc.name: concat_getprimitive_two_input_001
 * @tc.desc: Verify the nullptr return of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ConcatTwoInputBuilderTest, concat_getprimitive_two_input_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONCAT_AXIS);
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS