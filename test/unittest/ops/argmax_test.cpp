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

#include "frameworks/native/ops/argmax_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ArgMaxBuilderTest : public OpsTest {
public:
    void SetUp();
    void TearDown();

    void SetArgmaxAxis(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetArgmaxKeepdims(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

public:
    ArgMaxBuilder m_builder;
    std::vector<uint32_t> m_inputs{0};
    std::vector<uint32_t> m_outputs{1};
    std::vector<uint32_t> m_params{2, 3};
    std::vector<int32_t> m_input_dim{3, 3};
    std::vector<int32_t> m_output_dim{3, 3};
    std::vector<int32_t> m_param_dim{};
};

void ArgMaxBuilderTest::SetUp() {}

void ArgMaxBuilderTest::TearDown() {}

void ArgMaxBuilderTest::SetArgmaxAxis(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axisValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, axisValue);
    tensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
}

void ArgMaxBuilderTest::SetArgmaxKeepdims(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool* keepdimsValue = new (std::nothrow) bool(false);
    EXPECT_NE(nullptr, keepdimsValue);
    tensor->SetBuffer(keepdimsValue, sizeof(keepdimsValue));
    m_allTensors.emplace_back(tensor);
}

/**
 * @tc.name: argmax_build_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}
/**
 * @tc.name: argmax_build_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_002, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {1};
    m_params = {2, 3};

    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_003, TestSize.Level1)
{
    m_inputs = {};
    m_outputs = {1};
    m_params = {2, 3};

    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_004, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {};
    m_params = {1, 2};

    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_005, TestSize.Level1)
{
    m_inputs = {6};
    m_outputs = {1};
    m_params = {2, 3};

    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_006, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {6};
    m_params = {2, 3};

    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_007
 * @tc.desc: Verify the invalid axis of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_007, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_FLOAT32, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    float* axisValueTest = new (std::nothrow) float(0);
    EXPECT_NE(nullptr, axisValueTest);

    tensor->SetBuffer(axisValueTest, sizeof(float));
    m_allTensors.emplace_back(tensor);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_008
 * @tc.desc: Verify the invalid keepdims of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_008, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    int64_t* keepdimsValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, keepdimsValue);

    tensor->SetBuffer(keepdimsValue, sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_009
 * @tc.desc: Verify the invalid param to argmax of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_009, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_param_dim, nullptr, OH_NN_AVG_POOL_STRIDE);
    int64_t* strideValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, strideValue);

    tensor->SetBuffer(strideValue, sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_010
 * @tc.desc: Verify the argmax without set axis of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_010, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    m_allTensors.emplace_back(tensor);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: argmax_build_011
 * @tc.desc: Verify the argmax without set keepdims of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, argmax_build_011, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: add_getprimitive_001
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, add_getprimitive_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);
    EXPECT_NE(nullptr, primitive);

    int64_t returnValue = mindspore::lite::MindIR_ArgMaxFusion_GetAxis(primitive.get());
    EXPECT_EQ(returnValue, 0);
    bool keepdimsReturn = mindspore::lite::MindIR_ArgMaxFusion_GetKeepDims(primitive.get());
    EXPECT_EQ(keepdimsReturn, false);
}

/**
 * @tc.name: add_getprimitive_002
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ArgMaxBuilderTest, add_getprimitive_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetArgmaxAxis(OH_NN_INT64, m_param_dim, nullptr, OH_NN_ARG_MAX_AXIS);
    SetArgmaxKeepdims(OH_NN_BOOL, m_param_dim, nullptr, OH_NN_ARG_MAX_KEEPDIMS);
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS