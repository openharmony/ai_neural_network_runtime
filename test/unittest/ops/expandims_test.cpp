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

#include "frameworks/native/ops/expandims_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ExpandDimsBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

public:
    ExpandDimsBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {};
    std::vector<int32_t> m_input_dim {3, 3};
    std::vector<int32_t> m_output_dim {3, 3};
    std::vector<int32_t> m_param_dim {};
};

void ExpandDimsBuilderTest::SetUp() {}

void ExpandDimsBuilderTest::TearDown() {}

/**
 * @tc.name: expandims_build_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ExpandDimsBuilderTest, expandims_build_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: expandims_build_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ExpandDimsBuilderTest, expandims_build_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: expandims_build_003
 * @tc.desc: Verify the misssing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ExpandDimsBuilderTest, expandims_build_003, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {1};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: expandims_build_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ExpandDimsBuilderTest, expandims_build_004, TestSize.Level1)
{
    m_outputs = {};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: expandims_build_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ExpandDimsBuilderTest, expandims_build_005, TestSize.Level1)
{
    m_inputs = {0, 6};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: expandims_build_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(ExpandDimsBuilderTest, expandims_build_006, TestSize.Level1)
{
    m_outputs = {6};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: expandims_build_007
 * @tc.desc: Verify the paramIndex not empth of the build function
 * @tc.type: FUNC
 */

HWTEST_F(ExpandDimsBuilderTest, expandims_build_007, TestSize.Level1)
{
    m_params = {1};
    m_param_dim = {1};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: expandims_getprimitive_001
 * @tc.desc: Verify the success of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ExpandDimsBuilderTest, expandims_getprimitive_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);
}

/**
 * @tc.name: expandims_getprimitive_002
 * @tc.desc: Verify the nullptr return of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ExpandDimsBuilderTest, expandims_getprimitive_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS