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

#include "ops/gather_nd_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class GatherNDBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    GatherNDBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {};
    std::vector<int32_t> m_inputDim {4, 3};
    std::vector<int32_t> m_outputDim {4, 2};
};

void GatherNDBuilderTest::SetUp() {}

void GatherNDBuilderTest::TearDown() {}

/**
 * @tc.name: gatherND_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: gatherND_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: gatherND_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2};
    m_outputs = {3};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: gatherND_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_build_004, TestSize.Level0)
{
    std::vector<uint32_t> m_outputs = {2, 3};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: gatherND_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: gatherND_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: gatherND_build_007
 * @tc.desc: Verify that the build function returns a failed message with a virtual parameter.
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_build_007, TestSize.Level0)
{
    m_params = {3};
    std::vector<int32_t> paramDim = {};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    std::shared_ptr<NNTensor> paramTensor;
    paramTensor = TransToNNTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(paramTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: gatherND_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_getprimitive_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);
}

/**
 * @tc.name: gatherND_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(GatherNDBuilderTest, gatherND_getprimitive_002, TestSize.Level0)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}