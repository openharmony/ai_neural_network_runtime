/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#include "ops/space_to_depth_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class SpaceToDepthBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveBlockSize(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    SpaceToDepthBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2};
    std::vector<int32_t> m_inputDim {1, 3, 2, 2};
    std::vector<int32_t> m_outputDim {1, 12, 1, 1};
    std::vector<int32_t> m_paramDim {};
};

void SpaceToDepthBuilderTest::SetUp() {}

void SpaceToDepthBuilderTest::TearDown() {}

void SpaceToDepthBuilderTest::SaveBlockSize(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> blockSizeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* blockSizeValue = new (std::nothrow) int64_t [1] {2};
    EXPECT_NE(nullptr, blockSizeValue);
    blockSizeTensor->SetBuffer(blockSizeValue, sizeof(int64_t));
    m_allTensors.emplace_back(blockSizeTensor);
}

/**
 * @tc.name: space_to_depth_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveBlockSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SPACE_TO_DEPTH_BLOCK_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: space_to_depth_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveBlockSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SPACE_TO_DEPTH_BLOCK_SIZE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: space_to_depth_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_003, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveBlockSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SPACE_TO_DEPTH_BLOCK_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: space_to_depth_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_004, TestSize.Level1)
{
    m_outputs = {1, 2};
    m_params = {3};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveBlockSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SPACE_TO_DEPTH_BLOCK_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: space_to_depth_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: space_to_depth_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: space_to_depth_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid blockSize's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> blockSizeTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_SPACE_TO_DEPTH_BLOCK_SIZE);
    float* blockSizeValue = new (std::nothrow) float [1]{2.0f};
    EXPECT_NE(nullptr, blockSizeValue);
    blockSizeTensor->SetBuffer(blockSizeValue, sizeof(float));
    m_allTensors.emplace_back(blockSizeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    blockSizeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: space_to_depth_build_008
 * @tc.desc: Verify that the build function returns a failed message with passing invalid blockSize param.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveBlockSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: space_to_depth_build_009
 * @tc.desc: Verify that the build function returns a failed message without set buffer for blockSize.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> blockSizeTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_SPACE_TO_DEPTH_BLOCK_SIZE);
    m_allTensors.emplace_back(blockSizeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: space_to_depth_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveBlockSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_SPACE_TO_DEPTH_BLOCK_SIZE);

    int64_t blockSizeValue = 2;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));

    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnBlockSizeValue = mindspore::lite::MindIR_SpaceToDepth_GetBlockSize(primitive.get());
    EXPECT_EQ(returnBlockSizeValue, blockSizeValue);
}

/**
 * @tc.name: space_to_depth_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToDepthBuilderTest, space_to_depth_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}