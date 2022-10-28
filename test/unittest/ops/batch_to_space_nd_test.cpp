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

#include "frameworks/native/ops/batch_to_space_nd_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class BatchToSpaceNDBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

    void SetBlockSize(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetCrops(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

public:
    BatchToSpaceNDBuilder m_builder;
    std::vector<uint32_t> m_inputs{0};
    std::vector<uint32_t> m_outputs{1};
    std::vector<uint32_t> m_params{2, 3};
    std::vector<int32_t> m_input_dim{4, 1, 1, 1};
    std::vector<int32_t> m_output_dim{1, 2, 2, 1};
    std::vector<int32_t> m_block_dim{2};
    std::vector<int32_t> m_crops_dim{2, 2};
};

void BatchToSpaceNDBuilderTest::SetUp() {}

void BatchToSpaceNDBuilderTest::TearDown() {}

void BatchToSpaceNDBuilderTest::SetBlockSize(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int32_t blockNum = 2;
    int64_t* blockSizeValue = new (std::nothrow) int64_t[2]{2, 2};
    EXPECT_NE(nullptr, blockSizeValue);
    tensor->SetBuffer(blockSizeValue, sizeof(int64_t) * blockNum);
    m_allTensors.emplace_back(tensor);
}

void BatchToSpaceNDBuilderTest::SetCrops(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t cropsNum = 4;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* cropsValue = new (std::nothrow) int64_t[4]{0, 0, 0, 0};
    EXPECT_NE(nullptr, cropsValue);
    tensor->SetBuffer(cropsValue, sizeof(int64_t) * cropsNum);
    m_allTensors.emplace_back(tensor);
}

/**
 * @tc.name: batch_to_space_nd_build_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_003, TestSize.Level1)
{
    m_params = {1, 2};
    m_paramsIndex = m_params;
    m_inputs = {};
    m_outputs = {0};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_004, TestSize.Level1)
{
    m_inputs = {};
    m_outputs = {0};
    m_params = {1, 2};

    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_005, TestSize.Level1)
{
    m_inputs = {6};
    m_outputs = {1};
    m_params = {2, 3};

    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_006, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {6};
    m_params = {2, 3};

    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_007
 * @tc.desc: Verify the invalid crops of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_007, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_crops_dim, nullptr,
        OH_NN_BATCH_TO_SPACE_ND_CROPS);
    int32_t cropsNum = 4;
    int32_t* cropsValue = new (std::nothrow) int32_t[4]{0, 0, 0, 0};
    EXPECT_NE(nullptr, cropsValue);

    tensor->SetBuffer(cropsValue, sizeof(int32_t) * cropsNum);
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_008
 * @tc.desc: Verify the invalid blocksize of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_008, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_block_dim, nullptr,
        OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    int32_t blockNum = 2;
    int32_t* blockSizeValue = new (std::nothrow) int32_t[2]{2, 2};
    EXPECT_NE(nullptr, blockSizeValue);
    tensor->SetBuffer(blockSizeValue, sizeof(int32_t) * blockNum);
    m_allTensors.emplace_back(tensor);

    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_009
 * @tc.desc: Verify the invalid param to batchtospace of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_009, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_block_dim, nullptr,
        OH_NN_CONV2D_STRIDES);
    int64_t blockNum = 2;
    int64_t* blockSizeValue = new (std::nothrow) int64_t[2]{2, 2};
    EXPECT_NE(nullptr, blockSizeValue);
    tensor->SetBuffer(blockSizeValue, sizeof(int64_t) * blockNum);

    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_010
 * @tc.desc: Verify the batchtospacend without set blocksize of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_010, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_block_dim, nullptr,
        OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    m_allTensors.emplace_back(tensor);

    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_build_011
 * @tc.desc: Verify the batchtospacend without set crops of the build function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_build_011, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_crops_dim, nullptr,
        OH_NN_BATCH_TO_SPACE_ND_CROPS);
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: batch_to_space_nd_getprimitive_001
 * @tc.desc: Verify the success of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_getprimitive_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);

    std::vector<int64_t> blockSizeValue{2, 2};
    std::vector<std::vector<int64_t>> cropsValue{{0, 0}, {0, 0}};
    std::vector<int64_t> returnValue = mindspore::lite::MindIR_BatchToSpaceND_GetBlockShape(primitive.get());
    EXPECT_EQ(returnValue, blockSizeValue);
    std::vector<std::vector<int64_t>> cropsReturn = mindspore::lite::MindIR_BatchToSpaceND_GetCrops(primitive.get());
    EXPECT_EQ(cropsReturn, cropsValue);
}

/**
 * @tc.name: batch_to_space_nd_getprimitive_002
 * @tc.desc: Verify the nullptr of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(BatchToSpaceNDBuilderTest, batch_to_space_nd_getprimitive_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetBlockSize(OH_NN_INT64, m_block_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE);
    SetCrops(OH_NN_INT64, m_crops_dim, nullptr, OH_NN_BATCH_TO_SPACE_ND_CROPS);

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
