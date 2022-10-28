/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "frameworks/native/ops/space_to_batch_nd_builder.h"

#include <gtest/gtest.h>
#include "frameworks/native/nn_tensor.h"
#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class SpaceToBatchNDBuilderTest : public OpsTest {
protected:
    void InitTensor(const std::vector<uint32_t>& inputsIndex,
        const std::vector<uint32_t>& outputsIndex) override;
    void SaveBlockShapeTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SavePaddingsTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void InitTensor(const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex,
        const std::vector<uint32_t>& paramsIndex);

protected:
    SpaceToBatchNDBuilder m_builder;
    std::vector<int64_t> m_expectBlockShapeValue;
    std::vector<std::vector<int64_t>> m_expectPaddingsValue;
};

void SpaceToBatchNDBuilderTest::InitTensor(const std::vector<uint32_t>& inputsIndex,
    const std::vector<uint32_t>& outputsIndex)
{
    std::vector<uint32_t> paramsIndex = { 2, 3 };
    std::vector<int32_t> inputDim = {3, 2, 3};
    std::vector<int32_t> OutputDim = {1, 1, 3};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
}

void SpaceToBatchNDBuilderTest::InitTensor(const std::vector<uint32_t>& inputsIndex,
    const std::vector<uint32_t>& outputsIndex, const std::vector<uint32_t>& paramsIndex)
{
    std::vector<int32_t> inputDim = {1, 2, 2, 1};
    std::vector<int32_t> OutputDim = {4, 1, 1, 1};
    std::vector<int32_t> shapeDim = {3};
    std::vector<int32_t> paddingsDim = {2, 2};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);
}

void SpaceToBatchNDBuilderTest::SaveBlockShapeTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    const int blockShapeLen = 2;
    std::shared_ptr<NNTensor> blockShapeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* blockShapeValue = new (std::nothrow) int64_t[blockShapeLen] {2, 2};
    EXPECT_NE(nullptr, blockShapeValue);
    blockShapeTensor->SetBuffer(blockShapeValue, sizeof(int64_t) * blockShapeLen);
    m_allTensors.emplace_back(blockShapeTensor);
    m_expectBlockShapeValue.assign(blockShapeValue, blockShapeValue + blockShapeLen);
}

void SpaceToBatchNDBuilderTest::SavePaddingsTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    const int paddingsLen = 4;
    const int row = 2;
    const int col = 2;
    std::shared_ptr<NNTensor> paddingsTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* paddingsValue = new (std::nothrow) int64_t[paddingsLen] {0, 0, 0, 0};
    EXPECT_NE(nullptr, paddingsValue);
    paddingsTensor->SetBuffer(paddingsValue, sizeof(int64_t) * paddingsLen);
    m_allTensors.emplace_back(paddingsTensor);

    m_expectPaddingsValue.resize(row);
    for (int i = 0; i < row; ++i) {
        m_expectPaddingsValue[i].resize(col);
    }

    int i = 0;
    int j = 0;
    for (int k = 0; k < paddingsLen; ++k) {
        i = k / col;
        j = k % col;
        m_expectPaddingsValue[i][j] = paddingsValue[k];
    }
}

/**
 * @tc.name: spacetobatchnd_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_001, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: spacetobatchnd_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);

    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: spacetobatchnd_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_003, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_004, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_005
 * @tc.desc:  Provide empty input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_005, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = {};
    std::vector<uint32_t> outputsIndex = {};
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_006
 * @tc.desc: Provide param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_006, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);

    SaveBlockShapeTensor(OH_NN_INT32, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_007
 * @tc.desc: Provide param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_007, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);

    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT32, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_008
 * @tc.desc: Provide input dimensions error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_008, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<uint32_t> paramsIndex = { 2, 3 };
    InitTensor(inputsIndex, outputsIndex, paramsIndex);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_009
 * @tc.desc: Provide output dimensions error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_009, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<uint32_t> paramsIndex = { 2, 3 };
    InitTensor(inputsIndex, outputsIndex, paramsIndex);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_010
 * @tc.desc: Provide empty output to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_0010, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = {};
    std::vector<uint32_t> paramsIndex = { 1, 2 };

    InitTensor(inputsIndex, outputsIndex, paramsIndex);
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_011
 * @tc.desc: Provide block shape parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_011, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    std::shared_ptr<NNTensor> blockShapeTensor = TransToNNTensor(OH_NN_INT64, shapeDim, nullptr,
        OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    blockShapeTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(blockShapeTensor);

    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_012
 * @tc.desc: Provide paddings parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_012, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);

    std::shared_ptr<NNTensor> blockShapeTensor = TransToNNTensor(OH_NN_INT64, paddingsDim, nullptr,
        OH_NN_SPACE_TO_BATCH_ND_PADDINGS);
    blockShapeTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(blockShapeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_013
 * @tc.desc: Provide invalid parameter type to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_013, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SCALE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_014
 * @tc.desc: Provide block shape parameter dimension error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_014, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2, 3};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_015
 * @tc.desc: Provide paddings parameter dimension error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_015, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_build_016
 * @tc.desc: Provide paddings parameter dimension error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_build_016, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 3};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);

    const int paddingsLen = 6;
    std::shared_ptr<NNTensor> paddingsTensor = TransToNNTensor(OH_NN_INT64, paddingsDim, nullptr,
        OH_NN_SPACE_TO_BATCH_ND_PADDINGS);
    int64_t* paddingsValue = new (std::nothrow) int64_t[paddingsLen] {0, 0, 0, 0, 0, 0};
    EXPECT_NE(nullptr, paddingsValue);
    paddingsTensor->SetBuffer(paddingsValue, sizeof(int64_t) * paddingsLen);
    m_allTensors.emplace_back(paddingsTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: spacetobatchnd_getprimitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_getprimitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: spacetobatchnd_getprimitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(SpaceToBatchNDBuilderTest, spacetobatchnd_getprimitive_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> shapeDim = {2};
    std::vector<int32_t> paddingsDim = {2, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveBlockShapeTensor(OH_NN_INT64, shapeDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE);
    SavePaddingsTensor(OH_NN_INT64, paddingsDim, nullptr, OH_NN_SPACE_TO_BATCH_ND_PADDINGS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(primitive, expectPrimitive);

    auto returnValue = mindspore::lite::MindIR_SpaceToBatchND_GetPaddings(primitive.get());
    auto returnValueSize = returnValue.size();
    for (size_t i = 0; i < returnValueSize; ++i) {
        auto k = returnValue[i].size();
        for (size_t j = 0; j < k; ++j) {
            EXPECT_EQ(returnValue[i][j], m_expectPaddingsValue[i][j]);
        }
    }

    auto returnBlockShape = mindspore::lite::MindIR_SpaceToBatchND_GetBlockShape(primitive.get());
    auto returnBlockShapeSize = returnBlockShape.size();
    for (size_t i = 0; i < returnBlockShapeSize; ++i) {
        EXPECT_EQ(returnBlockShape[i], m_expectBlockShapeValue[i]);
    }
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
