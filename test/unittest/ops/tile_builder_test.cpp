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

#include "ops/tile_builder.h"

#include <gtest/gtest.h>
#include "nn_tensor.h"
#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class TileBuilderTest : public OpsTest {
protected:
    void InitTensor(const std::vector<uint32_t>& inputsIndex,
        const std::vector<uint32_t>& outputsIndex) override;
    void SaveDimsTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    TileBuilder m_builder;
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<uint32_t> paramsIndex = { 3 };
    std::vector<int32_t> paramDim = {};
};

void TileBuilderTest::InitTensor(const std::vector<uint32_t>& inputsIndex,
                                 const std::vector<uint32_t>& outputsIndex)
{
    std::vector<int32_t> inputDim = {2, 2};
    std::vector<int32_t> OutputDim = {4, 4};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
}

void TileBuilderTest::SaveDimsTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> dimsTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* dimsValue = new (std::nothrow) int64_t[1] {0};
    EXPECT_NE(nullptr, dimsValue);
    dimsTensor->SetBuffer(dimsValue, sizeof(int64_t));
    m_allTensors.emplace_back(dimsTensor);
}

/**
 * @tc.name: tile_build_001
 * @tc.desc: Provide normal input, output to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_001, TestSize.Level0)
{
    InitTensor(inputsIndex, outputsIndex);
    SaveDimsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_TILE_DIMS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: tile_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_002, TestSize.Level0)
{
    InitTensor(inputsIndex, outputsIndex);
    SaveDimsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_TILE_DIMS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: tile_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_003, TestSize.Level0)
{
    inputsIndex = { 0, 1, 2 };
    outputsIndex = { 3 };
    paramsIndex = { 4 };

    InitTensor(inputsIndex, outputsIndex);
    SaveDimsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_TILE_DIMS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: tile_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_004, TestSize.Level0)
{
    outputsIndex = { 2, 3 };
    paramsIndex = { 4 };

    InitTensor(inputsIndex, outputsIndex);
    SaveDimsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_TILE_DIMS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: tile_build_005
 * @tc.desc: Provide empty input, output, and parameters to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_builder.Build(paramsIndex, inputsIndex, outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: tile_build_006
 * @tc.desc: Provide empty output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_006, TestSize.Level0)
{
    std::vector<int32_t> inputDim = {2, 2};

    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: tile_build_007
 * @tc.desc: Provide a valid datatype param to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_007, TestSize.Level0)
{
    std::vector<int32_t> inputDim = {2, 2};
    std::vector<int32_t> OutputDim = {4, 4};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);

    std::shared_ptr<NNTensor> dimsTensor = TransToNNTensor(OH_NN_FLOAT32, paramDim,
        nullptr, OH_NN_TILE_DIMS);
    float* dimsValue = new (std::nothrow) float[1] {0.0f};
    EXPECT_NE(nullptr, dimsValue);
    dimsTensor->SetBuffer(dimsValue, sizeof(float));
    m_allTensors.emplace_back(dimsTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    dimsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: tile_build_008
 * @tc.desc: Provide a valid type param to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_008, TestSize.Level0)
{
    std::vector<int32_t> inputDim = {2, 2};
    std::vector<int32_t> OutputDim = {4, 4};
    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
    SaveDimsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: tile_build_009
 * @tc.desc: Provide a param without set buffer to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_build_009, TestSize.Level0)
{
    std::vector<int32_t> inputDim = {2, 2};
    std::vector<int32_t> OutputDim = {4, 4};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);

    std::shared_ptr<NNTensor> dimsTensor = TransToNNTensor(OH_NN_INT64, paramDim,
        nullptr, OH_NN_TILE_DIMS);
    m_allTensors.emplace_back(dimsTensor);


    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: tile_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = { nullptr, DestroyLiteGraphPrimitive };
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: tile_getprimitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(TileBuilderTest, tile_getprimitive_002, TestSize.Level0)
{
    InitTensor(inputsIndex, outputsIndex);
    SaveDimsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_TILE_DIMS);

    std::vector<int64_t> expectDimsValue = {0};
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = { nullptr, DestroyLiteGraphPrimitive };
    EXPECT_NE(primitive, expectPrimitive);

    auto returnDims = mindspore::lite::MindIR_TileFusion_GetDims(primitive.get());
    auto returnDimsSize = returnDims.size();
    for (size_t i = 0; i < returnDimsSize; ++i) {
        EXPECT_EQ(returnDims[i], expectDimsValue[i]);
    }
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
