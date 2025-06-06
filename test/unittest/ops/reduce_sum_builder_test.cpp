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

#include "ops/reducesum_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ReduceSumBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SetKeepDims(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetCoeff(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetReduceToEnd(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    ReduceSumBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {3, 4, 5};
    std::vector<int32_t> m_inputDim {1, 1, 2, 2};
    std::vector<int32_t> m_outputDim {1, 1, 1, 2};
    std::vector<int32_t> m_paramDim {1};
};

void ReduceSumBuilderTest::SetUp() {}

void ReduceSumBuilderTest::TearDown() {}

void ReduceSumBuilderTest::SetKeepDims(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> keepDimsTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool *keepDimsValue = new (std::nothrow) bool(true);
    EXPECT_NE(nullptr, keepDimsValue);
    keepDimsTensor->SetBuffer(keepDimsValue, sizeof(bool));
    m_allTensors.emplace_back(keepDimsTensor);
}

void ReduceSumBuilderTest::SetCoeff(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> coeffTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float *coeffValue = new (std::nothrow) float(0.0f);
    EXPECT_NE(nullptr, coeffValue);
    coeffTensor->SetBuffer(coeffValue, sizeof(float));
    m_allTensors.emplace_back(coeffTensor);
}

void ReduceSumBuilderTest::SetReduceToEnd(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> reduceToEndTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool *reduceToEndValue = new (std::nothrow) bool(true);
    EXPECT_NE(nullptr, reduceToEndValue);
    reduceToEndTensor->SetBuffer(reduceToEndValue, sizeof(bool));
    m_allTensors.emplace_back(reduceToEndTensor);
}

/**
 * @tc.name: reducesum_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: reducesum_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: reducesum_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2};
    m_outputs = {3};
    m_params = {4, 5, 6};

    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_004, TestSize.Level0)
{
    m_outputs = {2, 3};
    m_params = {4, 5, 6};

    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_005
 * @tc.desc: Verify that the build function return a failed message with null allTensor
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_006
 * @tc.desc: Verify that the build function return a failed message without output tensor
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_007
 * @tc.desc: Verify that the build function return a failed message with invalided keepdims's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_007, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> keepDimsTensor = TransToNNTensor(OH_NN_INT64,
        m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    int64_t keepDimsValue = 1;
    keepDimsTensor->SetBuffer(&keepDimsValue, sizeof(keepDimsValue));
    m_allTensors.emplace_back(keepDimsTensor);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    keepDimsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reducesum_build_008
 * @tc.desc: Verify that the build function return a failed message with invalided coeff's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_008, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    std::shared_ptr<NNTensor> coeffTensor = TransToNNTensor(OH_NN_INT64,
        m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    int64_t coeffValue = 1;
    coeffTensor->SetBuffer(&coeffValue, sizeof(int64_t));
    m_allTensors.emplace_back(coeffTensor);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    coeffTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reducesum_build_009
 * @tc.desc: Verify that the build function return a failed message with invalided reduceToEnd's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_009, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    std::shared_ptr<NNTensor> reduceToEndTensor = TransToNNTensor(OH_NN_INT64,
        m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);
    int64_t reduceToEndValue = 1;
    reduceToEndTensor->SetBuffer(&reduceToEndValue, sizeof(reduceToEndValue));
    m_allTensors.emplace_back(reduceToEndTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    reduceToEndTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reducesum_build_010
 * @tc.desc: Verify that the build function return a failed message with invalided keepdims's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_010, TestSize.Level0)
{
    std::vector<int32_t> m_paramDims = {1, 2};

    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> keepDimsTensor = TransToNNTensor(OH_NN_BOOL, m_paramDims,
        nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    bool keepDimsValue[2] = {true, true};
    keepDimsTensor->SetBuffer(keepDimsValue, 2 * sizeof(bool));
    m_allTensors.emplace_back(keepDimsTensor);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    keepDimsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reducesum_build_011
 * @tc.desc: Verify that the build function return a failed message with invalided coeff's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_011, TestSize.Level0)
{
    std::vector<int32_t> m_paramDims = {1, 2};

    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    std::shared_ptr<NNTensor> coeffTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDims,
        nullptr, OH_NN_REDUCE_SUM_COEFF);
    float coeffValue[2] = {1.0f, 1.0f};
    coeffTensor->SetBuffer(coeffValue, 2 * sizeof(float));
    m_allTensors.emplace_back(coeffTensor);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    coeffTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reducesum_build_012
 * @tc.desc: Verify that the build function return a failed message with invalided reduceToEnd's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_012, TestSize.Level0)
{
    std::vector<int32_t> m_paramDims = {1, 2};

    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    std::shared_ptr<NNTensor> reduceToEndTensor = TransToNNTensor(OH_NN_BOOL, m_paramDims,
        nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);
    bool reduceToEndValue[2] = {true, true};
    reduceToEndTensor->SetBuffer(reduceToEndValue, 2 * sizeof(bool));
    m_allTensors.emplace_back(reduceToEndTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    reduceToEndTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reducesum_build_013
 * @tc.desc: Verify that the build function return a failed message with invalided keepDims parameter
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_013, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_014
 * @tc.desc: Verify that the build function return a failed message with invalided coeff parameter
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_014, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_015
 * @tc.desc: Verify that the build function return a failed message with invalided reduceToEnd parameter
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_015, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_016
 * @tc.desc: Verify that the build function return a failed message with empty keepdims's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_016, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> keepDimsTensor = TransToNNTensor(OH_NN_BOOL,
        m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    m_allTensors.emplace_back(keepDimsTensor);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_017
 * @tc.desc: Verify that the build function return a failed message with empty coeff's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_017, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    std::shared_ptr<NNTensor> coeffTensor = TransToNNTensor(OH_NN_FLOAT32,
        m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    m_allTensors.emplace_back(coeffTensor);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_build_018
 * @tc.desc: Verify that the build function return a failed message with empty reduceToEnd's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_build_018, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    std::shared_ptr<NNTensor> reduceToEndTensor = TransToNNTensor(OH_NN_BOOL,
        m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);
    m_allTensors.emplace_back(reduceToEndTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reducesum_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: reducesum_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceSumBuilderTest, reducesum_get_primitive_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SetKeepDims(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_KEEP_DIMS);
    SetCoeff(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_REDUCE_SUM_COEFF);
    SetReduceToEnd(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_SUM_REDUCE_TO_END);

    bool keepDimsValue = true;
    float coeffValue = 0.0f;
    bool reduceToEndValue = true;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr reducesumPrimitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};

    EXPECT_NE(reducesumPrimitive, expectPrimitive);
    auto returnKeepDimsValue = mindspore::lite::MindIR_ReduceFusion_GetKeepDims(reducesumPrimitive.get());
    EXPECT_EQ(returnKeepDimsValue, keepDimsValue);
    auto returnCoeffValue = mindspore::lite::MindIR_ReduceFusion_GetCoeff(reducesumPrimitive.get());
    EXPECT_EQ(returnCoeffValue, coeffValue);
    auto returnReduceToEndValue = mindspore::lite::MindIR_ReduceFusion_GetReduceToEnd(reducesumPrimitive.get());
    EXPECT_EQ(returnReduceToEndValue, reduceToEndValue);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS