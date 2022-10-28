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

#include "frameworks/native/ops/reduceall_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ReduceAllBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveParamsTensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    ReduceAllBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {3};
    std::vector<int32_t> m_inputDim {1, 1, 2, 2};
    std::vector<int32_t> m_outputDim {1, 1, 1, 2};
    std::vector<int32_t> m_paramDim {1};
};

void ReduceAllBuilderTest::SetUp() {}

void ReduceAllBuilderTest::TearDown() {}

void ReduceAllBuilderTest::SaveParamsTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> keepDimsTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool *keepDimsValue = new (std::nothrow) bool(true);
    EXPECT_NE(nullptr, keepDimsValue);
    keepDimsTensor->SetBuffer(keepDimsValue, sizeof(bool));
    m_allTensors.emplace_back(keepDimsTensor);
}

/**
 * @tc.name: reduceall_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: reduceall_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: reduceall_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2};
    m_outputs = {3};
    m_params = {4};

    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reduceall_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_004, TestSize.Level0)
{
    m_outputs = {2, 3};
    m_params = {4};

    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reduceall_build_005
 * @tc.desc: Verify that the build function return a failed message with null allTensor
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reduceall_build_006
 * @tc.desc: Verify that the build function return a failed message without output tensor
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reduceall_build_007
 * @tc.desc: Verify that the build function return a failed message with invalided keepdims's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_007, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> keepDimsTensor = TransToNNTensor(OH_NN_INT64,
        m_paramDim, nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);
    int64_t keepDimsValue = 1;
    keepDimsTensor->SetBuffer(&keepDimsValue, sizeof(keepDimsValue));
    m_allTensors.emplace_back(keepDimsTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    keepDimsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reduceall_build_008
 * @tc.desc: Verify that the build function return a failed message with invalided keepdims's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_008, TestSize.Level0)
{
    m_paramDim = {1, 2};

    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> keepDimsTensor = TransToNNTensor(OH_NN_BOOL, m_paramDim,
        nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);
    bool keepDimsValue[2] = {true, true};
    keepDimsTensor->SetBuffer(keepDimsValue, 2 * sizeof(bool));
    m_allTensors.emplace_back(keepDimsTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    keepDimsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reduceall_build_009
 * @tc.desc: Verify that the build function return a failed message with invalided parameter
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_009, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: reduceall_build_010
 * @tc.desc: Verify that the build function return a failed message with empty keepdims's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_build_010, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> keepDimsTensor = TransToNNTensor(OH_NN_BOOL,
        m_paramDim, nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);
    m_allTensors.emplace_back(keepDimsTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    keepDimsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: reduceall_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: reduceall_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ReduceAllBuilderTest, reduceall_get_primitive_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_BOOL, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_BOOL, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);

    bool keepDimsValue = true;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr reduceallPrimitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(reduceallPrimitive, expectPrimitive);
    auto returnValue = mindspore::lite::MindIR_ReduceFusion_GetKeepDims(reduceallPrimitive.get());
    EXPECT_EQ(returnValue, keepDimsValue);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS