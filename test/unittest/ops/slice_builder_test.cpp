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

#include "ops/slice_builder.h"

#include <gtest/gtest.h>
#include "nn_tensor.h"
#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class SliceBuilderTest : public OpsTest {
protected:
    void InitTensor(const std::vector<uint32_t>& inputsIndex,
        const std::vector<uint32_t>& outputsIndex) override;
    void SaveAxesTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    SliceBuilder m_builder;
    std::vector<uint32_t> inputsIndex = { 0, 1, 2 };
    std::vector<uint32_t> outputsIndex = { 3 };
    std::vector<uint32_t> paramsIndex = { 4 };
    std::vector<int32_t> paramsDim = {};
};

void SliceBuilderTest::InitTensor(const std::vector<uint32_t>& inputsIndex,
                                  const std::vector<uint32_t>& outputsIndex)
{
    std::vector<int32_t> inputDim = {3, 2, 3};
    std::vector<int32_t> OutputDim = {1, 1, 3};

    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
}

void SliceBuilderTest::SaveAxesTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> axesTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axesValue = new (std::nothrow) int64_t[1]{0};
    EXPECT_NE(nullptr, axesValue);
    axesTensor->SetBuffer(axesValue, sizeof(int64_t));
    m_allTensors.emplace_back(axesTensor);
}

/**
 * @tc.name: slice_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_001, TestSize.Level0)
{
    InitTensor(inputsIndex, outputsIndex);
    SaveAxesTensor(OH_NN_INT64, paramsDim, nullptr, OH_NN_SLICE_AXES);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: slice_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_002, TestSize.Level0)
{
    InitTensor(inputsIndex, outputsIndex);
    SaveAxesTensor(OH_NN_INT64, paramsDim, nullptr, OH_NN_SLICE_AXES);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: slice_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_003, TestSize.Level0)
{
    inputsIndex = { 0, 1, 2, 3 };
    outputsIndex = { 4 };
    paramsIndex = { 5 };

    InitTensor(inputsIndex, outputsIndex);
    SaveAxesTensor(OH_NN_INT64, paramsDim, nullptr, OH_NN_SLICE_AXES);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: slice_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_004, TestSize.Level0)
{
    inputsIndex = { 0, 1, 2 };
    outputsIndex = { 3, 4 };
    paramsIndex = { 5 };

    InitTensor(inputsIndex, outputsIndex);
    SaveAxesTensor(OH_NN_INT64, paramsDim, nullptr, OH_NN_SLICE_AXES);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: slice_build_005
 * @tc.desc: Provide empty input, output, and parameters to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_005, TestSize.Level0)
{
    inputsIndex = {};
    outputsIndex = {};
    paramsIndex = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxesTensor(OH_NN_INT64, paramsDim, nullptr, OH_NN_SLICE_AXES);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: slice_build_006
 * @tc.desc: Provide empty output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_006, TestSize.Level0)
{
    inputsIndex = { 0, 1, 2 };
    outputsIndex = {};
    paramsIndex = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxesTensor(OH_NN_INT64, paramsDim, nullptr, OH_NN_SLICE_AXES);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: slice_build_007
 * @tc.desc: Provide a valid datatype param to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_007, TestSize.Level0)
{
    std::vector<int32_t> inputDim = { 3, 2, 3 };
    std::vector<int32_t> OutputDim = { 1, 1, 3 };
    std::vector<int32_t> paramsDim = {};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);

    std::shared_ptr<NNTensor> axesTensor = TransToNNTensor(OH_NN_FLOAT32, paramsDim,
        nullptr, OH_NN_SLICE_AXES);
    float* axesValue = new (std::nothrow) float[1] {0.0f};
    EXPECT_NE(nullptr, axesValue);
    axesTensor->SetBuffer(axesValue, sizeof(float));
    m_allTensors.emplace_back(axesTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: slice_build_008
 * @tc.desc: Provide a valid type param to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_008, TestSize.Level0)
{
    std::vector<int32_t> inputDim = { 3, 2, 3 };
    std::vector<int32_t> OutputDim = { 1, 1, 3 };
    std::vector<int32_t> paramsDim = {};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
    SaveAxesTensor(OH_NN_INT64, paramsDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);


    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: slice_build_009
 * @tc.desc: Provide a param without set buffer to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_build_009, TestSize.Level0)
{
    std::vector<int32_t> inputDim = { 3, 2, 3 };
    std::vector<int32_t> OutputDim = { 1, 1, 3 };
    std::vector<int32_t> paramsDim = {};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);

    std::shared_ptr<NNTensor> axesTensor = TransToNNTensor(OH_NN_INT64, paramsDim,
        nullptr, OH_NN_SLICE_AXES);
    m_allTensors.emplace_back(axesTensor);


    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: slice_getprimitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_getprimitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: slice_getprimitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(SliceBuilderTest, slice_getprimitive_002, TestSize.Level0)
{
    InitTensor(inputsIndex, outputsIndex);
    SaveAxesTensor(OH_NN_INT64, paramsDim, nullptr, OH_NN_SLICE_AXES);

    std::vector<int64_t> expectAxesValue = {0};
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(primitive, expectPrimitive);

    auto returnAxes = mindspore::lite::MindIR_SliceFusion_GetAxes(primitive.get());
    auto returnAxesSize = returnAxes.size();
    for (size_t i = 0; i < returnAxesSize; ++i) {
        EXPECT_EQ(returnAxes[i], expectAxesValue[i]);
    }
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
