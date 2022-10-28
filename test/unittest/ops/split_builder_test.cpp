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

#include "frameworks/native/ops/split_builder.h"

#include <gtest/gtest.h>
#include "frameworks/native/nn_tensor.h"
#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class SplitBuilderTest : public OpsTest {
protected:
    void InitTensor(const std::vector<uint32_t>& inputsIndex,
        const std::vector<uint32_t>& outputsIndex) override;
    void SaveAxisTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveOutputNumTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveSizeSplitsTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    SplitBuilder m_builder;
    int64_t m_expectOutputNum {0};
    int64_t m_expectAxis {0};
    std::vector<int64_t> m_expectSizeSplitsValue;
};

void SplitBuilderTest::InitTensor(const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex)
{
    std::vector<uint32_t> paramsIndex = { 3, 4, 5 };
    std::vector<int32_t> inputDim = {2, 4};
    std::vector<int32_t> OutputDim = {1, 4, 0, 0};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
}

void SplitBuilderTest::SaveOutputNumTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> outputNumTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* outputNumValue = new (std::nothrow) int64_t[1]{2};
    EXPECT_NE(nullptr, outputNumValue);
    outputNumTensor->SetBuffer(outputNumValue, sizeof(int64_t));
    m_allTensors.emplace_back(outputNumTensor);
    m_expectOutputNum = *outputNumValue;
}

void SplitBuilderTest::SaveSizeSplitsTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    const int sizeSplitsLen = 2;
    std::shared_ptr<NNTensor> sizeSplitsTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* sizeSplitsValue = new (std::nothrow) int64_t[sizeSplitsLen] {0, 0};
    EXPECT_NE(nullptr, sizeSplitsValue);
    sizeSplitsTensor->SetBuffer(sizeSplitsValue, sizeof(int64_t) * sizeSplitsLen);
    m_allTensors.emplace_back(sizeSplitsTensor);
    m_expectSizeSplitsValue.assign(sizeSplitsValue, sizeSplitsValue + sizeSplitsLen);
}

void SplitBuilderTest::SaveAxisTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axisValue = new (std::nothrow) int64_t[1]{0};
    EXPECT_NE(nullptr, axisValue);
    axisTensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(axisTensor);
    m_expectAxis = *axisValue;
}

/**
 * @tc.name: split_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_001, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);

    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: split_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: split_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_003, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2, 3 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_004, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2, 3 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);

    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_005
 * @tc.desc: Provide empty input, output, and parameters to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_005, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4, 5 };
    std::vector<uint32_t> paramsIndex = { 6, 7, 8 };
    OH_NN_ReturnCode ret = m_builder.Build(paramsIndex, inputsIndex, outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_006
 * @tc.desc: Provide axis param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_006, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_007
 * @tc.desc: Provide axis param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_007, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_008
 * @tc.desc: Provide size splits param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_008, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_009
 * @tc.desc: Provide output num param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_009, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_010
 * @tc.desc: Provide axis param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_010, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_BOOL, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_011
 * @tc.desc: Provide axis parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_011, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);

    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    axisTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(axisTensor);

    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_012
 * @tc.desc: Provide invalid parameter type to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_012, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SCALE_AXIS);
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_013
 * @tc.desc: Provide axis parameter not scalar to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_013, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};
    std::vector<int32_t> axisDim = {1, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, axisDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_014
 * @tc.desc: Provide output parameter not scalar to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_014, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};
    std::vector<int32_t> outputNumDim = {1, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, outputNumDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_build_015
 * @tc.desc: Provide empty output and param to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_build_015, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 1 };
    std::vector<uint32_t> outputsIndex = {};
    std::vector<uint32_t> paramsIndex = {};
    std::vector<int32_t> inputDim = {2, 4};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: split_getprimitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_getprimitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: split_getprimitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(SplitBuilderTest, split_getprimitive_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_AXIS);
    SaveOutputNumTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_OUTPUT_NUM);
    SaveSizeSplitsTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SPLIT_SIZE_SPLITS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnValue = mindspore::lite::MindIR_Split_GetSizeSplits(primitive.get());
    auto returnValueSize = returnValue.size();
    for (size_t i = 0; i < returnValueSize; ++i) {
        EXPECT_EQ(returnValue[i], m_expectSizeSplitsValue[i]);
    }

    auto returnOutputNum = mindspore::lite::MindIR_Split_GetOutputNum(primitive.get());
    EXPECT_EQ(returnOutputNum, m_expectOutputNum);

    auto returnAxis = mindspore::lite::MindIR_Split_GetAxis(primitive.get());
    EXPECT_EQ(returnAxis, m_expectAxis);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
