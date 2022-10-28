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

#include "frameworks/native/ops/strided_slice_builder.h"

#include <gtest/gtest.h>
#include "frameworks/native/nn_tensor.h"
#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class StridedSliceBuilderTest : public OpsTest {
protected:
    void InitTensor(const std::vector<uint32_t>& inputsIndex,
        const std::vector<uint32_t>& outputsIndex) override;
    void SaveBeginMaskTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveEndMaskTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveEllipsisMaskTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveNewAxisTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveShrinkAxisTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void InitParams();

protected:
    StridedSliceBuilder m_builder;
    int64_t m_expectBeginMaskValue {0};
    int64_t m_expectEndMaskValue {0};
    int64_t m_expectEllipsisMaskValue {0};
    int64_t m_expectNewAxisMaskValue {0};
    int64_t m_expectShrinkAxisMaskValue {0};
};

void StridedSliceBuilderTest::SaveBeginMaskTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> beginMaskTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* beginMaskValue = new (std::nothrow) int64_t[1]{0};
    EXPECT_NE(nullptr, beginMaskValue);
    beginMaskTensor->SetBuffer(beginMaskValue, sizeof(int64_t));
    m_allTensors.emplace_back(beginMaskTensor);
    m_expectBeginMaskValue = *beginMaskValue;
}

void StridedSliceBuilderTest::SaveEndMaskTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> endMaskTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* endMaskValue = new (std::nothrow) int64_t[1]{0};
    EXPECT_NE(nullptr, endMaskValue);
    endMaskTensor->SetBuffer(endMaskValue, sizeof(int64_t));
    m_allTensors.emplace_back(endMaskTensor);
    m_expectEndMaskValue = *endMaskValue;
}

void StridedSliceBuilderTest::SaveEllipsisMaskTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> ellipsisMaskTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* ellipsisMaskValue = new (std::nothrow) int64_t[1]{0};
    EXPECT_NE(nullptr, ellipsisMaskValue);
    ellipsisMaskTensor->SetBuffer(ellipsisMaskValue, sizeof(int64_t));
    m_allTensors.emplace_back(ellipsisMaskTensor);
    m_expectEllipsisMaskValue = *ellipsisMaskValue;
}

void StridedSliceBuilderTest::SaveNewAxisTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* newAxisMaskValue = new (std::nothrow) int64_t[1]{0};
    EXPECT_NE(nullptr, newAxisMaskValue);
    axisTensor->SetBuffer(newAxisMaskValue, sizeof(int64_t));
    m_allTensors.emplace_back(axisTensor);
    m_expectNewAxisMaskValue = *newAxisMaskValue;
}

void StridedSliceBuilderTest::SaveShrinkAxisTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> shrinkAxisMaskTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* shrinkAxisMaskValue = new (std::nothrow) int64_t[1]{0};
    EXPECT_NE(nullptr, shrinkAxisMaskValue);
    shrinkAxisMaskTensor->SetBuffer(shrinkAxisMaskValue, sizeof(int64_t));
    m_allTensors.emplace_back(shrinkAxisMaskTensor);
    m_expectShrinkAxisMaskValue = *shrinkAxisMaskValue;
}

void StridedSliceBuilderTest::InitTensor(const std::vector<uint32_t>& inputsIndex,
    const std::vector<uint32_t>& outputsIndex)
{
    std::vector<uint32_t> paramsIndex = { 5, 6, 7, 8, 9 };
    std::vector<int32_t> inputDim = {3, 2, 3};
    std::vector<int32_t> OutputDim = {1, 2, 2};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
}

void StridedSliceBuilderTest::InitParams()
{
    std::vector<int32_t> paramDim = {};
    SaveBeginMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_BEGIN_MASK);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);
}

/**
 * @tc.name: stridedslice_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_001, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };

    InitTensor(inputsIndex, outputsIndex);
    InitParams();

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: stridedslice_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };

    InitTensor(inputsIndex, outputsIndex);
    InitParams();

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: stridedslice_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_003, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    std::vector<uint32_t> outputsIndex = { 10 };
    std::vector<uint32_t> paramsIndex = { 11, 12, 13, 14, 15 };
    std::vector<int32_t> inputDim = {3, 2, 3};
    std::vector<int32_t> OutputDim = {1, 2, 2};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
    InitParams();

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_004, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4, 5 };
    std::vector<uint32_t> paramsIndex = { 6, 7, 8, 9, 10 };
    std::vector<int32_t> inputDim = {3, 2, 3};
    std::vector<int32_t> OutputDim = {1, 2, 2};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
    InitParams();

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_005
 * @tc.desc: Provide empty input, output, and parameters to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_005, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
    std::vector<uint32_t> outputsIndex = { 9 };
    std::vector<uint32_t> paramsIndex = { 10, 11, 12, 13, 14 };

    OH_NN_ReturnCode ret = m_builder.Build(paramsIndex, inputsIndex, outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_006
 * @tc.desc:Provide empty output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_006, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = {};
    std::vector<uint32_t> paramsIndex = { 4, 5, 6, 7, 8 };
    std::vector<int32_t> inputDim = {3, 2, 3};
    std::vector<int32_t> OutputDim = {1, 2, 2};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
    InitParams();

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_007
 * @tc.desc: Provide beginmask param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_007, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<uint32_t> paramsIndex = { 5, 6, 7, 8, 9 };
    std::vector<int32_t> inputDim = {3, 2, 3};
    std::vector<int32_t> OutputDim = {1, 2, 2};
    std::vector<int32_t> paramDim = {};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
    SaveBeginMaskTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_STRIDED_SLICE_BEGIN_MASK);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_008
 * @tc.desc: Provide endmask param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_008, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveEndMaskTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_009
 * @tc.desc: Provide ellipsismask param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_009, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_010
 * @tc.desc: Provide axis param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_010, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_011
 * @tc.desc: Provide shrinkaxis param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_011, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}


/**
 * @tc.name: stridedslice_build_012
 * @tc.desc: Provide begin mask parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_012, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    std::shared_ptr<NNTensor> beginMaskTensor = TransToNNTensor(OH_NN_INT64, paramDim, nullptr,
        OH_NN_STRIDED_SLICE_BEGIN_MASK);
    beginMaskTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(beginMaskTensor);

    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_013
 * @tc.desc: Provide end mask parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_013, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveBeginMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_BEGIN_MASK);

    std::shared_ptr<NNTensor> endMaskTensor = TransToNNTensor(OH_NN_INT64, paramDim, nullptr,
        OH_NN_STRIDED_SLICE_END_MASK);
    endMaskTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(endMaskTensor);

    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_014
 * @tc.desc: Provide ellipsis mask parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_014, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveBeginMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_BEGIN_MASK);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);

    std::shared_ptr<NNTensor> ellipsisMaskTensor = TransToNNTensor(OH_NN_INT64, paramDim, nullptr,
        OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    ellipsisMaskTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(ellipsisMaskTensor);

    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_015
 * @tc.desc: Provide new axis parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_015, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveBeginMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_BEGIN_MASK);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);

    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_INT64, paramDim, nullptr,
        OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    axisTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(axisTensor);

    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_016
 * @tc.desc: Provide shrink axis parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_016, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveBeginMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_BEGIN_MASK);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);

    std::shared_ptr<NNTensor> shrinkAxisTensor = TransToNNTensor(OH_NN_INT64, paramDim, nullptr,
        OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK);
    shrinkAxisTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(shrinkAxisTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_build_017
 * @tc.desc: Provide invalid parameter type to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_build_017, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveBeginMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_BEGIN_MASK);
    SaveEndMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_END_MASK);
    SaveEllipsisMaskTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_ELLIPSIS_MASK);
    SaveNewAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_STRIDED_SLICE_NEW_AXIS_MASK);
    SaveShrinkAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SCALE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: stridedslice_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = { nullptr, DestroyLiteGraphPrimitive };
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: stridedslice_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(StridedSliceBuilderTest, stridedslice_get_primitive_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };

    InitTensor(inputsIndex, outputsIndex);
    InitParams();

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = { nullptr, DestroyLiteGraphPrimitive };
    EXPECT_NE(primitive, expectPrimitive);

    auto beginMaskReturn = mindspore::lite::MindIR_StridedSlice_GetBeginMask(primitive.get());
    EXPECT_EQ(beginMaskReturn, m_expectBeginMaskValue);
    auto endMaskReturn = mindspore::lite::MindIR_StridedSlice_GetEndMask(primitive.get());
    EXPECT_EQ(endMaskReturn, m_expectEndMaskValue);
    auto ellipsisMaskReturn = mindspore::lite::MindIR_StridedSlice_GetEllipsisMask(primitive.get());
    EXPECT_EQ(ellipsisMaskReturn, m_expectEllipsisMaskValue);
    auto newAxisMaskReturn = mindspore::lite::MindIR_StridedSlice_GetNewAxisMask(primitive.get());
    EXPECT_EQ(newAxisMaskReturn, m_expectNewAxisMaskValue);
    auto shrinkAxisMaskReturn = mindspore::lite::MindIR_StridedSlice_GetShrinkAxisMask(primitive.get());
    EXPECT_EQ(shrinkAxisMaskReturn, m_expectShrinkAxisMaskValue);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
