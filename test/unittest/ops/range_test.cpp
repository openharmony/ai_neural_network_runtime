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

#include "ops/range_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class RangeBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveStart(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveLimit(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveDelta(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    RangeBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2, 3, 4};
    std::vector<int32_t> m_dim {3};
    std::vector<int32_t> m_paramDim {};
};

void RangeBuilderTest::SetUp() {}

void RangeBuilderTest::TearDown() {}

void RangeBuilderTest::SaveStart(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> startTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* startValue = new (std::nothrow) int64_t [1]{0};
    EXPECT_NE(nullptr, startValue);
    startTensor->SetBuffer(startValue, sizeof(int64_t));
    m_allTensors.emplace_back(startTensor);
}

void RangeBuilderTest::SaveLimit(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> limitTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* limitValue = new (std::nothrow) int64_t [1]{3};
    EXPECT_NE(nullptr, limitValue);
    limitTensor->SetBuffer(limitValue, sizeof(int64_t));
    m_allTensors.emplace_back(limitTensor);
}

void RangeBuilderTest::SaveDelta(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> deltaTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* deltaValue = new (std::nothrow) int64_t [1]{1};
    EXPECT_NE(nullptr, deltaValue);
    deltaTensor->SetBuffer(deltaValue, sizeof(int64_t));
    m_allTensors.emplace_back(deltaTensor);
}

/**
 * @tc.name: range_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: range_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: range_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_003, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3, 4, 5};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_004, TestSize.Level1)
{
    m_outputs = {1, 2};
    m_params = {3, 4, 5};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid start's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> startTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_RANGE_START);
    float* startValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, startValue);
    startTensor->SetBuffer(startValue, sizeof(float));
    m_allTensors.emplace_back(startTensor);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    startTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: range_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid limit's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    std::shared_ptr<NNTensor> limitTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_RANGE_LIMIT);
    float* limitValue = new (std::nothrow) float [1]{3.0f};
    EXPECT_NE(nullptr, limitValue);
    limitTensor->SetBuffer(limitValue, sizeof(float));
    m_allTensors.emplace_back(limitTensor);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    limitTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: range_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid delta's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    std::shared_ptr<NNTensor> deltaTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_RANGE_DELTA);
    float* deltaValue = new (std::nothrow) float [1]{1.0f};
    EXPECT_NE(nullptr, deltaValue);
    deltaTensor->SetBuffer(deltaValue, sizeof(float));
    m_allTensors.emplace_back(deltaTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    deltaTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: range_build_010
 * @tc.desc: Verify that the build function returns a failed message with passing invalid start param.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_010, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_011
 * @tc.desc: Verify that the build function returns a failed message with passing invalid limit param.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);
    
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_012
 * @tc.desc: Verify that the build function returns a failed message with passing invalid delta param.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_012, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_013
 * @tc.desc: Verify that the build function returns a failed message without set buffer for start.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_013, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> startTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_RANGE_START);
    m_allTensors.emplace_back(startTensor);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_014
 * @tc.desc: Verify that the build function returns a failed message without set buffer for limit.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_014, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    std::shared_ptr<NNTensor> limitTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_RANGE_LIMIT);
    m_allTensors.emplace_back(limitTensor);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_build_015
 * @tc.desc: Verify that the build function returns a failed message without set buffer for delta.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_build_015, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    std::shared_ptr<NNTensor> deltaTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_RANGE_DELTA);
    m_allTensors.emplace_back(deltaTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: range_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveStart(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_START);
    SaveLimit(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_LIMIT);
    SaveDelta(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RANGE_DELTA);

    int64_t startValue = 0;
    int64_t limitValue = 3;
    int64_t deltaValue = 1;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnStartValue = mindspore::lite::MindIR_Range_GetStart(primitive.get());
    EXPECT_EQ(returnStartValue, startValue);
    auto returnLimitValue = mindspore::lite::MindIR_Range_GetLimit(primitive.get());
    EXPECT_EQ(returnLimitValue, limitValue);
    auto returnDeltaValue = mindspore::lite::MindIR_Range_GetDelta(primitive.get());
    EXPECT_EQ(returnDeltaValue, deltaValue);
}

/**
 * @tc.name: range_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(RangeBuilderTest, range_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}