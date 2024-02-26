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

#include "ops/exp_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ExpBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveBase(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveScale(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveShift(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    ExpBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2, 3, 4};
    std::vector<int32_t> m_dim {1, 2, 2, 1};
    std::vector<int32_t> m_paramDim {};
};

void ExpBuilderTest::SetUp() {}

void ExpBuilderTest::TearDown() {}

void ExpBuilderTest::SaveBase(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> baseTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* baseValue = new (std::nothrow) float [1]{-1.0f};
    EXPECT_NE(nullptr, baseValue);
    baseTensor->SetBuffer(baseValue, sizeof(float));
    m_allTensors.emplace_back(baseTensor);
}

void ExpBuilderTest::SaveScale(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* scaleValue = new (std::nothrow) float [1]{1.0f};
    EXPECT_NE(nullptr, scaleValue);
    scaleTensor->SetBuffer(scaleValue, sizeof(float));
    m_allTensors.emplace_back(scaleTensor);
}

void ExpBuilderTest::SaveShift(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> shiftTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* shiftValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, shiftValue);
    shiftTensor->SetBuffer(shiftValue, sizeof(float));
    m_allTensors.emplace_back(shiftTensor);
}

/**
 * @tc.name: exp_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: exp_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: exp_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_003, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3, 4, 5};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_004, TestSize.Level1)
{
    m_outputs = {1, 2};
    m_params = {3, 4, 5};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid base's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> baseTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_EXP_BASE);
    int64_t* baseValue = new (std::nothrow) int64_t [1]{-1};
    EXPECT_NE(nullptr, baseValue);
    baseTensor->SetBuffer(baseValue, sizeof(int64_t));
    m_allTensors.emplace_back(baseTensor);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    baseTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: exp_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid scale's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    
    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_EXP_SCALE);
    int64_t* scaleValue = new (std::nothrow) int64_t [1]{1};
    EXPECT_NE(nullptr, scaleValue);
    scaleTensor->SetBuffer(scaleValue, sizeof(int64_t));
    m_allTensors.emplace_back(scaleTensor);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    scaleTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: exp_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid base's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    std::shared_ptr<NNTensor> shiftTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_EXP_SHIFT);
    int64_t* shiftValue = new (std::nothrow) int64_t [1]{0};
    EXPECT_NE(nullptr, shiftValue);
    shiftTensor->SetBuffer(shiftValue, sizeof(int64_t));
    m_allTensors.emplace_back(shiftTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    shiftTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: exp_build_010
 * @tc.desc: Verify that the build function returns a failed message with passing invalid base param.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_010, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_011
 * @tc.desc: Verify that the build function returns a failed message with passing invalid scale param.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_012
 * @tc.desc: Verify that the build function returns a failed message with passing invalid shift param.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_012, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_013
 * @tc.desc: Verify that the build function returns a failed message without set buffer for base.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_013, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> baseTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_EXP_BASE);
    m_allTensors.emplace_back(baseTensor);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_014
 * @tc.desc: Verify that the build function returns a failed message without set buffer for scale.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_014, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_EXP_SCALE);
    m_allTensors.emplace_back(scaleTensor);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);


    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_build_015
 * @tc.desc: Verify that the build function returns a failed message without set buffer for shift.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_build_015, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    std::shared_ptr<NNTensor> shiftTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_EXP_SHIFT);
    m_allTensors.emplace_back(shiftTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: exp_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveBase(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_BASE);
    SaveScale(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SCALE);
    SaveShift(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_EXP_SHIFT);

    float baseValue = -1.0f;
    float scaleValue = 1.0f;
    float shiftValue = 0.0f;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnBaseValue = mindspore::lite::MindIR_ExpFusion_GetBase(primitive.get());
    EXPECT_EQ(returnBaseValue, baseValue);
    auto returnScaleValue = mindspore::lite::MindIR_ExpFusion_GetScale(primitive.get());
    EXPECT_EQ(returnScaleValue, scaleValue);
    auto returnShiftValue = mindspore::lite::MindIR_ExpFusion_GetShift(primitive.get());
    EXPECT_EQ(returnShiftValue, shiftValue);
}

/**
 * @tc.name: exp_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(ExpBuilderTest, exp_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}