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

#include "ops/pow_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class PowBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveShift(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveScale(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    PowBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {3, 4};
    std::vector<int32_t> m_dim {1, 2, 2, 1};
    std::vector<int32_t> m_shiftDim {1};
    std::vector<int32_t> m_scaleDim {1};
};

void PowBuilderTest::SetUp() {}

void PowBuilderTest::TearDown() {}

void PowBuilderTest::SaveShift(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> shiftTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* shiftValue = new (std::nothrow) float[1] {0.0f};
    EXPECT_NE(nullptr, shiftValue);
    shiftTensor->SetBuffer(shiftValue, sizeof(float));
    m_allTensors.emplace_back(shiftTensor);
}

void PowBuilderTest::SaveScale(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* scaleValue = new (std::nothrow) float[1] {1.0f};
    EXPECT_NE(nullptr, scaleValue);
    scaleTensor->SetBuffer(scaleValue, sizeof(float));
    m_allTensors.emplace_back(scaleTensor);
}

/**
 * @tc.name: pow_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_POW_SHIFT);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_POW_SCALE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: pow_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_POW_SHIFT);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_POW_SCALE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: pow_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_003, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {3};
    m_params = {4, 5};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_POW_SHIFT);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_POW_SCALE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pow_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_004, TestSize.Level1)
{
    m_outputs = {2, 3};
    m_params = {4, 5};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_POW_SHIFT);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_POW_SCALE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pow_build_005
 * @tc.desc: Verify that the build function return a failed message with null allTensor
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pow_build_006
 * @tc.desc: Verify that the build function return a failed message without output tensor
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pow_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid shift's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> shiftTensor = TransToNNTensor(OH_NN_INT64, m_shiftDim,
        nullptr, OH_NN_POW_SHIFT);
    int64_t* shiftValue = new (std::nothrow) int64_t[1] {0};
    shiftTensor->SetBuffer(shiftValue, sizeof(shiftValue));
    m_allTensors.emplace_back(shiftTensor);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_POW_SCALE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    shiftTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: pow_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid scale's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_POW_SHIFT);
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(OH_NN_INT64, m_scaleDim,
        nullptr, OH_NN_POW_SCALE);
    int64_t* scaleValue = new (std::nothrow) int64_t[1] {1};
    scaleTensor->SetBuffer(scaleValue, sizeof(scaleValue));
    m_allTensors.emplace_back(scaleTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    scaleTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: pow_build_009
 * @tc.desc: Verify that the build function returns a failed message with passing invalid shift param.
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_POW_SCALE);;

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pow_build_010
 * @tc.desc: Verify that the build function returns a failed message with passing invalid scale param.
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_010, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_POW_SHIFT);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pow_build_011
 * @tc.desc: Verify that the build function returns a failed message without set buffer for shift.
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> shiftTensor = TransToNNTensor(OH_NN_FLOAT32, m_shiftDim,
        nullptr, OH_NN_POW_SHIFT);
    m_allTensors.emplace_back(shiftTensor);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_POW_SCALE);;

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pow_build_012
 * @tc.desc: Verify that the build function returns a failed message without set buffer for scale.
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_build_012, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_POW_SHIFT);
    std::shared_ptr<NNTensor> scaleTensor = TransToNNTensor(OH_NN_FLOAT32, m_scaleDim,
        nullptr, OH_NN_POW_SCALE);
    m_allTensors.emplace_back(scaleTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pow_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_get_primitive_001, TestSize.Level1)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: pow_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(PowBuilderTest, pow_get_primitive_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveShift(OH_NN_FLOAT32, m_shiftDim, nullptr, OH_NN_POW_SHIFT);
    SaveScale(OH_NN_FLOAT32, m_scaleDim, nullptr, OH_NN_POW_SCALE);

    float shiftValue = 0.0f;
    float scaleValue = 1.0f;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr powPrimitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(powPrimitive, expectPrimitive);

    auto returnShiftValue = mindspore::lite::MindIR_PowFusion_GetShift(powPrimitive.get());
    EXPECT_EQ(shiftValue, returnShiftValue);
    auto returnScaleValue = mindspore::lite::MindIR_PowFusion_GetScale(powPrimitive.get());
    EXPECT_EQ(scaleValue, returnScaleValue);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS