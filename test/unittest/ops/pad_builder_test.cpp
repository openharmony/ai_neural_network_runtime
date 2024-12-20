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

#include "ops/pad_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class PadBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SetConstValueTensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetPaddingModeTensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    PadBuilder m_pad;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {3, 4};
    std::vector<int32_t> m_inputDim {1, 1, 2, 3};
    std::vector<int32_t> m_outputDim {1, 2, 7, 7};
    std::vector<int32_t> m_paramDim {};
};

void PadBuilderTest::SetUp() {}

void PadBuilderTest::TearDown() {}

void PadBuilderTest::SetConstValueTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> constantValueTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* constantValue = new (std::nothrow) float(2.0);
    EXPECT_NE(nullptr, constantValue);
    constantValueTensor->SetBuffer(constantValue, sizeof(float));
    m_allTensors.emplace_back(constantValueTensor);
}

void PadBuilderTest::SetPaddingModeTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> paddingModeValueTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int32_t* paddingModeValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, paddingModeValue);
    paddingModeValueTensor->SetBuffer(paddingModeValue, sizeof(int32_t));
    m_allTensors.emplace_back(paddingModeValueTensor);
}

/**
 * @tc.name: pad_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_PAD_CONSTANT_VALUE);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: pad_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_PAD_CONSTANT_VALUE);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    EXPECT_EQ(OH_NN_SUCCESS, m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: pad_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2};
    m_outputs = {3};
    m_params = {4, 5};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_PAD_CONSTANT_VALUE);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pad_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_004, TestSize.Level0)
{
    m_outputs = {2, 3};
    m_params = {4, 5};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_PAD_CONSTANT_VALUE);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pad_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pad_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pad_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid constant's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_007, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    std::shared_ptr<NNTensor> constantValueTensor = TransToNNTensor(OH_NN_INT32, m_paramDim,
        nullptr, OH_NN_PAD_CONSTANT_VALUE);
    int32_t constantValue = 0;
    constantValueTensor->SetBuffer(&constantValue, sizeof(constantValue));
    m_allTensors.emplace_back(constantValueTensor);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    constantValueTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: pad_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid paddingMode's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_008, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_PAD_CONSTANT_VALUE);
    std::shared_ptr<NNTensor> paddingModeTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_PAD_PADDING_MODE);
    int64_t paddingMode = 0;
    paddingModeTensor->SetBuffer(&paddingMode, sizeof(int64_t));
    m_allTensors.emplace_back(paddingModeTensor);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    paddingModeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: pad_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid constant's dimension.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_009, TestSize.Level0)
{
    m_paramDim = {2};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> constantValueTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_PAD_CONSTANT_VALUE);
    float constantValue[2] = {2.0, 2.0};
    constantValueTensor->SetBuffer(constantValue, 2 * sizeof(float));
    m_allTensors.emplace_back(constantValueTensor);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    constantValueTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: pad_build_010
 * @tc.desc: Verify that the build function returns a failed message with passing invalid constvalue.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_010, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_ONE_HOT_AXIS);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pad_build_011
 * @tc.desc: Verify that the build function returns a failed message with passing invalid paddingMode.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_011, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_PAD_CONSTANT_VALUE);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_ONE_HOT_AXIS);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pad_build_012
 * @tc.desc: Verify that the build function returns a failed message without set buffer for constantValue.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_012, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    std::shared_ptr<NNTensor> constantValueTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_PAD_CONSTANT_VALUE);
    m_allTensors.emplace_back(constantValueTensor);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pad_build_013
 * @tc.desc: Verify that the build function returns a failed message without set buffer for paddingMode.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_build_013, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_PAD_CONSTANT_VALUE);
    std::shared_ptr<NNTensor> paddingModeTensor = TransToNNTensor(OH_NN_INT32, m_paramDim,
        nullptr, OH_NN_PAD_PADDING_MODE);
    m_allTensors.emplace_back(paddingModeTensor);

    OH_NN_ReturnCode ret = m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: pad_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_getprimitive_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SetConstValueTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_PAD_CONSTANT_VALUE);
    SetPaddingModeTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_PAD_PADDING_MODE);

    float constantValue = 2.0;
    mindspore::lite::PaddingMode paddingModeValue = mindspore::lite::PADDING_MODE_CONSTANT;
    EXPECT_EQ(OH_NN_SUCCESS, m_pad.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_pad.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnValue = mindspore::lite::MindIR_PadFusion_GetConstantValue(primitive.get());
    EXPECT_EQ(returnValue, constantValue);
    auto returnPaddingMode = mindspore::lite::MindIR_PadFusion_GetPaddingMode(primitive.get());
    EXPECT_EQ(returnPaddingMode, paddingModeValue);
}

/**
 * @tc.name: pad_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(PadBuilderTest, pad_getprimitive_002, TestSize.Level0)
{
    PadBuilder pad;
    LiteGraphPrimitvePtr primitive = m_pad.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}