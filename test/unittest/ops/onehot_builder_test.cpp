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

#include "frameworks/native/ops/onehot_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class OneHotBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveParamsTensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    OnehotBuilder m_oneHot;
    std::vector<uint32_t> m_inputs {0, 1, 2, 3};
    std::vector<uint32_t> m_outputs {4};
    std::vector<uint32_t> m_params {5};
    std::vector<int32_t> m_inputDim {3};
    std::vector<int32_t> m_outputDim {3, 3};
    std::vector<int32_t> m_paramDim {};
};

void OneHotBuilderTest::SetUp() {}

void OneHotBuilderTest::TearDown() {}

void OneHotBuilderTest::SaveParamsTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axisValue = new (std::nothrow) int64_t(-1);
    EXPECT_NE(nullptr, axisValue);
    axisTensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(axisTensor);
}

/**
 * @tc.name: onehot_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_ONE_HOT_AXIS);

    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: onehot_build_001
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_ONE_HOT_AXIS);

    EXPECT_EQ(OH_NN_SUCCESS, m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: onehot_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2, 3, 4};
    m_outputs = {5};
    m_params = {6};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_ONE_HOT_AXIS);

    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: onehot_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_004, TestSize.Level0)
{
    m_outputs = {4, 5};
    m_params = {6};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_ONE_HOT_AXIS);

    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: onehot_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: onehot_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: onehot_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid axis's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_007, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_ONE_HOT_AXIS);
    float axisValue = 1e-7;
    axisTensor->SetBuffer(&axisValue, sizeof(axisValue));
    m_allTensors.emplace_back(axisTensor);

    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    axisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: onehot_build_008
 * @tc.desc: Verify that the build function returns a failed message with passing invalid param.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_008, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: onehot_build_009
 * @tc.desc: Verify that the build function returns a failed message without set buffer for axis.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_build_009, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_ONE_HOT_AXIS);
    m_allTensors.emplace_back(axisTensor);

    OH_NN_ReturnCode ret = m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: onehot_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_getprimitive_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_ONE_HOT_AXIS);

    int64_t axisValue = -1;
    EXPECT_EQ(OH_NN_SUCCESS, m_oneHot.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_oneHot.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnValue = mindspore::lite::MindIR_OneHot_GetAxis(primitive.get());
    EXPECT_EQ(returnValue, axisValue);
}

/**
 * @tc.name: onehot_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(OneHotBuilderTest, onehot_getprimitive_002, TestSize.Level0)
{
    OnehotBuilder oneHot;
    LiteGraphPrimitvePtr primitive = m_oneHot.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}