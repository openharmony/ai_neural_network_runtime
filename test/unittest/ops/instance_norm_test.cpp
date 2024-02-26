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

#include "ops/instance_norm_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class InstanceNormBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveParamsTensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetInputTensor();

protected:
    InstanceNormBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1, 2};
    std::vector<uint32_t> m_outputs {3};
    std::vector<uint32_t> m_params {4};
    std::vector<int32_t> m_inputDim {2, 2, 2, 2};
    std::vector<int32_t> m_scaleAndBiasDim {1};
    std::vector<int32_t> m_outputDim {2, 2, 2, 2};
    std::vector<int32_t> m_paramDim {};
};

void InstanceNormBuilderTest::SetUp() {}

void InstanceNormBuilderTest::TearDown() {}

void InstanceNormBuilderTest::SaveParamsTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* epsilonValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, epsilonValue);
    epsilonTensor->SetBuffer(epsilonValue, sizeof(float));
    m_allTensors.emplace_back(epsilonTensor);
}

void InstanceNormBuilderTest::SetInputTensor()
{
    m_inputsIndex = m_inputs;
    std::shared_ptr<NNTensor> inputTensor;
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    std::shared_ptr<NNTensor> scaleTensor;
    scaleTensor = TransToNNTensor(OH_NN_FLOAT32, m_scaleAndBiasDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(scaleTensor);

    std::shared_ptr<NNTensor> biasTensor;
    biasTensor = TransToNNTensor(OH_NN_FLOAT32, m_scaleAndBiasDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(biasTensor);
}

/**
 * @tc.name: instance_norm_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_001, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_INSTANCE_NORM_EPSILON);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: instance_norm_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_002, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_INSTANCE_NORM_EPSILON);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: instance_norm_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_003, TestSize.Level1)
{
    m_inputs = {0, 1, 2, 3};
    m_outputs = {4};
    m_params = {5};

    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_INSTANCE_NORM_EPSILON);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: instance_norm_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_004, TestSize.Level1)
{
    m_outputs = {3, 4};
    m_params = {5};

    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_INSTANCE_NORM_EPSILON);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: instance_norm_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: instance_norm_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_006, TestSize.Level1)
{
    SetInputTensor();

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: instance_norm_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid epsilon's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_007, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_INSTANCE_NORM_EPSILON);
    int64_t* epsilonValue = new (std::nothrow) int64_t [1]{0.0f};
    EXPECT_NE(nullptr, epsilonValue);
    epsilonTensor->SetBuffer(epsilonValue, sizeof(epsilonValue));
    m_allTensors.emplace_back(epsilonTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    epsilonTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: instance_norm_build_008
 * @tc.desc: Verify that the build function returns a failed message with passing invalid epsilon param.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_008, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: instance_norm_build_009
 * @tc.desc: Verify that the build function returns a failed message without set buffer for epsilon.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_build_009, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_INSTANCE_NORM_EPSILON);
    m_allTensors.emplace_back(epsilonTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: instance_norm_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_getprimitive_001, TestSize.Level1)
{
    SetInputTensor();
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_INSTANCE_NORM_EPSILON);

    float epsilonValue = 0.0f;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnEpsilonValue = mindspore::lite::MindIR_InstanceNorm_GetEpsilon(primitive.get());
    EXPECT_EQ(returnEpsilonValue, epsilonValue);
}

/**
 * @tc.name: instance_norm_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(InstanceNormBuilderTest, instance_norm_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}