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

#include "frameworks/native/ops/batchnorm_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class BatchNormBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveParamsTensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    BatchNormBuilder m_batchNorm;
    std::vector<uint32_t> m_inputs {0, 1, 2, 3, 4};
    std::vector<uint32_t> m_outputs {5};
    std::vector<uint32_t> m_params {6};
    std::vector<int32_t> m_inputDim {2, 2};
    std::vector<int32_t> m_outputDim {2, 2};
    std::vector<int32_t> m_paramDim {};
};

void BatchNormBuilderTest::SetUp() {}

void BatchNormBuilderTest::TearDown() {}

void BatchNormBuilderTest::SaveParamsTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float *epsilonValue = new (std::nothrow) float(0.0f);
    EXPECT_NE(nullptr, epsilonValue);
    epsilonTensor->SetBuffer(epsilonValue, sizeof(float));
    m_allTensors.emplace_back(epsilonTensor);
}

/**
 * @tc.name: batchnorm_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_BATCH_NORM_EPSILON);

    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: batchnorm_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_BATCH_NORM_EPSILON);

    EXPECT_EQ(OH_NN_SUCCESS, m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: batchnorm_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2, 3, 4, 5};
    m_outputs = {6};
    m_params = {7};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_BATCH_NORM_EPSILON);

    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: batchnorm_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_004, TestSize.Level0)
{
    m_outputs = {5, 6};
    m_params = {7};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_BATCH_NORM_EPSILON);

    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: batchnorm_build_005
 * @tc.desc: Verify that the build function returns a failed message with null allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: batchnorm_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: batchnorm_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid epsilon's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_007, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(OH_NN_INT32, m_paramDim,
        nullptr, OH_NN_BATCH_NORM_EPSILON);
    float epsilonValue = 0.0f;
    epsilonTensor->SetBuffer(&epsilonValue, sizeof(epsilonValue));
    m_allTensors.emplace_back(epsilonTensor);

    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    epsilonTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: batchnorm_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid epsilon's dimension.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_008, TestSize.Level0)
{
    std::vector<int32_t> m_paramDim = { 2 };

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_BATCH_NORM_EPSILON);
    float epsilonValue[2] = {0.0f, 0.0f};
    epsilonTensor->SetBuffer(epsilonValue, 2 * sizeof(float));
    m_allTensors.emplace_back(epsilonTensor);

    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    epsilonTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: batchnorm_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid param.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_009, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);

    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: batchnorm_build_010
 * @tc.desc: Verify that the build function returns a failed message without set buffer successfully.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_build_010, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    std::shared_ptr<NNTensor> epsilonTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim, nullptr,
        OH_NN_BATCH_NORM_EPSILON);
    m_allTensors.emplace_back(epsilonTensor);

    OH_NN_ReturnCode ret = m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: batchnorm_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_getprimitive_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveParamsTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_BATCH_NORM_EPSILON);

    float epsilonValue = 0.9;
    EXPECT_EQ(OH_NN_SUCCESS, m_batchNorm.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_batchNorm.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnValue = mindspore::lite::MindIR_FusedBatchNorm_GetEpsilon(primitive.get());
    EXPECT_EQ(returnValue, epsilonValue);
}

/**
 * @tc.name: batchnorm_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(BatchNormBuilderTest, batchnorm_getprimitive_002, TestSize.Level0)
{
    BatchNormBuilder batchNorm;
    LiteGraphPrimitvePtr primitive = m_batchNorm.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}