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

#include "frameworks/native/ops/matmul_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class MatMulBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveTransposeATensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveTransposeBTensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveActivationTensor(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetInputTensor(std::shared_ptr<NNTensor> inputTensor);

protected:
    MatmulBuilder m_matmul;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {3, 4, 5};
    std::vector<int32_t> m_inputXDim {1, 1, 3, 2};
    std::vector<int32_t> m_inputYDim {1, 1, 2, 3};
    std::vector<int32_t> m_outputDim {1, 1, 3, 3};
    std::vector<int32_t> m_paramDim {};
    std::shared_ptr<NNTensor> m_inputTensor {};
};

void MatMulBuilderTest::SetUp() {}

void MatMulBuilderTest::TearDown() {}

void MatMulBuilderTest::SaveTransposeATensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> transposeATensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool* transposeAValue = new (std::nothrow) bool(false);
    EXPECT_NE(nullptr, transposeAValue);
    transposeATensor->SetBuffer(transposeAValue, sizeof(bool));
    m_allTensors.emplace_back(transposeATensor);
}

void MatMulBuilderTest::SaveTransposeBTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> transposeBTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool* transposeBValue = new (std::nothrow) bool(false);
    EXPECT_NE(nullptr, transposeBValue);
    transposeBTensor->SetBuffer(transposeBValue, sizeof(bool));
    m_allTensors.emplace_back(transposeBTensor);
}

void MatMulBuilderTest::SaveActivationTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> activationTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* activationValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValue);
    activationTensor->SetBuffer(activationValue, sizeof(int8_t));
    m_allTensors.emplace_back(activationTensor);
}

void MatMulBuilderTest::SetInputTensor(std::shared_ptr<NNTensor> inputTensor)
{
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputXDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputYDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);
}
/**
 * @tc.name: matmul_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_001, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: matmul_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_002, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    EXPECT_EQ(OH_NN_SUCCESS, m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: matmul_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2};
    m_outputs = {3};
    m_params = {4, 5, 6};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: matmul_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_004, TestSize.Level0)
{
    m_outputs = {2, 3};
    m_params = {4, 5, 6};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);
    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: matmul_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: matmul_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_006, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: matmul_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid transposeA's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_007, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    std::shared_ptr<NNTensor> transposeATensor;
    transposeATensor = TransToNNTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    int32_t transposeAValue = 1;
    transposeATensor->SetBuffer(&transposeAValue, sizeof(transposeAValue));
    m_allTensors.emplace_back(transposeATensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    transposeATensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: matmul_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid transposeA's dimension.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_008, TestSize.Level0)
{
    std::vector<int32_t> expectParamDim = {2};

    SetInputTensor(m_inputTensor);

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    std::shared_ptr<NNTensor> transposeATensor;
    transposeATensor = TransToNNTensor(OH_NN_BOOL, expectParamDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    bool transposeAValue[2] = {false, false};
    transposeATensor->SetBuffer(transposeAValue, 2 * sizeof(bool));
    m_allTensors.emplace_back(transposeATensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    transposeATensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: matmul_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid transposeB's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_009, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    std::shared_ptr<NNTensor> transposeBTensor;
    transposeBTensor = TransToNNTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    int32_t transposeBValue = 1;
    transposeBTensor->SetBuffer(&transposeBValue, sizeof(transposeBValue));
    m_allTensors.emplace_back(transposeBTensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    transposeBTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: matmul_build_010
 * @tc.desc: Verify that the build function returns a failed message with invalid transposeB's dimension.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matMul_build_010, TestSize.Level0)
{
    std::vector<int32_t> expectParamDim = {2};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    std::shared_ptr<NNTensor> transposeBTensor;
    transposeBTensor = TransToNNTensor(OH_NN_BOOL, expectParamDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    bool transposeBValue[2] = {false, false};
    transposeBTensor->SetBuffer(transposeBValue, 2 * sizeof(bool));
    m_allTensors.emplace_back(transposeBTensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    transposeBTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: matmul_build_011
 * @tc.desc: Verify that the build function returns a failed message with invalid activation's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matMul_build_011, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);

    std::shared_ptr<NNTensor> activationTensor;
    activationTensor = TransToNNTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);
    bool activationValue = false;
    activationTensor->SetBuffer(&activationValue, sizeof(activationValue));
    m_allTensors.emplace_back(activationTensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    activationTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: matmul_build_012
 * @tc.desc: Verify that the build function returns a failed message with invalid activation's dimension.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_012, TestSize.Level0)
{
    std::vector<int32_t> expectParamDim = {2};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);

    std::shared_ptr<NNTensor> activationTensor;
    activationTensor = TransToNNTensor(OH_NN_INT8, expectParamDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);
    int8_t activationValue[2] = {0, 1};
    activationTensor->SetBuffer(activationValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(activationTensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    activationTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: matmul_build_013
 * @tc.desc: Verify that the build function returns a failed message with invalid activation's data.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_013, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);

    std::shared_ptr<NNTensor> activationTensor;
    activationTensor = TransToNNTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);
    int8_t activationValue = -1;
    activationTensor->SetBuffer(&activationValue, sizeof(activationValue));
    m_allTensors.emplace_back(activationTensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    activationTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: matmul_build_014
 * @tc.desc: Verify that the build function returns a failed message with passing invalid param.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_014, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: matmul_build_015
 * @tc.desc: Verify that the build function returns a failed message without set buffer for transposeA.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_015, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    std::shared_ptr<NNTensor> transposeATensor;
    transposeATensor = TransToNNTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    m_allTensors.emplace_back(transposeATensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: matmul_build_016
 * @tc.desc: Verify that the build function returns a failed message without set buffer for transposeB.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_016, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    std::shared_ptr<NNTensor> transposeBTensor;
    transposeBTensor = TransToNNTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    m_allTensors.emplace_back(transposeBTensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: matmul_build_017
 * @tc.desc: Verify that the build function returns a failed message without set buffer for activation.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_build_017, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);

    std::shared_ptr<NNTensor> activationTensor;
    activationTensor = TransToNNTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);
    m_allTensors.emplace_back(activationTensor);

    OH_NN_ReturnCode ret = m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: matmul_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_getprimitive_001, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveTransposeATensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_A);
    SaveTransposeBTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MATMUL_TRANSPOSE_B);
    SaveActivationTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_MATMUL_ACTIVATION_TYPE);

    bool transposeAValue = false;
    bool transposeBValue = false;
    int8_t activationValue = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_matmul.Build(m_params, m_inputs, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_matmul.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnValue = mindspore::lite::MindIR_MatMulFusion_GetTransposeA(primitive.get());
    EXPECT_EQ(returnValue, transposeAValue);
    returnValue = mindspore::lite::MindIR_MatMulFusion_GetTransposeB(primitive.get());
    EXPECT_EQ(returnValue, transposeBValue);
    returnValue = mindspore::lite::MindIR_MatMulFusion_GetActivationType(primitive.get());
    EXPECT_EQ(returnValue, activationValue);
}

/**
 * @tc.name: matmul_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(MatMulBuilderTest, matmul_getprimitive_002, TestSize.Level0)
{
    MatmulBuilder matmul;
    LiteGraphPrimitvePtr primitive = m_matmul.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}