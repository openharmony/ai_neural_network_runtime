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

#include "frameworks/native/ops/layernorm_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class LayerNormBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
void SaveNormAixsTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
void SaveEpsilonTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
void SaveParamAxisTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
void SetInputTensor(std::shared_ptr<NNTensor> inputTensor);

public:
    LayerNormBuilder m_layerNorm;
    std::vector<uint32_t> m_inputs {0, 1, 2};
    std::vector<uint32_t> m_outputs {3};
    std::vector<uint32_t> m_params {4, 5, 6};
    std::vector<int32_t> m_inputDimNorm {2, 3};
    std::vector<int32_t> m_inputDimEpsilon {3};
    std::vector<int32_t> m_inputDimParam {3};
    std::vector<int32_t> m_outputDim {3};
    std::vector<int32_t> m_paramDim {};
    std::shared_ptr<NNTensor> m_inputTensor {};
};

void LayerNormBuilderTest::SetUp() {}

void LayerNormBuilderTest::TearDown() {}

void LayerNormBuilderTest::SaveNormAixsTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t* beginNormAxisValue = new (std::nothrow) int32_t(1);
    EXPECT_NE(nullptr, beginNormAxisValue);
    std::shared_ptr<NNTensor> normAxisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    normAxisTensor->SetBuffer(beginNormAxisValue, sizeof(int32_t));
    m_allTensors.emplace_back(normAxisTensor);
}

void LayerNormBuilderTest::SaveEpsilonTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    float* epsilonValue = new (std::nothrow) float(0.0f);
    EXPECT_NE(nullptr, epsilonValue);
    std::shared_ptr<NNTensor> transposeBTensor = TransToNNTensor(dataType, dim, quantParam, type);
    transposeBTensor->SetBuffer(epsilonValue, sizeof(float));
    m_allTensors.emplace_back(transposeBTensor);
}

void LayerNormBuilderTest::SaveParamAxisTensor(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t* beginNormParamValue = new (std::nothrow) int32_t(1);
    EXPECT_NE(nullptr, beginNormParamValue);
    std::shared_ptr<NNTensor> paramAxisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    paramAxisTensor->SetBuffer(beginNormParamValue, sizeof(int32_t));
    m_allTensors.emplace_back(paramAxisTensor);
}

void LayerNormBuilderTest::SetInputTensor(std::shared_ptr<NNTensor> inputTensor)
{
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputDimNorm, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputDimEpsilon, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputDimParam, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);
}

/**
 * @tc.name: layernorm_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_001, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: layernorm_build_002
 * @tc.desc: Verify that the build function returns a failed message with duplicate Build().
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_002, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    EXPECT_EQ(OH_NN_SUCCESS, m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: layernorm_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_003, TestSize.Level0)
{
    m_inputs = {0, 1, 2, 3};
    m_outputs = {4};
    m_params = {5, 6, 7};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: layernorm_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_004, TestSize.Level0)
{
    m_outputs = {3, 4};
    m_params = {5, 6, 7};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: layernorm_build_005
 * @tc.desc: Verify that the build function returns a failed message with null allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: layernorm_build_006
 * @tc.desc: Verify that the build function returns a failed message with invalided allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_006, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: layernorm_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid beginNormAxis's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_007, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> normAxisTensor;
    normAxisTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    float beginNormAxisValue = 1e-7;
    normAxisTensor->SetBuffer(&beginNormAxisValue, sizeof(beginNormAxisValue));
    m_allTensors.emplace_back(normAxisTensor);

    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    normAxisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: layernorm_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid beginNormAxis's dimension.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_008, TestSize.Level0)
{
    std::vector<int32_t> expectParamDim = {2};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> normAxisTensor;
    normAxisTensor = TransToNNTensor(OH_NN_INT32, expectParamDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    int32_t beginNormAxisValue[2] = {1, 2};
    normAxisTensor->SetBuffer(beginNormAxisValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(normAxisTensor);

    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    normAxisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: layernorm_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid epsilon's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_009, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    std::shared_ptr<NNTensor> epsilonTensor;
    epsilonTensor = TransToNNTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    int32_t epsilonValue = 1;
    epsilonTensor->SetBuffer(&epsilonValue, sizeof(epsilonValue));
    m_allTensors.emplace_back(epsilonTensor);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    epsilonTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: layernorm_build_010
 * @tc.desc: Verify that the build function returns a failed message with invalid epsilon's dimension.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_010, TestSize.Level0)
{
    std::vector<int32_t> expectParamDim = {2};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    std::shared_ptr<NNTensor> epsilonTensor;
    epsilonTensor = TransToNNTensor(OH_NN_FLOAT32, expectParamDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    float epsilonValue[2] = {1e-7, 1e-7};
    epsilonTensor->SetBuffer(epsilonValue, 2 * sizeof(float));
    m_allTensors.emplace_back(epsilonTensor);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    epsilonTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: layernorm_build_011
 * @tc.desc: Verify that the build function returns a failed message with invalid beginParamAxis's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_011, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);

    std::shared_ptr<NNTensor> paramAxisTensor;
    paramAxisTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);
    float beginNormParamValue = 1;
    paramAxisTensor->SetBuffer(&beginNormParamValue, sizeof(beginNormParamValue));
    m_allTensors.emplace_back(paramAxisTensor);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    paramAxisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: layernorm_build_012
 * @tc.desc: Verify that the build function returns a failed message with invalid beginParamAxis's dimension.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_012, TestSize.Level0)
{
    std::vector<int32_t> expectParamDim = {2};

    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);

    std::shared_ptr<NNTensor> paramAxisTensor;
    paramAxisTensor = TransToNNTensor(OH_NN_INT32, expectParamDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);
    int32_t beginNormParamValue[2] = {1, 1};
    paramAxisTensor->SetBuffer(beginNormParamValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(paramAxisTensor);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    paramAxisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: layernorm_build_0013
 * @tc.desc: Verify that the build function returns a failed message with invalid param.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_0013, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_BATCH_NORM_EPSILON);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: layernorm_build_014
 * @tc.desc: Verify that the build function returns a failed message without set buffer for normAxis.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_014, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    std::shared_ptr<NNTensor> normAxisTensor;
    normAxisTensor = TransToNNTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    m_allTensors.emplace_back(normAxisTensor);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: layernorm_build_015
 * @tc.desc: Verify that the build function returns a failed message without set buffer for epsilon.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_015, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    std::shared_ptr<NNTensor> epsilonTensor;
    epsilonTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    m_allTensors.emplace_back(epsilonTensor);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: layernorm_build_016
 * @tc.desc: Verify that the build function returns a failed message without set buffer for paramsAxis.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_build_016, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);

    std::shared_ptr<NNTensor> paramAxisTensor;
    paramAxisTensor = TransToNNTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);
    m_allTensors.emplace_back(paramAxisTensor);

    OH_NN_ReturnCode ret = m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: layernorm_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_getprimitive_001, TestSize.Level0)
{
    SetInputTensor(m_inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveNormAixsTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_NORM_AXIS);
    SaveEpsilonTensor(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_EPSILON);
    SaveParamAxisTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS);

    int32_t beginNormAxisValue = 1;
    float epsilonValue = 0.0f;
    int32_t beginNormParamValue = 1;
    EXPECT_EQ(OH_NN_SUCCESS, m_layerNorm.Build(m_params, m_inputs, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_layerNorm.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);
    auto returnValue = mindspore::lite::MindIR_LayerNormFusion_GetBeginNormAxis(primitive.get());
    EXPECT_EQ(returnValue, beginNormAxisValue);
    returnValue = mindspore::lite::MindIR_LayerNormFusion_GetEpsilon(primitive.get());
    EXPECT_EQ(returnValue, epsilonValue);
    returnValue = mindspore::lite::MindIR_LayerNormFusion_GetBeginParamsAxis(primitive.get());
    EXPECT_EQ(returnValue, beginNormParamValue);
}

/**
 * @tc.name: layernorm_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(LayerNormBuilderTest, layernorm_getprimitive_002, TestSize.Level0)
{
    LayerNormBuilder layerNorm;
    LiteGraphPrimitvePtr primitive = m_layerNorm.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}