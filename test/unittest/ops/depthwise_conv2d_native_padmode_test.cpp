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

#include "frameworks/native/ops/depthwise_conv2d_native_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class DepthwiseConv2DNativePadModeBuilderTest : public OpsTest {
public:
    void SetUp();
    void TearDown();

    void SetDepthwiseConv2dInput();
    void SetPadMode(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetParam();

public:
    DepthwiseConv2DNativeBuilder m_builder;
    std::vector<uint32_t> m_inputs{0, 1, 2};
    std::vector<uint32_t> m_outputs{3};
    std::vector<uint32_t> m_params{4, 5, 6, 7};
    std::vector<int32_t> m_output_dim{1, 4, 4, 2};
    std::vector<int32_t> m_stride_dim{2};
    std::vector<int32_t> m_dilation_dim{2};
    std::vector<int32_t> m_param_dim{};
};

void DepthwiseConv2DNativePadModeBuilderTest::SetUp() {}

void DepthwiseConv2DNativePadModeBuilderTest::TearDown() {}

void DepthwiseConv2DNativePadModeBuilderTest::SetPadMode(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* padModeValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, padModeValue);
    tensor->SetBuffer(padModeValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
}

void DepthwiseConv2DNativePadModeBuilderTest::SetParam()
{
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE);
}

void DepthwiseConv2DNativePadModeBuilderTest::SetDepthwiseConv2dInput()
{
    int32_t weightNum = 8;
    int32_t biasNum = 2;
    std::vector<int32_t> m_input_dim{1, 3, 3, 2};
    std::vector<int32_t> weightDim = {2, 2, 2, 1};
    std::vector<int32_t> biasDim = {2};
    std::shared_ptr<NNTensor> inputTensor;
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    inputTensor = TransToNNTensor(OH_NN_FLOAT32, weightDim, nullptr, OH_NN_TENSOR);
    float* weightValue = new (std::nothrow) float[8]{1, 0, 0, 1, 0, 1, 1, 0};
    EXPECT_NE(nullptr, weightValue);

    inputTensor->SetBuffer(weightValue, weightNum * sizeof(weightValue));
    m_allTensors.emplace_back(inputTensor);

    inputTensor = TransToNNTensor(OH_NN_FLOAT32, biasDim, nullptr, OH_NN_TENSOR);
    float* biasValue = new (std::nothrow) float[2]{0, 0};
    EXPECT_NE(nullptr, biasValue);

    inputTensor->SetBuffer(biasValue, biasNum * sizeof(float));
    m_allTensors.emplace_back(inputTensor);
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_003, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {1};
    m_params = {2, 3, 4, 5};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_004, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {};
    m_params = {3, 4, 5, 6};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_005, TestSize.Level1)
{
    m_inputs = {0, 1, 9};
    m_outputs = {3};
    m_params = {4, 5, 6, 7};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_006, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {9};
    m_params = {4, 5, 6, 7};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_007
 * @tc.desc: Verify the invalid stride  of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_007, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_stride_dim, nullptr,
        OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES);
    int32_t* strideValue = new (std::nothrow) int32_t[2]{1, 1};
    EXPECT_NE(nullptr, strideValue);

    tensor->SetBuffer(strideValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_008
 * @tc.desc: Verify the invalid dilation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_008, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_dilation_dim, nullptr,
        OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION);
    int32_t* dilationValue = new (std::nothrow) int32_t[2]{1, 1};
    EXPECT_NE(nullptr, dilationValue);

    tensor->SetBuffer(dilationValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_009
 * @tc.desc: Verify the invalid pad of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_009, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr,
        OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE);
    int32_t* padModeValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, padModeValue);
    tensor->SetBuffer(padModeValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_010
 * @tc.desc: Verify the invalid activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_010, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr,
        OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE);
    int32_t* activationValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, activationValue);
    tensor->SetBuffer(activationValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_011
 * @tc.desc: Verify the scalar activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_011, TestSize.Level1)
{
    std::vector<int32_t> activationDim = {2};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, activationDim, nullptr,
        OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE);
    int8_t* activationValue = new (std::nothrow) int8_t[2]{0, 0};
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_build_padmode_012
 * @tc.desc: Verify the invalid pad of the build function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_build_padmode_012, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE);
    int8_t* padModeValue = new (std::nothrow) int8_t(10);
    EXPECT_NE(nullptr, padModeValue);

    tensor->SetBuffer(padModeValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: depthwiseconv2d_getprimitive_padmode_001
 * @tc.desc: Verify the success of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_getprimitive_padmode_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);

    std::vector<int64_t> returnStrides = mindspore::lite::MindIR_Conv2DFusion_GetStride(primitive.get());
    std::vector<int64_t> strideValueTest{1, 1};
    std::vector<int64_t> returnDliation = mindspore::lite::MindIR_Conv2DFusion_GetDilation(primitive.get());
    std::vector<int64_t> dilationValueTest{1, 1};
    EXPECT_EQ(dilationValueTest, returnDliation);

    int returnpadMode = mindspore::lite::MindIR_Conv2DFusion_GetPadMode(primitive.get());
    EXPECT_EQ(1, returnpadMode);
    int returnActivation = mindspore::lite::MindIR_Conv2DFusion_GetActivationType(primitive.get());
    EXPECT_EQ(0, returnActivation);
}

/**
 * @tc.name: depthwiseconv2d_getprimitive_padmode_002
 * @tc.desc: Verify the nullptr return of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(DepthwiseConv2DNativePadModeBuilderTest, depthwiseconv2d_getprimitive_padmode_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetDepthwiseConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
