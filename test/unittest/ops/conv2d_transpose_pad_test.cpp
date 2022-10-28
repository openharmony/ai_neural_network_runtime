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

#include "frameworks/native/ops/conv2d_transpose_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class Conv2DTransposeBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

    void SetConv2dTransposeInput();
    void SetPad(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetOutPaddings(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetPadParam();

public:
    Conv2DTransposeBuilder m_builder;
    std::vector<uint32_t> m_inputs{0, 1, 2};
    std::vector<uint32_t> m_outputs{3};
    std::vector<uint32_t> m_params{4, 5, 6, 7, 8, 9};
    std::vector<int32_t> m_output_dim{1, 3, 3, 1};
    std::vector<int32_t> m_stride_dim{2};
    std::vector<int32_t> m_dilation_dim{2};
    std::vector<int32_t> m_outpaddings_dim{2};
    std::vector<int32_t> m_pad_dim{4};
    std::vector<int32_t> m_param_dim{};
};

void Conv2DTransposeBuilderTest::SetUp() {}

void Conv2DTransposeBuilderTest::TearDown() {}

void Conv2DTransposeBuilderTest::SetPad(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t padNum = 4;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* padValue = new (std::nothrow) int64_t[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, padValue);

    tensor->SetBuffer(padValue, padNum * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
}

void Conv2DTransposeBuilderTest::SetOutPaddings(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t outPaddingsNum = 2;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* outPaddingsValue = new (std::nothrow) int64_t[2]{0, 0};
    EXPECT_NE(nullptr, outPaddingsValue);

    tensor->SetBuffer(outPaddingsValue, outPaddingsNum * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
}

void Conv2DTransposeBuilderTest::SetPadParam()
{
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);
    SetOutPaddings(OH_NN_INT64, m_outpaddings_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
}

void Conv2DTransposeBuilderTest::SetConv2dTransposeInput()
{
    int32_t weightNum = 4;
    std::vector<int32_t> m_input_dim{1, 4, 4, 1};
    std::vector<int32_t> weightDim = {1, 2, 2, 1};
    std::vector<int32_t> biasDim = {1};
    std::shared_ptr<NNTensor> tensor;
    tensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(tensor);

    tensor = TransToNNTensor(OH_NN_FLOAT32, weightDim, nullptr, OH_NN_TENSOR);
    float* weightValue = new (std::nothrow) float[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, weightValue);

    tensor->SetBuffer(weightValue, weightNum * sizeof(weightValue));
    m_allTensors.emplace_back(tensor);
    tensor = TransToNNTensor(OH_NN_FLOAT32, biasDim, nullptr, OH_NN_TENSOR);
    float* biasValue = new (std::nothrow) float[1]{0};
    EXPECT_NE(nullptr, biasValue);

    tensor->SetBuffer(biasValue, sizeof(float));
    m_allTensors.emplace_back(tensor);
}

/**
 * @tc.name: conv2dtranpose_build_pad_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_001, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_002, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_003, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {1};
    m_params = {2, 3, 4, 5, 6, 7};

    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_004, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {};
    m_params = {3, 4, 5, 6, 7, 8};

    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_005, TestSize.Level1)
{
    m_inputs = {0, 1, 10};
    m_outputs = {3};
    m_params = {4, 5, 6, 7, 8, 9};

    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_006, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {10};
    m_params = {4, 5, 6, 7, 8, 9};;

    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_007
 * @tc.desc: Verify the invalid stride of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_007, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_stride_dim, nullptr,
        OH_NN_CONV2D_TRANSPOSE_STRIDES);
    int32_t* strideValue = new (std::nothrow) int32_t[2]{1, 1};
    EXPECT_NE(nullptr, strideValue);

    tensor->SetBuffer(strideValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);
    SetOutPaddings(OH_NN_INT64, m_outpaddings_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_008
 * @tc.desc: Verify the invalid dilation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_008, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_STRIDES);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_dilation_dim, nullptr,
        OH_NN_CONV2D_TRANSPOSE_DILATION);
    int32_t* dilationValue = new (std::nothrow) int32_t[2]{1, 1};
    EXPECT_NE(nullptr, dilationValue);

    tensor->SetBuffer(dilationValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);
    SetOutPaddings(OH_NN_INT64, m_outpaddings_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_009
 * @tc.desc: Verify the invalid pad of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_009, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_DILATION);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);
    int32_t* padValue = new (std::nothrow) int32_t[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, padValue);

    tensor->SetBuffer(padValue, 4 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetOutPaddings(OH_NN_INT64, m_outpaddings_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_010
 * @tc.desc: Verify the invalid outpaddings of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_010, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_outpaddings_dim, nullptr,
        OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);

    int32_t* outPaddingsTypeInvalid = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, outPaddingsTypeInvalid);
    tensor->SetBuffer(outPaddingsTypeInvalid, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_011
 * @tc.desc: Verify the invalid group of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_011, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);
    SetOutPaddings(OH_NN_INT64, m_outpaddings_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_GROUP);
    int32_t* groupValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, groupValue);

    tensor->SetBuffer(groupValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_012
 * @tc.desc: Verify the invalid activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_012, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);
    SetOutPaddings(OH_NN_INT64, m_outpaddings_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_GROUP);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr,
        OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
    int32_t* activationValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_013
 * @tc.desc: Verify the group scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_013, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);
    SetOutPaddings(OH_NN_INT64, m_outpaddings_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_outpaddings_dim, nullptr,
        OH_NN_CONV2D_TRANSPOSE_GROUP);
    int64_t* groupValue = new (std::nothrow) int64_t[2]{0, 0};
    EXPECT_NE(nullptr, groupValue);

    tensor->SetBuffer(groupValue, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_build_pad_014
 * @tc.desc: Verify the activation scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_build_pad_014, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_PAD);
    SetOutPaddings(OH_NN_INT64, m_outpaddings_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_TRANSPOSE_GROUP);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_outpaddings_dim, nullptr,
        OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE);
    int8_t* activationValue = new (std::nothrow) int8_t[2]{0, 0};
    EXPECT_NE(nullptr, activationValue);
    tensor->SetBuffer(activationValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2dtranpose_getprimitive_padmode_001
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_getprimitive_padmode_001, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    std::vector<int64_t> returnStrides = mindspore::lite::MindIR_Conv2dTransposeFusion_GetStride(primitive.get());
    std::vector<int64_t> strideValueTest{1, 1};
    std::vector<int64_t> returnDliation = mindspore::lite::MindIR_Conv2dTransposeFusion_GetDilation(primitive.get());
    std::vector<int64_t> dilationValueTest{1, 1};
    std::vector<int64_t> returnPad = mindspore::lite::MindIR_Conv2dTransposeFusion_GetPadList(primitive.get());
    std::vector<int64_t> padValueTest{1, 1, 1, 1};
    int returnGroup = mindspore::lite::MindIR_Conv2dTransposeFusion_GetGroup(primitive.get());
    EXPECT_EQ(0, returnGroup);

    std::vector<int64_t> outPaddingReturn =
        mindspore::lite::MindIR_Conv2dTransposeFusion_GetOutputPaddings(primitive.get());
    std::vector<int64_t> outPaddingTest{0, 0};
    EXPECT_EQ(outPaddingTest, outPaddingReturn);

    int returnActivation = mindspore::lite::MindIR_Conv2dTransposeFusion_GetActivationType(primitive.get());
    EXPECT_EQ(0, returnActivation);
}

/**
 * @tc.name: conv2dtranpose_getprimitive_padmode_002
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DTransposeBuilderTest, conv2dtranpose_getprimitive_padmode_002, TestSize.Level1)
{
    SetConv2dTransposeInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
