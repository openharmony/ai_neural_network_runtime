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

#include "frameworks/native/ops/conv2d_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class Conv2DBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

    void SetConv2dInput();
    void SetPad(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetPadParam();

public:
    Conv2DBuilder m_builder;
    std::vector<uint32_t> m_inputs{0, 1, 2};
    std::vector<uint32_t> m_outputs{3};
    std::vector<uint32_t> m_params{4, 5, 6, 7, 8};
    std::vector<int32_t> m_output_dim{1, 3, 3, 1};
    std::vector<int32_t> m_stride_dim{2};
    std::vector<int32_t> m_dilation_dim{2};
    std::vector<int32_t> m_pad_dim{4};
    std::vector<int32_t> m_param_dim{};
};

void Conv2DBuilderTest::SetUp() {}

void Conv2DBuilderTest::TearDown() {}

void Conv2DBuilderTest::SetPad(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t padNum = 4;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* padValue = new (std::nothrow) int64_t[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, padValue);
    tensor->SetBuffer(padValue, padNum * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
}

void Conv2DBuilderTest::SetPadParam()
{
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
}

void Conv2DBuilderTest::SetConv2dInput()
{
    int32_t weightNum = 4;
    std::vector<int32_t> m_input_dim{1, 4, 4, 1};
    std::vector<int32_t> weightDim = {1, 2, 2, 1};
    std::vector<int32_t> biasDim = {1};
    std::shared_ptr<NNTensor> inputsTensor;
    inputsTensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputsTensor);

    inputsTensor = TransToNNTensor(OH_NN_FLOAT32, weightDim, nullptr, OH_NN_TENSOR);
    float* weightValue = new (std::nothrow) float[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, weightValue);

    inputsTensor->SetBuffer(weightValue, weightNum * sizeof(weightValue));
    m_allTensors.emplace_back(inputsTensor);
    inputsTensor = TransToNNTensor(OH_NN_FLOAT32, biasDim, nullptr, OH_NN_TENSOR);
    float* biasValue = new (std::nothrow) float[1]{0};
    EXPECT_NE(nullptr, biasValue);

    inputsTensor->SetBuffer(biasValue, sizeof(float));
    m_allTensors.emplace_back(inputsTensor);
}

/**
 * @tc.name: conv2d_build_pad_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_003, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {1};
    m_params = {2, 3, 4, 5, 6};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_004, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {};
    m_params = {3, 4, 5, 6, 7};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}
/**
 * @tc.name: conv2d_build_pad_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_005, TestSize.Level1)
{
    m_inputs = {0, 1, 9};
    m_outputs = {3};
    m_params = {4, 5, 6, 7, 8};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_006, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {9};
    m_params = {4, 5, 6, 7, 8};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_007
 * @tc.desc: Verify the invalid stride of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_007, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    int32_t* strideValue = new (std::nothrow) int32_t[2]{1, 1};
    EXPECT_NE(nullptr, strideValue);

    tensor->SetBuffer(strideValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_008
 * @tc.desc: Verify the invalid dilation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_008, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    int32_t* dilationValue = new (std::nothrow) int32_t[2]{1, 1};
    EXPECT_NE(nullptr, dilationValue);

    tensor->SetBuffer(dilationValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_009
 * @tc.desc: Verify the invalid pad of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_009, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    SetConv2dInput();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);

    int32_t* padValue = new (std::nothrow) int32_t[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, padValue);
    tensor->SetBuffer(padValue, 4 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_010
 * @tc.desc: Verify the invalid group of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_010, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);

    int32_t* groupValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, groupValue);
    tensor->SetBuffer(groupValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}


/**
 * @tc.name: conv2d_build_pad_011
 * @tc.desc: Verify the invalid activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_011, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);

    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    int32_t* activationValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_012
 * @tc.desc: Verify the group scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_012, TestSize.Level1)
{
    std::vector<int32_t> groupDim = {2};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, groupDim, nullptr, OH_NN_CONV2D_GROUP);
    int64_t* groupValue = new (std::nothrow) int64_t[2]{0, 0};
    EXPECT_NE(nullptr, groupValue);

    tensor->SetBuffer(groupValue, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_013
 * @tc.desc: Verify the scalar activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_013, TestSize.Level1)
{
    std::vector<int32_t> activationDim = {2};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, activationDim, nullptr,
        OH_NN_CONV2D_ACTIVATION_TYPE);
    int8_t* activationValue = new (std::nothrow) int8_t[2]{0, 0};
    EXPECT_NE(nullptr, activationValue);
    tensor->SetBuffer(activationValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_014
 * @tc.desc: Verify the conv2d without set stride of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_014, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    SetConv2dInput();

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_stride_dim, nullptr,
        OH_NN_CONV2D_STRIDES);
    m_allTensors.emplace_back(tensor);

    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_015
 * @tc.desc: Verify the conv2d without set dilation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_015, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_dilation_dim, nullptr,
        OH_NN_CONV2D_DILATION);
    m_allTensors.emplace_back(tensor);

    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_016
 * @tc.desc: Verify the conv2d without set pad of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_016, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_pad_dim, nullptr,
        OH_NN_CONV2D_PAD);
    m_allTensors.emplace_back(tensor);

    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_017
 * @tc.desc: Verify the conv2d without set group of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_017, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_param_dim, nullptr,
        OH_NN_CONV2D_GROUP);
    m_allTensors.emplace_back(tensor);

    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_pad_018
 * @tc.desc: Verify the conv2d without set activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_build_pad_018, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPad(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_CONV2D_ACTIVATION_TYPE);
    m_allTensors.emplace_back(tensor);

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_getprimitive_pad_001
 * @tc.desc: Verify the nullptr return of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_getprimitive_pad_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    std::vector<int64_t> returnStrides = mindspore::lite::MindIR_Conv2DFusion_GetStride(primitive.get());
    std::vector<int64_t> strideValueTest{1, 1};
    EXPECT_EQ(strideValueTest, returnStrides);

    std::vector<int64_t> returnDliation = mindspore::lite::MindIR_Conv2DFusion_GetDilation(primitive.get());
    std::vector<int64_t> dilationValueTest{1, 1};
    EXPECT_EQ(dilationValueTest, returnDliation);

    std::vector<int64_t> returnPaddings = mindspore::lite::MindIR_Conv2DFusion_GetPadList(primitive.get());
    std::vector<int64_t> padValueTest{1, 1, 1, 1};
    EXPECT_EQ(padValueTest, returnPaddings);

    int returnGroup = mindspore::lite::MindIR_Conv2DFusion_GetGroup(primitive.get());
    EXPECT_EQ(0, returnGroup);
    int returnActivation = mindspore::lite::MindIR_Conv2DFusion_GetActivationType(primitive.get());
    EXPECT_EQ(0, returnActivation);
}

/**
 * @tc.name: conv2d_getprimitive_pad_002
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderTest, conv2d_getprimitive_pad_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetPadParam();

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
