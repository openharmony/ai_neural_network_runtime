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
class Conv2DBuilderPadmodeTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

    void SetConv2dInput();
    void SetPadMode(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetParam();

public:
    Conv2DBuilder m_builder;
    std::vector<uint32_t> m_inputs{0, 1, 2};
    std::vector<uint32_t> m_outputs{3};
    std::vector<uint32_t> m_params{4, 5, 6, 7, 8};
    std::vector<int32_t> m_output_dim{1, 3, 3, 1};
    std::vector<int32_t> m_stride_dim{2};
    std::vector<int32_t> m_dilation_dim{2};
    std::vector<int32_t> m_param_dim{};
};

void Conv2DBuilderPadmodeTest::SetUp() {}

void Conv2DBuilderPadmodeTest::TearDown() {}

void Conv2DBuilderPadmodeTest::SetPadMode(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* padModeValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, padModeValue);

    tensor->SetBuffer(padModeValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
}

void Conv2DBuilderPadmodeTest::SetParam()
{
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
}

void Conv2DBuilderPadmodeTest::SetConv2dInput()
{
    int32_t weightNum = 4;
    std::vector<int32_t> m_input_dim{1, 4, 4, 1};
    std::vector<int32_t> weightDim = {1, 2, 2, 1};
    std::vector<int32_t> biasDim = {1};

    std::shared_ptr<NNTensor> inputTensor;
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    inputTensor = TransToNNTensor(OH_NN_FLOAT32, weightDim, nullptr, OH_NN_TENSOR);
    float* weightValue = new (std::nothrow) float[4]{1, 1, 1, 1};
    EXPECT_NE(nullptr, weightValue);
    inputTensor->SetBuffer(weightValue, weightNum * sizeof(weightValue));
    m_allTensors.emplace_back(inputTensor);

    inputTensor = TransToNNTensor(OH_NN_FLOAT32, biasDim, nullptr, OH_NN_TENSOR);
    float* biasValue = new (std::nothrow) float[1]{0};
    EXPECT_NE(nullptr, biasValue);

    inputTensor->SetBuffer(biasValue, sizeof(float));
    m_allTensors.emplace_back(inputTensor);
}


/**
 * @tc.name: conv2d_build_padmode_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_003, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {1};
    m_params = {2, 3, 4, 5, 6};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_004, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {};
    m_params = {3, 4, 5, 6, 7};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}
/**
 * @tc.name: conv2d_build_padmode_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_005, TestSize.Level1)
{
    m_inputs = {0, 1, 9};
    m_outputs = {3};
    m_params = {4, 5, 6, 7, 8};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_006, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {9};
    m_params = {4, 5, 6, 7, 8};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_007
 * @tc.desc: Verify the invalid stride of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_007, TestSize.Level1)
{
    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    int32_t* strideValue = new (std::nothrow) int32_t[2]{1, 1};
    EXPECT_NE(nullptr, strideValue);

    tensor->SetBuffer(strideValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_008
 * @tc.desc: Verify the invalid dilation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_008, TestSize.Level1)
{
    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    int32_t* dilationValue = new (std::nothrow) int32_t[2]{1, 1};
    EXPECT_NE(nullptr, dilationValue);

    tensor->SetBuffer(dilationValue, 2 * sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);

    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_009
 * @tc.desc: Verify the invalid padMode of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_009, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    int32_t* padModeValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, padModeValue);
    tensor->SetBuffer(padModeValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);

    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_010
 * @tc.desc: Verify the invalid group of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_010, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    int32_t* groupValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, groupValue);

    tensor->SetBuffer(groupValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}


/**
 * @tc.name: conv2d_build_padmode_011
 * @tc.desc: Verify the invalid activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_011, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    int32_t* activationValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, activationValue);
    tensor->SetBuffer(activationValue, sizeof(int32_t));

    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_012
 * @tc.desc: Verify the group scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_012, TestSize.Level1)
{
    std::vector<int32_t> groupDim = {2};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, groupDim, nullptr, OH_NN_CONV2D_GROUP);
    int64_t* groupValue = new (std::nothrow) int64_t[2]{0, 0};
    EXPECT_NE(nullptr, groupValue);
    tensor->SetBuffer(groupValue, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);

    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_013
 * @tc.desc: Verify the activation scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_013, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);

    std::vector<int32_t> activationDim = {2};
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, activationDim, nullptr,
        OH_NN_CONV2D_ACTIVATION_TYPE);
    int8_t* activationValue = new (std::nothrow) int8_t[2]{0, 0};
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_014
 * @tc.desc: Verify the param invalid to conv2d of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_014, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    std::vector<int32_t> activationDim = {2};

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, activationDim, nullptr,
        OH_NN_DIV_ACTIVATIONTYPE);
    int8_t* activationValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValue);
    tensor->SetBuffer(activationValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_015
 * @tc.desc: Verify the pad value of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_015, TestSize.Level1)
{
    std::vector<int32_t> activationDim = {2};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    int8_t* padModeValue = new (std::nothrow) int8_t(10);
    EXPECT_NE(nullptr, padModeValue);

    tensor->SetBuffer(padModeValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_016
 * @tc.desc: Verify the activation scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_016, TestSize.Level1)
{
    std::vector<int32_t> activationDim = {2};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;
    int32_t weightNum = 3;
    std::vector<int32_t> m_input_dim{1, 4, 4, 1};
    std::vector<int32_t> weightDim = {1, 3, 1};
    std::vector<int32_t> biasDim = {1};

    std::shared_ptr<NNTensor> inputTensor;
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_input_dim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, weightDim, nullptr, OH_NN_TENSOR);
    float* weightValue = new (std::nothrow) float[3]{1, 1, 1};
    EXPECT_NE(nullptr, weightValue);

    inputTensor->SetBuffer(weightValue, weightNum * sizeof(weightValue));
    m_allTensors.emplace_back(inputTensor);
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, biasDim, nullptr, OH_NN_TENSOR);
    float* biasValue = new (std::nothrow) float[1]{0};
    EXPECT_NE(nullptr, biasValue);

    inputTensor->SetBuffer(biasValue, sizeof(float));
    m_allTensors.emplace_back(inputTensor);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_017
 * @tc.desc: Verify the activation value of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_017, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_CONV2D_PAD_MODE);
    SetGroup(OH_NN_INT64, m_param_dim, nullptr, OH_NN_CONV2D_GROUP);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_CONV2D_ACTIVATION_TYPE);
    int8_t* activationValue = new (std::nothrow) int8_t(10);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_build_padmode_018
 * @tc.desc: Verify the activation value of the build function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_build_padmode_018, TestSize.Level1)
{
    std::vector<int32_t> m_pad_dim = {3};
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_CONV2D_STRIDES);
    SetDilation(OH_NN_INT64, m_dilation_dim, nullptr, OH_NN_CONV2D_DILATION);

    int32_t padNum = 3;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT64, m_pad_dim, nullptr, OH_NN_CONV2D_PAD);
    int64_t* padValue = new (std::nothrow) int64_t[3]{1, 1, 1};
    EXPECT_NE(nullptr, padValue);

    tensor->SetBuffer(padValue, padNum * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: conv2d_getprimitive_padmode_001
 * @tc.desc: Verify the success of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_getprimitive_padmode_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    std::vector<int64_t> expectStrides = mindspore::lite::MindIR_Conv2DFusion_GetStride(primitive.get());
    std::vector<int64_t> strideValueTest{1, 1};
    EXPECT_EQ(strideValueTest, expectStrides);

    std::vector<int64_t> expectDliation = mindspore::lite::MindIR_Conv2DFusion_GetDilation(primitive.get());
    std::vector<int64_t> dilationValueTest{1, 1};
    EXPECT_EQ(dilationValueTest, expectDliation);
    int expectpadMode = mindspore::lite::MindIR_Conv2DFusion_GetPadMode(primitive.get());
    EXPECT_EQ(1, expectpadMode);

    int expectGroup = mindspore::lite::MindIR_Conv2DFusion_GetGroup(primitive.get());
    EXPECT_EQ(0, expectGroup);

    int expectActivation = mindspore::lite::MindIR_Conv2DFusion_GetActivationType(primitive.get());
    EXPECT_EQ(0, expectActivation);
}

/**
 * @tc.name: conv2d_getprimitive_padmode_002
 * @tc.desc: Verify the nullptr return of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(Conv2DBuilderPadmodeTest, conv2d_getprimitive_padmode_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    m_inputsIndex = m_inputs;

    SetConv2dInput();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();

    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS