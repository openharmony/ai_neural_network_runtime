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

#include "frameworks/native/ops/maxpool_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class MaxPoolBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

    void SetPadMode(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetParam();

public:
    MaxPoolBuilder m_builder;
    std::vector<uint32_t> m_inputs{0};
    std::vector<uint32_t> m_outputs{1};
    std::vector<uint32_t> m_params{2, 3, 4, 5};
    std::vector<int32_t> m_input_dim{1, 3, 3, 1};
    std::vector<int32_t> m_output_dim{1, 2, 2, 1};
    std::vector<int32_t> m_kenelsize_dim{2};
    std::vector<int32_t> m_stride_dim{2};
    std::vector<int32_t> m_param_dim{};
};

void MaxPoolBuilderTest::SetUp() {}

void MaxPoolBuilderTest::TearDown() {}

void MaxPoolBuilderTest::SetPadMode(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* padModeValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, padModeValue);
    tensor->SetBuffer(padModeValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
}

void MaxPoolBuilderTest::SetParam()
{
    SetKernelSize(OH_NN_INT64, m_kenelsize_dim, nullptr, OH_NN_MAX_POOL_KERNEL_SIZE);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_MAX_POOL_STRIDE);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_PAD_MODE);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_ACTIVATION_TYPE);
}

/**
 * @tc.name: maxpool_build_pad_mode_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_002
 * @tc.desc: Verify the forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_003
 * @tc.desc: Verify the missing input of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_003, TestSize.Level1)
{
    m_inputs = {};
    m_outputs = {0};
    m_params = {1, 2, 3, 4};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_004
 * @tc.desc: Verify the missing output of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_004, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {};
    m_params = {1, 2, 3, 4};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_005
 * @tc.desc: Verify the inputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_005, TestSize.Level1)
{
    m_inputs = {6};
    m_outputs = {1};
    m_params = {2, 3, 4, 5};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_006
 * @tc.desc: Verify the outputIndex out of bounds of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_006, TestSize.Level1)
{
    m_inputs = {0};
    m_outputs = {6};
    m_params = {2, 3, 4, 5};
    m_paramsIndex = m_params;

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetParam();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_007
 * @tc.desc: Verify the invalid kernelSize of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_007, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    int32_t kernelsNum{2};
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_kenelsize_dim, nullptr,
        OH_NN_MAX_POOL_KERNEL_SIZE);
    int32_t* kernelSizeValue = new (std::nothrow) int32_t[kernelsNum]{1, 1};
    EXPECT_NE(nullptr, kernelSizeValue);
    tensor->SetBuffer(kernelSizeValue, sizeof(int32_t) * kernelsNum);
    m_allTensors.emplace_back(tensor);

    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_MAX_POOL_STRIDE);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_PAD_MODE);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_008
 * @tc.desc: Verify the invalid stride of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_008, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetKernelSize(OH_NN_INT64, m_kenelsize_dim, nullptr, OH_NN_MAX_POOL_KERNEL_SIZE);
    int32_t strideNum{2};
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_stride_dim, nullptr, OH_NN_MAX_POOL_STRIDE);
    int32_t* strideValue = new (std::nothrow) int32_t[strideNum]{1, 1};
    EXPECT_NE(nullptr, strideValue);

    tensor->SetBuffer(strideValue, sizeof(int32_t) * strideNum);
    m_allTensors.emplace_back(tensor);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_PAD_MODE);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_009
 * @tc.desc: Verify the invalid padmode of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_009, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetKernelSize(OH_NN_INT64, m_kenelsize_dim, nullptr, OH_NN_MAX_POOL_KERNEL_SIZE);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_MAX_POOL_STRIDE);
    int32_t *padValueTest = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, padValueTest);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr, OH_NN_MAX_POOL_PAD);
    tensor->SetBuffer(padValueTest, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    SetActivation(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_ACTIVATION_TYPE);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}


/**
 * @tc.name: maxpool_build_pad_mode_010
 * @tc.desc: Verify the invalid activation of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_010, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);
    SetKernelSize(OH_NN_INT64, m_kenelsize_dim, nullptr, OH_NN_MAX_POOL_KERNEL_SIZE);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_MAX_POOL_STRIDE);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_PAD_MODE);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT32, m_param_dim, nullptr,
        OH_NN_MAX_POOL_ACTIVATION_TYPE);
    int32_t* activationValue = new (std::nothrow) int32_t(0);
    EXPECT_NE(nullptr, activationValue);

    tensor->SetBuffer(activationValue, sizeof(int32_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_build_pad_mode_011
 * @tc.desc: Verify the scalar length of the build function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_build_pad_mode_011, TestSize.Level1)
{
    m_param_dim = {2};
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetKernelSize(OH_NN_INT64, m_kenelsize_dim, nullptr, OH_NN_MAX_POOL_KERNEL_SIZE);
    SetStride(OH_NN_INT64, m_stride_dim, nullptr, OH_NN_MAX_POOL_STRIDE);
    SetPadMode(OH_NN_INT8, m_param_dim, nullptr, OH_NN_MAX_POOL_PAD_MODE);
    int8_t* activationValue = new (std::nothrow) int8_t[2]{1, 2};
    EXPECT_NE(nullptr, activationValue);

    std::shared_ptr<NNTensor> tensor = TransToNNTensor(OH_NN_INT8, m_param_dim, nullptr,
        OH_NN_MAX_POOL_ACTIVATION_TYPE);
    tensor->SetBuffer(activationValue, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
}

/**
 * @tc.name: maxpool_getprimitive_pad_mode_001
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_getprimitive_pad_mode_001, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetParam();
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(expectPrimitive, primitive);

    std::vector<int64_t> returnKernelSize = mindspore::lite::MindIR_MaxPoolFusion_GetKernelSize(primitive.get());
    std::vector<int64_t> kernelSizeValueTest{1, 1};
    EXPECT_EQ(kernelSizeValueTest, returnKernelSize);

    std::vector<int64_t> returnStrides = mindspore::lite::MindIR_MaxPoolFusion_GetStrides(primitive.get());
    std::vector<int64_t> strideValueTest{1, 1};
    int returnPadMode = mindspore::lite::MindIR_MaxPoolFusion_GetPadMode(primitive.get());
    EXPECT_EQ(1, returnPadMode);

    int returnActivation = mindspore::lite::MindIR_MaxPoolFusion_GetActivationType(primitive.get());
    EXPECT_EQ(0, returnActivation);
}

/**
 * @tc.name: maxpool_getprimitive_pad_mode_002
 * @tc.desc: Verify the behavior of the GetPrimitive function
 * @tc.type: FUNC
 */
HWTEST_F(MaxPoolBuilderTest, maxpool_getprimitive_pad_mode_002, TestSize.Level1)
{
    m_paramsIndex = m_params;
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_input_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_output_dim, nullptr);

    SetParam();
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(expectPrimitive, primitive);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS