/*
 * Copyright (c) 2022 Huawei Device Co., Ltd.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "frameworks/native/ops/sub_builder.h"

#include <gtest/gtest.h>
#include "frameworks/native/nn_tensor.h"
#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class SubBuilderTest : public OpsTest {
protected:
    void InitTensor(const std::vector<uint32_t>& inputsIndex,
        const std::vector<uint32_t>& outputsIndex) override;
    void SaveActivateTypeTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    SubBuilder m_builder;
    int8_t m_expectActivationTypeValue {0};
};

void SubBuilderTest::SaveActivateTypeTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> activationTypeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* activationTypeValue = new (std::nothrow) int8_t[1]{OH_NN_FUSED_RELU6};
    EXPECT_NE(nullptr, activationTypeValue);
    activationTypeTensor->SetBuffer(activationTypeValue, sizeof(int8_t));
    m_allTensors.emplace_back(activationTypeTensor);
    m_expectActivationTypeValue = mindspore::lite::ACTIVATION_TYPE_RELU6;
}

void SubBuilderTest::InitTensor(const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex)
{
    std::vector<uint32_t> paramsIndex = { 3 };
    std::vector<int32_t> inputDim = {3};
    std::vector<int32_t> OutputDim = {3};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
}

/**
 * @tc.name: sub_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_001, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: sub_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: sub_build_003
 * @tc.desc:Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_003, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2, 3 };
    std::vector<uint32_t> outputsIndex = { 4 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_004, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2, 3 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_005
 * @tc.desc: Provide empty input, output, and parameters to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_005, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2 };
    std::vector<uint32_t> outputsIndex = { 3 };
    std::vector<uint32_t> paramsIndex = { 4 };

    OH_NN_ReturnCode ret = m_builder.Build(paramsIndex, inputsIndex, outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_006
 * @tc.desc: Provide empty output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_006, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2 };
    std::vector<uint32_t> outputsIndex = {};
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_BOOL, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_007
 * @tc.desc: Provide param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_007, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_BOOL, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_008
 * @tc.desc:Provide param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_008, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_009
 * @tc.desc: Provide activate type parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_009, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);

    std::shared_ptr<NNTensor> activationTypeTensor = TransToNNTensor(OH_NN_INT8, paramDim, nullptr,
        OH_NN_SUB_ACTIVATIONTYPE);
    activationTypeTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(activationTypeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_010
 * @tc.desc: Provide parameter is not scaler to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_010, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {1, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_011
 * @tc.desc: Provide invalid parameter type to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_011, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_SCALE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_build_012
 * @tc.desc: Provide invalid param value to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_build_012, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);

    std::shared_ptr<NNTensor> activationTypeTensor = TransToNNTensor(OH_NN_INT8, paramDim, nullptr,
        OH_NN_SUB_ACTIVATIONTYPE);
    int8_t* activationTypeValue = new (std::nothrow) int8_t[1]{-1};
    EXPECT_NE(nullptr, activationTypeValue);
    activationTypeTensor->SetBuffer(activationTypeValue, sizeof(int8_t));
    m_allTensors.emplace_back(activationTypeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: sub_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = { nullptr, DestroyLiteGraphPrimitive };
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: sub_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(SubBuilderTest, sub_get_primitive_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1 };
    std::vector<uint32_t> outputsIndex = { 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveActivateTypeTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_SUB_ACTIVATIONTYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = { nullptr, DestroyLiteGraphPrimitive };
    EXPECT_NE(primitive, expectPrimitive);

    auto returnValue = mindspore::lite::MindIR_SubFusion_GetActivationType(primitive.get());
    EXPECT_EQ(returnValue, m_expectActivationTypeValue);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
