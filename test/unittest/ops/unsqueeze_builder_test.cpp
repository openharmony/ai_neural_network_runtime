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

#include "frameworks/native/ops/unsqueeze_builder.h"

#include <gtest/gtest.h>
#include "frameworks/native/nn_tensor.h"
#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class UnsqueezeBuilderTest : public OpsTest {
protected:
    void InitTensor(const std::vector<uint32_t>& inputsIndex,
        const std::vector<uint32_t>& outputsIndex) override;
    void SaveAxisTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    UnsqueezeBuilder m_builder;
    std::vector<int64_t> m_expectAxisValue;
};

void UnsqueezeBuilderTest::SaveAxisTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axisValue = new (std::nothrow) int64_t[1]{1};
    axisTensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(axisTensor);
    m_expectAxisValue.emplace_back(*axisValue);
}

void UnsqueezeBuilderTest::InitTensor(const std::vector<uint32_t>& inputsIndex,
    const std::vector<uint32_t>& outputsIndex)
{
    std::vector<uint32_t> paramsIndex = { 2 };
    std::vector<int32_t> inputDim = {1, 5, 1};
    std::vector<int32_t> OutputDim = {1, 1, 5, 1};

    m_paramsIndex = paramsIndex;
    SaveInputTensor(inputsIndex, OH_NN_FLOAT32, inputDim, nullptr);
    SaveOutputTensor(outputsIndex, OH_NN_FLOAT32, OutputDim, nullptr);
}

/**
 * @tc.name: unsqueeze_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_001, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: unsqueeze_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: unsqueeze_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_003, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0, 1, 2 };
    std::vector<uint32_t> outputsIndex = { 3 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_004, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1, 2 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_build_005
 * @tc.desc: Provide empty input, output, and parameters to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_005, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<uint32_t> paramsIndex = { 2 };

    OH_NN_ReturnCode ret = m_builder.Build(paramsIndex, inputsIndex, outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_build_006
 * @tc.desc: Provide empty output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_006, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = {};
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_build_007
 * @tc.desc: Provide param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_007, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT8, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_build_008
 * @tc.desc: Provide param type error to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_008, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT32, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_build_009
 * @tc.desc: Provide axis parameter buffer is nullptr to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_009, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);

    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);
    axisTensor->SetBuffer(nullptr, 0);
    m_allTensors.emplace_back(axisTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_build_010
 * @tc.desc: Provide axis parameter is not scaler to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_010, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> paramDim = {1, 2};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_build_011
 * @tc.desc: Provide invalid parameter type to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_build_011, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_SCALE_AXIS);

    OH_NN_ReturnCode ret = m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: unsqueeze_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = { nullptr, DestroyLiteGraphPrimitive };
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: unsqueeze_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(UnsqueezeBuilderTest, unsqueeze_get_primitive_002, TestSize.Level0)
{
    std::vector<uint32_t> inputsIndex = { 0 };
    std::vector<uint32_t> outputsIndex = { 1 };
    std::vector<int32_t> paramDim = {};

    InitTensor(inputsIndex, outputsIndex);
    SaveAxisTensor(OH_NN_INT64, paramDim, nullptr, OH_NN_UNSQUEEZE_AXIS);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_paramsIndex, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = { nullptr, DestroyLiteGraphPrimitive };
    EXPECT_NE(primitive, expectPrimitive);

    auto axisReturn = mindspore::lite::MindIR_Unsqueeze_GetAxis(primitive.get());
    auto axisReturnSize = axisReturn.size();
    for (size_t i = 0; i < axisReturnSize; ++i) {
        EXPECT_EQ(axisReturn[i], m_expectAxisValue[i]);
    }
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
