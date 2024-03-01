/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#include "ops/clip_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ClipBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveMax(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveMin(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    ClipBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2, 3};
    std::vector<int32_t> m_dim {1, 2, 2, 1};
    std::vector<int32_t> m_paramDim {};
};

void ClipBuilderTest::SetUp() {}

void ClipBuilderTest::TearDown() {}

void ClipBuilderTest::SaveMax(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> maxTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* maxValue = new (std::nothrow) float [1]{10.0f};
    EXPECT_NE(nullptr, maxValue);
    maxTensor->SetBuffer(maxValue, sizeof(float));
    m_allTensors.emplace_back(maxTensor);
}

void ClipBuilderTest::SaveMin(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> minTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* minValue = new (std::nothrow) float [1]{1.0f};
    EXPECT_NE(nullptr, minValue);
    minTensor->SetBuffer(minValue, sizeof(float));
    m_allTensors.emplace_back(minTensor);
}

/**
 * @tc.name: clip_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MAX);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MIN);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: clip_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MAX);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MIN);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: clip_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_003, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3, 4};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MAX);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MIN);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: clip_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_004, TestSize.Level1)
{
    m_outputs = {1, 2};
    m_params = {3, 4};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MAX);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MIN);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: clip_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: clip_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: clip_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid max's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> maxTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_CLIP_MAX);
    int64_t* maxValue = new (std::nothrow) int64_t [1]{10};
    EXPECT_NE(nullptr, maxValue);
    maxTensor->SetBuffer(maxValue, sizeof(int64_t));
    m_allTensors.emplace_back(maxTensor);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MIN);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    maxTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: clip_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid min's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MAX);
    std::shared_ptr<NNTensor> minTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_CLIP_MIN);
    int64_t* minValue = new (std::nothrow) int64_t [1]{1};
    EXPECT_NE(nullptr, minValue);
    minTensor->SetBuffer(minValue, sizeof(int64_t));
    m_allTensors.emplace_back(minTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    minTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: clip_build_009
 * @tc.desc: Verify that the build function returns a failed message with passing invalid max param.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MIN);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: clip_build_010
 * @tc.desc: Verify that the build function returns a failed message with passing invalid min param.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_010, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    
    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MAX);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: clip_build_011
 * @tc.desc: Verify that the build function returns a failed message without set buffer for max.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    std::shared_ptr<NNTensor> maxTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_CLIP_MAX);
    m_allTensors.emplace_back(maxTensor);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MIN);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: clip_build_012
 * @tc.desc: Verify that the build function returns a failed message without set buffer for min.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_build_012, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);

    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MAX);
    std::shared_ptr<NNTensor> minTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_CLIP_MIN);
    m_allTensors.emplace_back(minTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: clip_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_dim, nullptr);
    SaveMax(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MAX);
    SaveMin(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_CLIP_MIN);

    float maxValue = 10.0f;
    float minValue = 1.0f;
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnMaxValue = mindspore::lite::MindIR_Clip_GetMax(primitive.get());
    EXPECT_EQ(returnMaxValue, maxValue);
    auto returnMinValue = mindspore::lite::MindIR_Clip_GetMin(primitive.get());
    EXPECT_EQ(returnMinValue, minValue);
}

/**
 * @tc.name: clip_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(ClipBuilderTest, clip_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}