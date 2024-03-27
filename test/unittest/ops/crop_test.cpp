/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#include "ops/crop_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class CropBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveAxis(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveOffset(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetInputAndShape();

protected:
    CropBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1};
    std::vector<uint32_t> m_outputs {2};
    std::vector<uint32_t> m_params {3, 4};
    std::vector<int32_t> m_inputDim {2, 3, 4, 5};
    std::vector<int32_t> m_shapeDim {1};
    std::vector<int32_t> m_outputDim {2, 3, 4, 5};
    std::vector<int32_t> m_axisDim {};
    std::vector<int32_t> m_offsetDim {1};
};

void CropBuilderTest::SetUp() {}

void CropBuilderTest::TearDown() {}

void CropBuilderTest::SetInputAndShape()
{
    m_inputsIndex = m_inputs;
    std::shared_ptr<NNTensor> inputTensor;
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    std::shared_ptr<NNTensor> shapeTensor;
    shapeTensor = TransToNNTensor(OH_NN_FLOAT32, m_shapeDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(shapeTensor);
}

void CropBuilderTest::SaveAxis(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* axisValue = new (std::nothrow) int64_t[1] {0};
    axisTensor->SetBuffer(axisValue, sizeof(int64_t));
    m_allTensors.emplace_back(axisTensor);
}

void CropBuilderTest::SaveOffset(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> offsetTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* offsetValue = new (std::nothrow) int64_t[1] {1};
    offsetTensor->SetBuffer(offsetValue, sizeof(int64_t));
    m_allTensors.emplace_back(offsetTensor);
}

/**
 * @tc.name: crop_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_001, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_CROP_AXIS);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_CROP_OFFSET);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: crop_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_002, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_CROP_AXIS);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_CROP_OFFSET);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: crop_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_003, TestSize.Level1)
{
    m_inputs = {0, 1, 2};
    m_outputs = {3};
    m_params = {4, 5};

    SetInputAndShape();
    std::shared_ptr<NNTensor> inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_CROP_AXIS);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_CROP_OFFSET);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: crop_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_004, TestSize.Level1)
{
    m_outputs = {2, 3};
    m_params = {4, 5};

    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_CROP_AXIS);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_CROP_OFFSET);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: crop_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: crop_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_006, TestSize.Level1)
{
    SetInputAndShape();

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: crop_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid axis's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_007, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> axisTensor = TransToNNTensor(OH_NN_FLOAT32, m_axisDim,
        nullptr, OH_NN_CROP_AXIS);
    float* axisValue = new (std::nothrow) float [1]{0.0f};
    axisTensor->SetBuffer(&axisValue, sizeof(float));
    m_allTensors.emplace_back(axisTensor);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_CROP_OFFSET);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    axisTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: crop_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid offset's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_008, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_CROP_AXIS);
    std::shared_ptr<NNTensor> offsetTensor = TransToNNTensor(OH_NN_FLOAT32, m_offsetDim,
        nullptr, OH_NN_CROP_OFFSET);
    float* offsetValue = new (std::nothrow) float[1] {1.0f};
    int32_t offsetSize = 1;
    EXPECT_NE(nullptr, offsetValue);
    offsetTensor->SetBuffer(offsetValue, sizeof(float) * offsetSize);
    m_allTensors.emplace_back(offsetTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    offsetTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: crop_build_009
 * @tc.desc: Verify that the build function returns a failed message with passing invalid dataType param.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_009, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_CROP_OFFSET);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: crop_build_010
 * @tc.desc: Verify that the build function returns a failed message with passing invalid value param.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_010, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_CROP_AXIS);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: crop_build_011
 * @tc.desc: Verify that the build function returns a failed message without set buffer for dataType.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_011, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> dataTypeTensor = TransToNNTensor(OH_NN_INT64, m_axisDim,
        nullptr, OH_NN_CROP_AXIS);
    m_allTensors.emplace_back(dataTypeTensor);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_CROP_OFFSET);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: crop_build_012
 * @tc.desc: Verify that the build function returns a failed message without set buffer for value.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_build_012, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);

    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_CROP_AXIS);
    std::shared_ptr<NNTensor> valueTensor = TransToNNTensor(OH_NN_INT64, m_offsetDim,
        nullptr, OH_NN_CROP_OFFSET);
    m_allTensors.emplace_back(valueTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: crop_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_getprimitive_001, TestSize.Level1)
{
    SetInputAndShape();
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_outputDim, nullptr);
    SaveAxis(OH_NN_INT64, m_axisDim, nullptr, OH_NN_CROP_AXIS);
    SaveOffset(OH_NN_INT64, m_offsetDim, nullptr, OH_NN_CROP_OFFSET);

    int64_t axisValue = 0;
    std::vector<int64_t> offsetsValue = {1};
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnAxisValue = mindspore::lite::MindIR_Crop_GetAxis(primitive.get());
    EXPECT_EQ(returnAxisValue, axisValue);
    auto returnOffsets = mindspore::lite::MindIR_Crop_GetOffsets(primitive.get());
    auto returnOffsetsSize = returnOffsets.size();
    for (size_t i = 0; i < returnOffsetsSize; ++i) {
        EXPECT_EQ(returnOffsets[i], offsetsValue[i]);
    }
}

/**
 * @tc.name: crop_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(CropBuilderTest, crop_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}