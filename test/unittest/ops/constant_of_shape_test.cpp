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

#include "ops/constant_of_shape_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ConstantOfShapeBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveDataType(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveValue(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    ConstantOfShapeBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2, 3};
    std::vector<int32_t> m_inputDim {3};
    std::vector<int32_t> m_outputDim {3};
    std::vector<int32_t> m_dataTypeDim {};
    std::vector<int32_t> m_valueDim {1};
};

void ConstantOfShapeBuilderTest::SetUp() {}

void ConstantOfShapeBuilderTest::TearDown() {}

void ConstantOfShapeBuilderTest::SaveDataType(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> dataTypeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* dataTypeValue = new (std::nothrow) int64_t[1] {0};
    dataTypeTensor->SetBuffer(dataTypeValue, sizeof(int64_t));
    m_allTensors.emplace_back(dataTypeTensor);
}

void ConstantOfShapeBuilderTest::SaveValue(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> valueTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* valueValue = new (std::nothrow) float[1] {1.0f};
    int32_t valueSize = 1;
    EXPECT_NE(nullptr, valueValue);
    valueTensor->SetBuffer(valueValue, sizeof(float) * valueSize);
    m_allTensors.emplace_back(valueTensor);
}

/**
 * @tc.name: constant_of_shape_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: constant_of_shape_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_002, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: constant_of_shape_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_003, TestSize.Level1)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3, 4};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: constant_of_shape_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_004, TestSize.Level1)
{
    m_outputs = {1, 2};
    m_params = {3, 4};

    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: constant_of_shape_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: constant_of_shape_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_006, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: constant_of_shape_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid dataType's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_007, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> dataTypeTensor = TransToNNTensor(OH_NN_FLOAT32, m_dataTypeDim,
        nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    float* dataTypeValue = new (std::nothrow) float [1]{0.0f};
    dataTypeTensor->SetBuffer(&dataTypeValue, sizeof(float));
    m_allTensors.emplace_back(dataTypeTensor);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    dataTypeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: constant_of_shape_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid value's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_008, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);

    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    std::shared_ptr<NNTensor> valueTensor = TransToNNTensor(OH_NN_INT64, m_valueDim,
        nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);
    int64_t* valueValue = new (std::nothrow) int64_t[1] {1};
    int32_t valueSize = 1;
    EXPECT_NE(nullptr, valueValue);
    valueTensor->SetBuffer(valueValue, sizeof(float) * valueSize);
    m_allTensors.emplace_back(valueTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    valueTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: constant_of_shape_build_009
 * @tc.desc: Verify that the build function returns a failed message with passing invalid dataType param.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_009, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: constant_of_shape_build_010
 * @tc.desc: Verify that the build function returns a failed message with passing invalid value param.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_010, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: constant_of_shape_build_011
 * @tc.desc: Verify that the build function returns a failed message without set buffer for dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_011, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);

    std::shared_ptr<NNTensor> dataTypeTensor = TransToNNTensor(OH_NN_INT64, m_dataTypeDim,
        nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    m_allTensors.emplace_back(dataTypeTensor);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: constant_of_shape_build_012
 * @tc.desc: Verify that the build function returns a failed message without set buffer for value.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_build_012, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);

    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    std::shared_ptr<NNTensor> valueTensor = TransToNNTensor(OH_NN_FLOAT32, m_valueDim,
        nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);
    m_allTensors.emplace_back(valueTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: constant_of_shape_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_getprimitive_001, TestSize.Level1)
{
    SaveInputTensor(m_inputs, OH_NN_INT32, m_inputDim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT32, m_outputDim, nullptr);
    SaveDataType(OH_NN_INT64, m_dataTypeDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_DATA_TYPE);
    SaveValue(OH_NN_FLOAT32, m_valueDim, nullptr, OH_NN_CONSTANT_OF_SHAPE_VALUE);

    int64_t dataTypeValue = 0;
    std::vector<float> valueValue = {1.0f};
    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnDataTypeValue = mindspore::lite::MindIR_ConstantOfShape_GetDataType(primitive.get());
    EXPECT_EQ(returnDataTypeValue, dataTypeValue);
    auto returnValue = mindspore::lite::MindIR_ConstantOfShape_GetValue(primitive.get());
    auto returnValueSize = returnValue.size();
    for (size_t i = 0; i < returnValueSize; ++i) {
        EXPECT_EQ(returnValue[i], valueValue[i]);
    }
}

/**
 * @tc.name: constant_of_shape_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(ConstantOfShapeBuilderTest, constant_of_shape_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}