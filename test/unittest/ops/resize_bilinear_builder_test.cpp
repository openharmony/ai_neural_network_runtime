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

#include "frameworks/native/ops/resize_bilinear_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ResizeBilinearBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveHeightTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveWidthTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveRatioTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveModeTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveOutsideTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetParameterTensor();

protected:
    ResizeBilinearBuilder m_builder;

    std::shared_ptr<NNTensor> heightTensor {nullptr};
    std::shared_ptr<NNTensor> widthTensor {nullptr};
    std::shared_ptr<NNTensor> ratioTensor {nullptr};
    std::shared_ptr<NNTensor> modeTensor {nullptr};
    std::shared_ptr<NNTensor> outsideTensor {nullptr};

    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2, 3, 4, 5, 6};
    std::vector<int32_t> m_dim {1, 2, 2, 2};
    std::vector<int32_t> m_paramDim {};
};

void ResizeBilinearBuilderTest::SetUp() {}

void ResizeBilinearBuilderTest::TearDown() {}

void ResizeBilinearBuilderTest::SaveHeightTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    heightTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t *heightValue = new (std::nothrow) int64_t(1);
    EXPECT_NE(nullptr, heightValue);
    heightTensor->SetBuffer(heightValue, sizeof(int64_t));
    m_allTensors.emplace_back(heightTensor);
}

void ResizeBilinearBuilderTest::SaveWidthTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    widthTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t *widthValue = new (std::nothrow) int64_t(1);
    EXPECT_NE(nullptr, widthValue);
    widthTensor->SetBuffer(widthValue, sizeof(int64_t));
    m_allTensors.emplace_back(widthTensor);
}

void ResizeBilinearBuilderTest::SaveRatioTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    ratioTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool *ratioValue = new (std::nothrow) bool(true);
    EXPECT_NE(nullptr, ratioValue);
    ratioTensor->SetBuffer(ratioValue, sizeof(bool));
    m_allTensors.emplace_back(ratioTensor);
}

void ResizeBilinearBuilderTest::SaveModeTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    modeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t *modeValue = new (std::nothrow) int8_t(1);
    EXPECT_NE(nullptr, modeValue);
    modeTensor->SetBuffer(modeValue, sizeof(int8_t));
    m_allTensors.emplace_back(modeTensor);
}

void ResizeBilinearBuilderTest::SaveOutsideTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    outsideTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t *outsideValue = new (std::nothrow) int64_t(1);
    EXPECT_NE(nullptr, outsideValue);
    outsideTensor->SetBuffer(outsideValue, sizeof(int64_t));
    m_allTensors.emplace_back(outsideTensor);
}

void ResizeBilinearBuilderTest::SetParameterTensor()
{
    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);
}

/**
 * @tc.name: resizebilinear_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SetParameterTensor();

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: resizebilinear_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SetParameterTensor();

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: resizebilinear_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_003, TestSize.Level0)
{
    m_inputs = {0, 1};
    m_outputs = {2};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SetParameterTensor();

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: resizebilinear_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_004, TestSize.Level0)
{
    m_outputs = {1, 2};

    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SetParameterTensor();

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: resizebilinear_build_005
 * @tc.desc: Verify that the build function return a failed message with null allTensor
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: resizebilinear_build_006
 * @tc.desc: Verify that the build function return a failed message without output tensor
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: resizebilinear_build_007
 * @tc.desc: Verify that the build function return a failed message with invalided height's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_007, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    heightTensor = TransToNNTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    int32_t heightValues = 1;
    heightTensor->SetBuffer(&heightValues, sizeof(heightValues));
    m_allTensors.emplace_back(heightTensor);

    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);


    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    heightTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_008
 * @tc.desc: Verify that the build function return a failed message with invalided width's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_008, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);


    widthTensor = TransToNNTensor(OH_NN_INT32, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    int32_t widthValues = 1;
    widthTensor->SetBuffer(&widthValues, sizeof(widthValues));
    m_allTensors.emplace_back(widthTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    widthTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_009
 * @tc.desc: Verify that the build function return a failed message with invalided ratio's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_009, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    ratioTensor = TransToNNTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    int64_t ratioValues = 1;
    ratioTensor->SetBuffer(&ratioValues, sizeof(ratioValues));
    m_allTensors.emplace_back(ratioTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    ratioTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_010
 * @tc.desc: Verify that the build function return a failed message with invalided mode's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_010, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    modeTensor = TransToNNTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    int64_t modeValues = 1;
    modeTensor->SetBuffer(&modeValues, sizeof(modeValues));
    m_allTensors.emplace_back(modeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    modeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_011
 * @tc.desc: Verify that the build function return a failed message with invalided outside's dataType
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_011, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);

    outsideTensor = TransToNNTensor(OH_NN_INT32,
        m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);
    int32_t outsideValues = 1;
    outsideTensor->SetBuffer(&outsideValues, sizeof(outsideValues));
    m_allTensors.emplace_back(outsideTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    outsideTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_012
 * @tc.desc: Verify that the build function return a failed message with invalided height's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_012, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    std::vector<int32_t> heightDim = {2};
    heightTensor = TransToNNTensor(OH_NN_INT64, heightDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    int64_t heightValues[2] = {1, 1};
    heightTensor->SetBuffer(heightValues, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(heightTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    heightTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_013
 * @tc.desc: Verify that the build function return a failed message with invalided width's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_013, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    std::vector<int32_t> widthDim = {2};
    widthTensor = TransToNNTensor(OH_NN_INT64, widthDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    int64_t widthValues[2] = {1, 1};
    widthTensor->SetBuffer(widthValues, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(widthTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    widthTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_014
 * @tc.desc: Verify that the build function return a failed message with invalided ratio's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_014, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    std::vector<int32_t> ratioDim = {2};
    ratioTensor = TransToNNTensor(OH_NN_BOOL,
        ratioDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    bool ratioValues[2] = {true, true};
    ratioTensor->SetBuffer(ratioValues, 2 * sizeof(bool));
    m_allTensors.emplace_back(ratioTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    ratioTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_015
 * @tc.desc: Verify that the build function return a failed message with invalided mode's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_015, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);


    std::vector<int32_t> modeDim = {2};
    modeTensor = TransToNNTensor(OH_NN_INT8,
        modeDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    int8_t modeValues[2] = {1, 1};
    modeTensor->SetBuffer(modeValues, 2 * sizeof(int8_t));
    m_allTensors.emplace_back(modeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    modeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_016
 * @tc.desc: Verify that the build function return a failed message with invalided outside's dimension
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_016, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);

    std::vector<int32_t> outsideDim = {2};
    outsideTensor = TransToNNTensor(OH_NN_INT64,
        outsideDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);
    int64_t outsideValues[2] = {1, 1};
    outsideTensor->SetBuffer(outsideValues, 2 * sizeof(int64_t));
    m_allTensors.emplace_back(outsideTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    outsideTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_017
 * @tc.desc: Verify that the build function return a failed message with invalided parameter
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_017, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: resizebilinear_build_018
 * @tc.desc: Verify that the build function return a failed message with empty height's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_018, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    heightTensor = TransToNNTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    m_allTensors.emplace_back(heightTensor);

    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    heightTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_019
 * @tc.desc: Verify that the build function return a failed message with empty width's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_019, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    widthTensor = TransToNNTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    m_allTensors.emplace_back(widthTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    widthTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_020
 * @tc.desc: Verify that the build function return a failed message with empty ratio's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_020, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    ratioTensor = TransToNNTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    m_allTensors.emplace_back(ratioTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    ratioTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_021
 * @tc.desc: Verify that the build function return a failed message with empty mode's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_021, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveOutsideTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);

    modeTensor = TransToNNTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);
    m_allTensors.emplace_back(modeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    modeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_build_022
 * @tc.desc: Verify that the build function return a failed message with empty outside's buffer
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_build_022, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SaveHeightTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_HEIGHT);
    SaveWidthTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_NEW_WIDTH);
    SaveRatioTensor(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO);
    SaveModeTensor(OH_NN_INT8, m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE);

    outsideTensor = TransToNNTensor(OH_NN_INT64,
        m_paramDim, nullptr, OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE);
    m_allTensors.emplace_back(outsideTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    outsideTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: resizebilinear_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: resizebilinear_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(ResizeBilinearBuilderTest, resizebilinear_get_primitive_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_FLOAT32, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_FLOAT32, m_dim, nullptr);

    SetParameterTensor();

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(primitive, expectPrimitive);

    int64_t heightValue = 1;
    int64_t widthValue = 1;
    bool ratioValue = true;
    int8_t modeValue = 1;
    int64_t outsideValue = 1;

    int64_t heightReturn = mindspore::lite::MindIR_Resize_GetNewHeight(primitive.get());
    EXPECT_EQ(heightReturn, heightValue);
    int64_t widthReturn = mindspore::lite::MindIR_Resize_GetNewWidth(primitive.get());
    EXPECT_EQ(widthReturn, widthValue);
    bool ratioReturn = mindspore::lite::MindIR_Resize_GetPreserveAspectRatio(primitive.get());
    EXPECT_EQ(ratioReturn, ratioValue);
    int8_t modeReturn = mindspore::lite::MindIR_Resize_GetCoordinateTransformMode(primitive.get());
    EXPECT_EQ(modeReturn, modeValue);
    int64_t outsideReturn = mindspore::lite::MindIR_Resize_GetExcludeOutside(primitive.get());
    EXPECT_EQ(outsideReturn, outsideValue);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS