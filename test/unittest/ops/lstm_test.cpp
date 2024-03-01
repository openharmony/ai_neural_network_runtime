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

#include "ops/lstm_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class LSTMBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveBidirectional(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveHasBias(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveInputSize(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveHiddenSize(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveNumLayers(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveNumDirections(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveDropout(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveZoneoutCell(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveZoneoutHidden(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveProjSize(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetInputTensor();
    void SetOutputTensor();

protected:
    LSTMBuilder m_builder;
    std::vector<uint32_t> m_inputs {0, 1, 2, 3, 4, 5};
    std::vector<uint32_t> m_outputs {7, 8, 9};
    std::vector<uint32_t> m_params {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};
    std::vector<int32_t> m_inputDim {2, 2, 2};
    std::vector<int32_t> m_inputXDim {2};
    std::vector<int32_t> m_outputDim {2, 2};
    std::vector<int32_t> m_outputYDim {2};
    std::vector<int32_t> m_paramDim {};
    
    std::shared_ptr<NNTensor> m_tensor {};
};

void LSTMBuilderTest::SetUp() {}

void LSTMBuilderTest::TearDown() {}

void LSTMBuilderTest::SaveBidirectional(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> bidirectionalTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool* bidirectionalValue = new (std::nothrow) bool(false);
    EXPECT_NE(nullptr, bidirectionalValue);
    bidirectionalTensor->SetBuffer(bidirectionalValue, sizeof(bool));
    m_allTensors.emplace_back(bidirectionalTensor);
}

void LSTMBuilderTest::SaveHasBias(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> hasBiasTensor = TransToNNTensor(dataType, dim, quantParam, type);
    bool* hasBiasValue = new (std::nothrow) bool(false);
    EXPECT_NE(nullptr, hasBiasValue);
    hasBiasTensor->SetBuffer(hasBiasValue, sizeof(bool));
    m_allTensors.emplace_back(hasBiasTensor);
}

void LSTMBuilderTest::SaveInputSize(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> inputSizeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* inputSizeValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, inputSizeValue);
    inputSizeTensor->SetBuffer(inputSizeValue, sizeof(int64_t));
    m_allTensors.emplace_back(inputSizeTensor);
}

void LSTMBuilderTest::SaveHiddenSize(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> hiddenSizeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* hiddenSizeValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, hiddenSizeValue);
    hiddenSizeTensor->SetBuffer(hiddenSizeValue, sizeof(int64_t));
    m_allTensors.emplace_back(hiddenSizeTensor);
}

void LSTMBuilderTest::SaveNumLayers(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> numLayersTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* numLayersValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, numLayersValue);
    numLayersTensor->SetBuffer(numLayersValue, sizeof(int64_t));
    m_allTensors.emplace_back(numLayersTensor);
}

void LSTMBuilderTest::SaveNumDirections(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> numDirectionsTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* numDirectionsValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, numDirectionsValue);
    numDirectionsTensor->SetBuffer(numDirectionsValue, sizeof(int64_t));
    m_allTensors.emplace_back(numDirectionsTensor);
}

void LSTMBuilderTest::SaveDropout(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> dropoutTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* dropoutValue = new (std::nothrow) float(0.0f);
    EXPECT_NE(nullptr, dropoutValue);
    dropoutTensor->SetBuffer(dropoutValue, sizeof(float));
    m_allTensors.emplace_back(dropoutTensor);
}

void LSTMBuilderTest::SaveZoneoutCell(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> zoneoutCellTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* zoneoutCellValue = new (std::nothrow) float(0.0f);
    EXPECT_NE(nullptr, zoneoutCellValue);
    zoneoutCellTensor->SetBuffer(zoneoutCellValue, sizeof(float));
    m_allTensors.emplace_back(zoneoutCellTensor);
}

void LSTMBuilderTest::SaveZoneoutHidden(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> zoneoutHiddenTensor = TransToNNTensor(dataType, dim, quantParam, type);
    float* zoneoutHiddenValue = new (std::nothrow) float(0.0f);
    EXPECT_NE(nullptr, zoneoutHiddenValue);
    zoneoutHiddenTensor->SetBuffer(zoneoutHiddenValue, sizeof(float));
    m_allTensors.emplace_back(zoneoutHiddenTensor);
}

void LSTMBuilderTest::SaveProjSize(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> projSizeTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* projSizeValue = new (std::nothrow) int64_t(0.0f);
    EXPECT_NE(nullptr, projSizeValue);
    projSizeTensor->SetBuffer(projSizeValue, sizeof(int64_t));
    m_allTensors.emplace_back(projSizeTensor);
}

void LSTMBuilderTest::SetInputTensor()
{
    m_inputsIndex = m_inputs;
    std::shared_ptr<NNTensor> inputTensor;
    inputTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(inputTensor);

    std::shared_ptr<NNTensor> hxTensor;
    hxTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputXDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(hxTensor);

    std::shared_ptr<NNTensor> cxTensor;
    cxTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputXDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(cxTensor);

    std::shared_ptr<NNTensor> wihTensor;
    wihTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputXDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(wihTensor);

    std::shared_ptr<NNTensor> whhTensor;
    whhTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputXDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(whhTensor);

    std::shared_ptr<NNTensor> bihTensor;
    bihTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputXDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(bihTensor);

    std::shared_ptr<NNTensor> bhhTensor;
    bhhTensor = TransToNNTensor(OH_NN_FLOAT32, m_inputXDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(bhhTensor);
}

void LSTMBuilderTest::SetOutputTensor()
{
    m_outputsIndex = m_outputs;
    std::shared_ptr<NNTensor> outputTensor;
    outputTensor = TransToNNTensor(OH_NN_FLOAT32, m_outputDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(outputTensor);

    std::shared_ptr<NNTensor> hyTensor;
    hyTensor = TransToNNTensor(OH_NN_FLOAT32, m_outputYDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(hyTensor);

    std::shared_ptr<NNTensor> cyTensor;
    cyTensor = TransToNNTensor(OH_NN_FLOAT32, m_outputYDim, nullptr, OH_NN_TENSOR);
    m_allTensors.emplace_back(cyTensor);
}

/**
 * @tc.name: LSTM_build_001
 * @tc.desc: Verify that the build function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_001, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: LSTM_build_002
 * @tc.desc: Verify that the build function returns a failed message with true m_isBuild.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_002, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: LSTM_build_003
 * @tc.desc: Verify that the build function returns a failed message with invalided input.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_003, TestSize.Level1)
{
    m_inputs = {0, 1, 2, 3, 4, 5, 6};
    m_outputs = {7, 8, 9};
    m_params = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_004
 * @tc.desc: Verify that the build function returns a failed message with invalided output.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_004, TestSize.Level1)
{
    m_outputs = {6, 7, 8, 9};
    m_params = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_005
 * @tc.desc: Verify that the build function returns a failed message with empty allTensor.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_005, TestSize.Level1)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_006
 * @tc.desc: Verify that the build function returns a failed message without output tensor.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_006, TestSize.Level1)
{
    SetInputTensor();

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_007
 * @tc.desc: Verify that the build function returns a failed message with invalid bidirectional's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_007, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    std::shared_ptr<NNTensor> bidirectionalTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    int64_t* bidirectionalValue = new (std::nothrow) int64_t [1]{0};
    EXPECT_NE(nullptr, bidirectionalValue);
    bidirectionalTensor->SetBuffer(bidirectionalValue, sizeof(int64_t));
    m_allTensors.emplace_back(bidirectionalTensor);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    bidirectionalTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_008
 * @tc.desc: Verify that the build function returns a failed message with invalid has_bias's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_008, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    std::shared_ptr<NNTensor> hasBiasTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_HAS_BIAS);
    int64_t* hasBiasValue = new (std::nothrow) int64_t [1]{0};
    EXPECT_NE(nullptr, hasBiasValue);
    hasBiasTensor->SetBuffer(hasBiasValue, sizeof(int64_t));
    m_allTensors.emplace_back(hasBiasTensor);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    hasBiasTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_009
 * @tc.desc: Verify that the build function returns a failed message with invalid input_size's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_009, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    std::shared_ptr<NNTensor> inputSizeTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LSTM_INPUT_SIZE);
    float* inputSizeValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, inputSizeValue);
    inputSizeTensor->SetBuffer(inputSizeValue, sizeof(float));
    m_allTensors.emplace_back(inputSizeTensor);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    inputSizeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_010
 * @tc.desc: Verify that the build function returns a failed message with invalid hidden_size's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_010, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    std::shared_ptr<NNTensor> hiddenSizeTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    float* hiddenSizeValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, hiddenSizeValue);
    hiddenSizeTensor->SetBuffer(hiddenSizeValue, sizeof(float));
    m_allTensors.emplace_back(hiddenSizeTensor);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    hiddenSizeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_011
 * @tc.desc: Verify that the build function returns a failed message with invalid num_layers's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_011, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    std::shared_ptr<NNTensor> numLayersTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LSTM_NUM_LAYERS);
    float* numLayersValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, numLayersValue);
    numLayersTensor->SetBuffer(numLayersValue, sizeof(float));
    m_allTensors.emplace_back(numLayersTensor);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    numLayersTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_012
 * @tc.desc: Verify that the build function returns a failed message with invalid num_directions's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_012, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    std::shared_ptr<NNTensor> numDirectionsTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    float* numDirectionsValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, numDirectionsValue);
    numDirectionsTensor->SetBuffer(numDirectionsValue, sizeof(float));
    m_allTensors.emplace_back(numDirectionsTensor);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    numDirectionsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_013
 * @tc.desc: Verify that the build function returns a failed message with invalid dropout's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_013, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    std::shared_ptr<NNTensor> dropoutTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_DROPOUT);
    int64_t* dropoutValue = new (std::nothrow) int64_t [1]{0};
    EXPECT_NE(nullptr, dropoutValue);
    dropoutTensor->SetBuffer(dropoutValue, sizeof(int64_t));
    m_allTensors.emplace_back(dropoutTensor);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    dropoutTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_014
 * @tc.desc: Verify that the build function returns a failed message with invalid zoneout_cell's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_014, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    std::shared_ptr<NNTensor> zoneoutCellTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    int64_t* zoneoutCellValue = new (std::nothrow) int64_t [1]{0};
    EXPECT_NE(nullptr, zoneoutCellValue);
    zoneoutCellTensor->SetBuffer(zoneoutCellValue, sizeof(int64_t));
    m_allTensors.emplace_back(zoneoutCellTensor);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    zoneoutCellTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_015
 * @tc.desc: Verify that the build function returns a failed message with invalid zoneout_hidden's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_015, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    std::shared_ptr<NNTensor> zoneoutHiddenTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    int64_t* zoneoutHiddenValue = new (std::nothrow) int64_t [1]{0};
    EXPECT_NE(nullptr, zoneoutHiddenValue);
    zoneoutHiddenTensor->SetBuffer(zoneoutHiddenValue, sizeof(int64_t));
    m_allTensors.emplace_back(zoneoutHiddenTensor);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    zoneoutHiddenTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_016
 * @tc.desc: Verify that the build function returns a failed message with invalid proj_size's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_016, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    std::shared_ptr<NNTensor> projSizeTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LSTM_PROJ_SIZE);
    float* projSizeValue = new (std::nothrow) float [1]{0.0f};
    EXPECT_NE(nullptr, projSizeValue);
    projSizeTensor->SetBuffer(projSizeValue, sizeof(float));
    m_allTensors.emplace_back(projSizeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    projSizeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_017
 * @tc.desc: Verify that the build function returns a failed message with passing invalid bidirectional param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_017, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_018
 * @tc.desc: Verify that the build function returns a failed message with passing invalid has_bias param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_018, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_019
 * @tc.desc: Verify that the build function returns a failed message with passing invalid input_size param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_019, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_020
 * @tc.desc: Verify that the build function returns a failed message with passing invalid hidden_size param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_020, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_021
 * @tc.desc: Verify that the build function returns a failed message with passing invalid num_layers param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_021, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_022
 * @tc.desc: Verify that the build function returns a failed message with passing invalid num_directions param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_022, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_023
 * @tc.desc: Verify that the build function returns a failed message with passing invalid dropout param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_023, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_024
 * @tc.desc: Verify that the build function returns a failed message with passing invalid zoneout_cell param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_024, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_025
 * @tc.desc: Verify that the build function returns a failed message with passing invalid zoneout_cell param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_025, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_026
 * @tc.desc: Verify that the build function returns a failed message with passing invalid zoneout_hidden param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_026, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_027
 * @tc.desc: Verify that the build function returns a failed message with passing invalid proj_size param.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_027, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_MUL_ACTIVATION_TYPE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_028
 * @tc.desc: Verify that the build function returns a failed message without set buffer for bidirectional.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_028, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    std::shared_ptr<NNTensor> bidirectionalTensor = TransToNNTensor(OH_NN_BOOL, m_paramDim,
        nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    m_allTensors.emplace_back(bidirectionalTensor);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_029
 * @tc.desc: Verify that the build function returns a failed message without set buffer for has_bias.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_029, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    std::shared_ptr<NNTensor> hasBiasTensor = TransToNNTensor(OH_NN_BOOL, m_paramDim,
        nullptr, OH_NN_LSTM_HAS_BIAS);
    m_allTensors.emplace_back(hasBiasTensor);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_030
 * @tc.desc: Verify that the build function returns a failed message without set buffer for input_size.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_030, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    std::shared_ptr<NNTensor> inputSizeTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_INPUT_SIZE);
    m_allTensors.emplace_back(inputSizeTensor);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_031
 * @tc.desc: Verify that the build function returns a failed message without set buffer for hidden_size.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_031, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    std::shared_ptr<NNTensor> hiddenSizeTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    m_allTensors.emplace_back(hiddenSizeTensor);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    hiddenSizeTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_032
 * @tc.desc: Verify that the build function returns a failed message without set buffer for num_layers.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_032, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    std::shared_ptr<NNTensor> numLayersTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_NUM_LAYERS);
    m_allTensors.emplace_back(numLayersTensor);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    numLayersTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_033
 * @tc.desc: Verify that the build function returns a failed message without set buffer for num_directions.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_033, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    std::shared_ptr<NNTensor> numDirectionsTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    m_allTensors.emplace_back(numDirectionsTensor);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    numDirectionsTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: LSTM_build_034
 * @tc.desc: Verify that the build function returns a failed message without set buffer for dropout.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_034, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    std::shared_ptr<NNTensor> dropoutTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LSTM_DROPOUT);
    m_allTensors.emplace_back(dropoutTensor);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_035
 * @tc.desc: Verify that the build function returns a failed message without set buffer for zoneout_cell.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_035, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    std::shared_ptr<NNTensor> zoneoutCellTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    m_allTensors.emplace_back(zoneoutCellTensor);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_036
 * @tc.desc: Verify that the build function returns a failed message without set buffer for zoneout_hidden.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_036, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    std::shared_ptr<NNTensor> zoneoutHiddenTensor = TransToNNTensor(OH_NN_FLOAT32, m_paramDim,
        nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    m_allTensors.emplace_back(zoneoutHiddenTensor);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_build_037
 * @tc.desc: Verify that the build function returns a failed message without set buffer for proj_size.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_build_037, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    std::shared_ptr<NNTensor> projSizeTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_LSTM_PROJ_SIZE);
    m_allTensors.emplace_back(projSizeTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: LSTM_getprimitive_001
 * @tc.desc: Verify that the getPrimitive function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_getprimitive_001, TestSize.Level1)
{
    SetInputTensor();
    SetOutputTensor();

    SaveBidirectional(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_BIDIRECTIONAL);
    SaveHasBias(OH_NN_BOOL, m_paramDim, nullptr, OH_NN_LSTM_HAS_BIAS);
    SaveInputSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_INPUT_SIZE);
    SaveHiddenSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_HIDDEN_SIZE);
    SaveNumLayers(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_LAYERS);
    SaveNumDirections(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_NUM_DIRECTIONS);
    SaveDropout(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_DROPOUT);
    SaveZoneoutCell(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_CELL);
    SaveZoneoutHidden(OH_NN_FLOAT32, m_paramDim, nullptr, OH_NN_LSTM_ZONEOUT_HIDDEN);
    SaveProjSize(OH_NN_INT64, m_paramDim, nullptr, OH_NN_LSTM_PROJ_SIZE);

    bool bidirectionalValue = false;
    bool hasBiasValue = false;
    int64_t inputSizeValue = 0;
    int64_t hiddenSizeValue = 0;
    int64_t numLayersValue = 0;
    int64_t numDirectionsValue = 0;
    float dropoutValue = 0.0f;
    float zoneoutCellValue = 0.0f;
    float zoneoutHiddenValue = 0.0f;
    int64_t projSizeValue = 0;

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_NE(expectPrimitive, primitive);

    auto returnBidirectionalValue = mindspore::lite::MindIR_LSTM_GetBidirectional(primitive.get());
    EXPECT_EQ(returnBidirectionalValue, bidirectionalValue);
    auto returnHasBiasValue = mindspore::lite::MindIR_LSTM_GetHasBias(primitive.get());
    EXPECT_EQ(returnHasBiasValue, hasBiasValue);
    auto returnInputSizeValue = mindspore::lite::MindIR_LSTM_GetInputSize(primitive.get());
    EXPECT_EQ(returnInputSizeValue, inputSizeValue);
    auto returnHiddenSizeValue = mindspore::lite::MindIR_LSTM_GetHiddenSize(primitive.get());
    EXPECT_EQ(returnHiddenSizeValue, hiddenSizeValue);
    auto returnNumLayersValue = mindspore::lite::MindIR_LSTM_GetNumLayers(primitive.get());
    EXPECT_EQ(returnNumLayersValue, numLayersValue);
    auto returnNumDirectionsValue = mindspore::lite::MindIR_LSTM_GetNumDirections(primitive.get());
    EXPECT_EQ(returnNumDirectionsValue, numDirectionsValue);
    auto returnDropoutValue = mindspore::lite::MindIR_LSTM_GetDropout(primitive.get());
    EXPECT_EQ(returnDropoutValue, dropoutValue);
    auto returnZoneoutCellValue = mindspore::lite::MindIR_LSTM_GetZoneoutCell(primitive.get());
    EXPECT_EQ(returnZoneoutCellValue, zoneoutCellValue);
    auto returnZoneoutHiddenValue = mindspore::lite::MindIR_LSTM_GetZoneoutHidden(primitive.get());
    EXPECT_EQ(returnZoneoutHiddenValue, zoneoutHiddenValue);
    auto returnProjSizeValue = mindspore::lite::MindIR_LSTM_GetProjSize(primitive.get());
    EXPECT_EQ(returnProjSizeValue, projSizeValue);
}

/**
 * @tc.name: LSTM_getprimitive_002
 * @tc.desc: Verify that the getPrimitive function returns a failed message without build.
 * @tc.type: FUNC
 */
HWTEST_F(LSTMBuilderTest, LSTM_getprimitive_002, TestSize.Level1)
{
    LiteGraphPrimitvePtr primitive = m_builder.GetPrimitive();
    LiteGraphPrimitvePtr expectPrimitive(nullptr, DestroyLiteGraphPrimitive);
    EXPECT_EQ(expectPrimitive, primitive);
}
}
}
}