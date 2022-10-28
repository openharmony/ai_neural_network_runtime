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

#include "executor_test.h"

#include "common/scoped_trace.h"
#include "frameworks/native/compilation.h"
#include "frameworks/native/inner_model.h"
#include "test/unittest/common/mock_idevice.h"

using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Ops;
using namespace OHOS::HDI::Nnrt::V1_0;
using namespace OHOS::HiviewDFX;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
using NNTensorPtr = std::shared_ptr<NNTensor>;

MSLITE::LiteGraph* ExecutorTest::BuildLiteGraph(const std::vector<int32_t> dim, const std::vector<int32_t> dimOut)
{
    MSLITE::LiteGraph* liteGraph = new (std::nothrow) MSLITE::LiteGraph();
    if (liteGraph == nullptr) {
        LOGE("liteGraph build failed");
        return nullptr;
    }
    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_.emplace_back(0);
    liteGraph->output_indices_.emplace_back(1);
    const std::vector<MSLITE::QuantParam> quant_params;

    for (size_t indexInput = 0; indexInput < liteGraph->input_indices_.size(); ++indexInput) {
        const std::vector<uint8_t> data(36, 1);
        void* liteGraphTensor1 = MSLITE::MindIR_Tensor_Create(liteGraph->name_,
            MSLITE::DATA_TYPE_FLOAT32, dim, MSLITE::FORMAT_NCHW, data, quant_params);
        liteGraph->all_tensors_.emplace_back(liteGraphTensor1);
    }

    for (size_t indexOutput = 0; indexOutput < liteGraph->output_indices_.size(); ++indexOutput) {
        const std::vector<uint8_t> dataOut(36, 1);
        void* liteGraphTensor2 = MSLITE::MindIR_Tensor_Create(liteGraph->name_,
            MSLITE::DATA_TYPE_FLOAT32, dimOut, MSLITE::FORMAT_NCHW, dataOut, quant_params);
        liteGraph->all_tensors_.emplace_back(liteGraphTensor2);
    }

    return liteGraph;
}

OH_NN_Tensor ExecutorTest::SetTensor(OH_NN_DataType dataType, uint32_t dimensionCount, const int32_t *dimensions,
    const OH_NN_QuantParam *quantParam, OH_NN_TensorType type)
{
    OH_NN_Tensor tensor;
    tensor.dataType = dataType;
    tensor.dimensionCount = dimensionCount;
    tensor.dimensions = dimensions;
    tensor.quantParam = quantParam;
    tensor.type = type;

    return tensor;
}

void ExecutorTest::SetMermory(OH_NN_Memory** &memory)
{
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;
}

/*
 * @tc.name: executor_set_input_001
 * @tc.desc: Verify that the SetInput function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_001, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_set_input_002
 * @tc.desc: Verify that the SetInput function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_002, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    m_index = 6;
    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_003
 * @tc.desc: Verify that the SetInput function returns a failed message with dynamic shape.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_003, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    const int dim = -1;
    m_dimensionCount = 1;
    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, &dim, nullptr, OH_NN_TENSOR);
    size_t length = 1 * sizeof(float);
    float data = 0;
    void* buffer = &data;

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_004
 * @tc.desc: Verify that the SetInput function returns a failed message with invalid tensor's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_004, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_INT64, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_005
 * @tc.desc: Verify that the SetInput function returns a failed message with invalid length.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_005, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);


    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    size_t length = 1 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_006
 * @tc.desc: Verify that the SetInput function returns a failed message with allocating buffer is unsuccessfully.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_006, testing::ext::TestSize.Level0)
{
    HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_INVALID_PARAMETER;
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}

/*
 * @tc.name: executor_set_input_007
 * @tc.desc: Verify that the SetInput function returns a failed message with empty buffer.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_007, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = nullptr;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}

/*
 * @tc.name: executor_set_input_008
 * @tc.desc: Verify that the SetInput function returns a successful message with dataLength <= curBufferLength.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_008, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    float dataArry[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    void* buffer = dataArry;
    size_t length = 9 * sizeof(float);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInput(m_index, tensor, buffer, length));

    float expectArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* expectBuffer = expectArry;
    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, expectBuffer, length);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_set_input_009
 * @tc.desc: Verify that the SetInput function returns a failed message with length less than dataLength.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_009, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInputFromMemory(m_index, tensor, memory));

    float expectData = 0;
    void* buffer = &expectData;
    size_t length = 1 * sizeof(float);

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_010
 * @tc.desc: Verify that the SetInput function returns a failed message with BuildFromOHNNTensor unsuccessfully.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_010, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    m_dimensionCount = 0;
    OH_NN_Tensor tensor = SetTensor(OH_NN_UNKNOWN, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_011
 * @tc.desc: Verify that the SetInput function returns a successful message with dataLength <= curBufferLength.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_011, testing::ext::TestSize.Level0)
{
    const std::vector<int32_t> expectDim = {3, -1};
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(expectDim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInput(m_index, tensor, buffer, length));

    const int32_t testDim[2] = {3, 5};
    OH_NN_Tensor expectTensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, testDim, nullptr, OH_NN_TENSOR);
    size_t expectLength = 15 * sizeof(float);
    float expectArry[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    void* expectBuffer = expectArry;
    OH_NN_ReturnCode ret = executorTest.SetInput(m_index, expectTensor, expectBuffer, expectLength);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_set_input_from_memory_001
 * @tc.desc: Verify that the SetInputFromMemory function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_from_memory_001, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetInputFromMemory(m_index, tensor, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_set_input_from_memory_002
 * @tc.desc: Verify that the SetInputFromMemory function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_from_memory_002, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    m_index = 6;
    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetInputFromMemory(m_index, tensor, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_from_memory_003
 * @tc.desc: Verify that the SetInputFromMemory function returns a failed message with dynamic shape.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_from_memory_003, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    const int dim = -1;
    OH_NN_Tensor tensor;
    tensor.dataType = OH_NN_FLOAT32;
    tensor.dimensionCount = 1;
    tensor.dimensions = &dim;
    tensor.quantParam = nullptr;
    tensor.type = OH_NN_TENSOR;
    float value = 0;
    void* const data = &value;
    OH_NN_Memory memory = {data, 1 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetInputFromMemory(m_index, tensor, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_from_memory_004
 * @tc.desc: Verify that the SetInputFromMemory function returns a failed message with invalid tensor's dataType.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_from_memory_004, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_INT64, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetInputFromMemory(m_index, tensor, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_from_memory_005
 * @tc.desc: Verify that the SetInput function returns a failed message with invalid memory.length.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_input_from_memory_005, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 1 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetInputFromMemory(m_index, tensor, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_001
 * @tc.desc: Verify that the SetOutput function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_001, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = executorTest.SetOutput(m_index, buffer, length);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_set_output_002
 * @tc.desc: Verify that the SetOutput function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    m_index = 6;
    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = executorTest.SetOutput(m_index, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_003
 * @tc.desc: Verify that the SetOutput function returns a failed message with invalid length.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 2 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = executorTest.SetOutput(m_index, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_004
 * @tc.desc: Verify that the SetOutput function returns a failed message with allocating buffer is failed.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_004, testing::ext::TestSize.Level0)
{
    HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_INVALID_PARAMETER;
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = executorTest.SetOutput(m_index, buffer, length);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}

/*
 * @tc.name: executor_set_output_005
 * @tc.desc: Verify that the SetOutput function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_005, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutputFromMemory(m_index, memory));

    size_t length = 1 * sizeof(float);
    float expectData = 0;
    void* buffer = &expectData;
    OH_NN_ReturnCode ret = executorTest.SetOutput(m_index, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_006
 * @tc.desc: Verify that the SetOutput function returns a successful message with length <= curBufferLength.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_006, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutput(m_index, buffer, length));

    float expectDataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* expectBuffer = expectDataArry;
    OH_NN_ReturnCode ret = executorTest.SetOutput(m_index, expectBuffer, length);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_set_output_007
 * @tc.desc: Verify that the SetOutput function returns a successful message with length > curBufferLength.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_007, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutput(m_index, buffer, length));

    size_t expectLength = 15 * sizeof(float);
    float expectDataArry[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    void* expectBuffer = expectDataArry;
    OH_NN_ReturnCode ret = executorTest.SetOutput(m_index, expectBuffer, expectLength);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_set_output_from_memory_001
 * @tc.desc: Verify that the SetOutputFromMemory function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_from_memory_001, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetOutputFromMemory(m_index, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_set_output_from_memory_002
 * @tc.desc: Verify that the SetOutputFromMemory function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_from_memory_002, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    m_index = 6;
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetOutputFromMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_from_memory_003
 * @tc.desc: Verify that the SetOutputFromMemory function returns a failed message with invalid memory.length.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_from_memory_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 0};

    OH_NN_ReturnCode ret = executorTest.SetOutputFromMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_from_memory_004
 * @tc.desc: Verify that the SetOutputFromMemory function returns a failed message with memory.length < dataLength.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_from_memory_004, testing::ext::TestSize.Level0)
{
    const std::vector<int32_t> expectDim = {4, 4};
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, expectDim);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetOutputFromMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_from_memory_005
 * @tc.desc: Verify that the SetOutputFromMemory function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_set_output_from_memory_005, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutput(m_index, buffer, length));
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = executorTest.SetOutputFromMemory(m_index, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_get_output_dimensions_001
 * @tc.desc: Verify that the GetOutputShape function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_get_output_dimensions_001, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInput(m_index, tensor, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutput(m_index, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.Run());

    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    int32_t** dimensions = &ptr;
    uint32_t dimensionCount = 2;

    OH_NN_ReturnCode ret = executorTest.GetOutputShape(m_index, dimensions, dimensionCount);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_get_output_dimensions_002
 * @tc.desc: Verify that the GetOutputShape function returns a failed message without run.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_get_output_dimensions_002, testing::ext::TestSize.Level0)
{
    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInput(m_index, tensor, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutput(m_index, buffer, length));

    int32_t testDim[2] = {3, 3};
    int32_t* ptr = testDim;
    int32_t** dimensions = &ptr;
    uint32_t dimensionCount = 2;

    OH_NN_ReturnCode ret = executorTest.GetOutputShape(m_index, dimensions, dimensionCount);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: executor_get_output_dimensions_003
 * @tc.desc: Verify that the GetOutputShape function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_get_output_dimensions_003, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);
    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInput(m_index, tensor, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutput(m_index, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.Run());

    uint32_t testIndex = 6;
    int32_t testDim[2] = {3, 3};
    int32_t* ptr = testDim;
    int32_t** dimensions = &ptr;
    uint32_t dimensionCount = 2;

    OH_NN_ReturnCode ret = executorTest.GetOutputShape(testIndex, dimensions, dimensionCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_create_input_memory_001
 * @tc.desc: Verify that the CreateInputMemory function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_create_input_memory_001, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = executorTest.CreateInputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_create_input_memory_002
 * @tc.desc: Verify that the CreateInputMemory function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_create_input_memory_002, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    m_index = 6;
    OH_NN_ReturnCode ret = executorTest.CreateInputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_create_input_memory_003
 * @tc.desc: Verify that the CreateInputMemory function returns a failed message with allocating buffer unsuccessfully.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_create_input_memory_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_INVALID_PARAMETER;
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    OH_NN_ReturnCode ret = executorTest.CreateInputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}

/*
 * @tc.name: executor_destroy_input_memory_001
 * @tc.desc: Verify that the DestroyInputMemory function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_destroy_input_memory_001, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    OH_NN_Memory** memory = &ptr;

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.CreateInputMemory(m_index, length, memory));
    OH_NN_ReturnCode ret = executorTest.DestroyInputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_destroy_input_memory_002
 * @tc.desc: Verify that the DestroyInputMemory function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_destroy_input_memory_002, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    uint32_t testIndex = 6;
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.CreateInputMemory(m_index, length, memory));
    OH_NN_ReturnCode ret = executorTest.DestroyInputMemory(testIndex, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_destroy_input_memory_003
 * @tc.desc: Verify that the DestroyInputMemory function returns a failed message without creating memory.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_destroy_input_memory_003, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    OH_NN_ReturnCode ret = executorTest.DestroyInputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_destroy_input_memory_004
 * @tc.desc: Verify that the DestroyInputMemory function returns a failed message with invalid memory.data.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_destroy_input_memory_004, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.CreateInputMemory(m_index, length, memory));

    float arry[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    void* const expectData = arry;
    OH_NN_Memory mptr = {expectData, 9 * sizeof(float)};
    OH_NN_Memory* expectPtr = &mptr;
    OH_NN_Memory** expectMemory = &expectPtr;

    OH_NN_ReturnCode ret = executorTest.DestroyInputMemory(m_index, expectMemory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_create_output_memory_001
 * @tc.desc: Verify that the CreateOutputMemory function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_create_output_memory_001, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    OH_NN_ReturnCode ret = executorTest.CreateOutputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_create_output_memory_002
 * @tc.desc:  Verify that the CreateOutputMemory function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_create_output_memory_002, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    m_index = 6;
    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    OH_NN_ReturnCode ret = executorTest.CreateOutputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_create_output_memory_003
 * @tc.desc: Verify that the CreateOutputMemory function returns a failed message with allocating buffer unsuccessfully.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_create_output_memory_003, testing::ext::TestSize.Level0)
{
    HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_INVALID_PARAMETER;
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    OH_NN_ReturnCode ret = executorTest.CreateOutputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}

/*
 * @tc.name: executor_destroy_output_memory_001
 * @tc.desc: Verify that the DestroyOutputMemory function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_destroy_output_memory_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.CreateOutputMemory(m_index, length, memory));
    OH_NN_ReturnCode ret = executorTest.DestroyOutputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_destroy_output_memory_002
 * @tc.desc: Verify that the DestroyOutputMemory function returns a failed message with out-of-range index.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_destroy_output_memory_002, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    uint32_t testIndex = 6;
    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.CreateOutputMemory(m_index, length, memory));
    OH_NN_ReturnCode ret = executorTest.DestroyOutputMemory(testIndex, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_destroy_output_memory_003
 * @tc.desc: Verify that the DestroyOutputMemory function returns a failed message without creating memory.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_destroy_output_memory_003, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    OH_NN_ReturnCode ret = executorTest.DestroyOutputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_destroy_output_memory_004
 * @tc.desc: Verify that the DestroyOutputMemory function returns a failed message with invalid memory.data.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_destroy_output_memory_004, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    SetMermory(memory);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.CreateOutputMemory(m_index, length, memory));

    float arry[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    void* const expectData = arry;
    OH_NN_Memory mptr = {expectData, 9 * sizeof(float)};
    OH_NN_Memory* expectPtr = &mptr;
    OH_NN_Memory** expectMemory = &expectPtr;

    OH_NN_ReturnCode ret = executorTest.DestroyOutputMemory(m_index, expectMemory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_run_test_001
 * @tc.desc: Verify that the Run function returns a successful message.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_run_test_001, testing::ext::TestSize.Level0)
{
    HiviewDFX::HiTraceId traceId = HiTraceChain::Begin("executor_run_test_001", HITRACE_FLAG_TP_INFO);
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    size_t length = 9 * sizeof(float);
    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInput(m_index, tensor, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutput(m_index, buffer, length));
    OH_NN_ReturnCode ret = executorTest.Run();
    HiTraceChain::End(traceId);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_run_test_002
 * @tc.desc: Verify that the DestroyOutputMemory function returns a failed message without SetInput.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_run_test_002, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_ReturnCode ret = executorTest.Run();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_run_test_003
 * @tc.desc: Verify that the DestroyOutputMemory function returns a failed message without SetOutput.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_run_test_003, testing::ext::TestSize.Level0)
{
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInput(m_index, tensor, buffer, length));
    OH_NN_ReturnCode ret = executorTest.Run();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_run_test_004
 * @tc.desc: Verify that the DestroyOutputMemory function returns a failed message with failed executionPlan.Run.
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, executor_run_test_004, testing::ext::TestSize.Level0)
{
    HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    const MSLITE::LiteGraph* liteGraphTest = BuildLiteGraph(m_dim, m_dimOut);
    InnerModel innerModel;
    innerModel.BuildFromLiteGraph(liteGraphTest);
    Compilation compilation(&innerModel);
    Executor executorTest(&compilation);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetInput(m_index, tensor, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executorTest.SetOutput(m_index, buffer, length));
    OH_NN_ReturnCode ret = executorTest.Run();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS