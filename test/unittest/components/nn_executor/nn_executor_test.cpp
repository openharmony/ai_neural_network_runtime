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

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "nnexecutor.h"
#include "nncompiler.h"
#include "nnbackend.h"
#include "device.h"
#include "prepared_model.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"
#include "common/utils.h"
#include "common/log.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class NNExecutorTest : public testing::Test {
public:
    NNExecutorTest() = default;
    ~NNExecutorTest() = default;

public:
    uint32_t m_index {0};
    const std::vector<int32_t> m_dim {3, 3};
    const std::vector<int32_t> m_dimOut {3, 3};
    const int32_t m_dimArry[2] {3, 3};
    uint32_t m_dimensionCount {2};
    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
};

class MockIDevice : public Device {
public:
    MOCK_METHOD1(GetDeviceName, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetVendorName, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetVersion, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetDeviceType, OH_NN_ReturnCode(OH_NN_DeviceType&));
    MOCK_METHOD1(GetDeviceStatus, OH_NN_ReturnCode(DeviceStatus&));
    MOCK_METHOD2(GetSupportedOperation, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
        std::vector<bool>&));
    MOCK_METHOD1(IsFloat16PrecisionSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsPerformanceModeSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsPrioritySupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsDynamicInputSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsModelCacheSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD3(PrepareModel, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
                                          const ModelConfig&,
                                          std::shared_ptr<PreparedModel>&));
    MOCK_METHOD3(PrepareModel, OH_NN_ReturnCode(const void*,
                                          const ModelConfig&,
                                          std::shared_ptr<PreparedModel>&));
    MOCK_METHOD4(PrepareModelFromModelCache, OH_NN_ReturnCode(const std::vector<Buffer>&,
                                                              const ModelConfig&,
                                                              std::shared_ptr<PreparedModel>&,
                                                              bool&));
    MOCK_METHOD3(PrepareOfflineModel, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
                                                 const ModelConfig&,
                                                 std::shared_ptr<PreparedModel>&));
    MOCK_METHOD1(AllocateBuffer, void*(size_t));
    MOCK_METHOD2(AllocateTensorBuffer, void*(size_t, std::shared_ptr<TensorDesc>));
    MOCK_METHOD2(AllocateTensorBuffer, void*(size_t, std::shared_ptr<NNTensor>));
    MOCK_METHOD1(ReleaseBuffer, OH_NN_ReturnCode(const void*));
    MOCK_METHOD2(AllocateBuffer, OH_NN_ReturnCode(size_t, int&));
    MOCK_METHOD2(ReleaseBuffer, OH_NN_ReturnCode(int, size_t));
};

class MockIPreparedModel : public PreparedModel {
public:
    MOCK_METHOD1(ExportModelCache, OH_NN_ReturnCode(std::vector<Buffer>&));
    MOCK_METHOD4(Run, OH_NN_ReturnCode(const std::vector<IOTensor>&,
                                 const std::vector<IOTensor>&,
                                 std::vector<std::vector<int32_t>>&,
                                 std::vector<bool>&));
    MOCK_METHOD4(Run, OH_NN_ReturnCode(const std::vector<NN_Tensor*>&,
                                 const std::vector<NN_Tensor*>&,
                                 std::vector<std::vector<int32_t>>&,
                                 std::vector<bool>&));
    MOCK_CONST_METHOD1(GetModelID, OH_NN_ReturnCode(uint32_t&));
    MOCK_METHOD2(GetInputDimRanges, OH_NN_ReturnCode(std::vector<std::vector<uint32_t>>&,
                                               std::vector<std::vector<uint32_t>>&));
    MOCK_METHOD0(ReleaseBuiltModel, OH_NN_ReturnCode());
};

class MockTensorDesc : public TensorDesc {
public:
    MOCK_METHOD1(GetDataType, OH_NN_ReturnCode(OH_NN_DataType*));
    MOCK_METHOD1(SetDataType, OH_NN_ReturnCode(OH_NN_DataType));
    MOCK_METHOD1(GetFormat, OH_NN_ReturnCode(OH_NN_Format*));
    MOCK_METHOD1(SetFormat, OH_NN_ReturnCode(OH_NN_Format));
    MOCK_METHOD2(GetShape, OH_NN_ReturnCode(int32_t**, size_t*));
    MOCK_METHOD2(SetShape, OH_NN_ReturnCode(const int32_t*, size_t));
    MOCK_METHOD1(GetElementNum, OH_NN_ReturnCode(size_t*));
    MOCK_METHOD1(GetByteSize, OH_NN_ReturnCode(size_t*));
    MOCK_METHOD1(SetName, OH_NN_ReturnCode(const char*));
    MOCK_METHOD1(GetName, OH_NN_ReturnCode(const char**));
};

OH_NN_Tensor SetTensor(OH_NN_DataType dataType, uint32_t dimensionCount, const int32_t *dimensions,
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

/**
 * @tc.name: nnexecutortest_construct_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_construct_001, TestSize.Level0)
{
    LOGE("NNExecutor nnexecutortest_construct_001");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_outputTensorDescs[m_index].first))
        .WillRepeatedly(::testing::Return(reinterpret_cast<void*>(0x1000)));

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);
    EXPECT_NE(nullptr, nnExecutor);

    OH_NN_Memory** memory = nullptr;
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* mPtr = &memoryPtr;
    memory = &mPtr;

    OH_NN_ReturnCode retOutput = nnExecutor->CreateOutputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_SUCCESS, retOutput);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_inputTensorDescs[m_index].first))
        .WillRepeatedly(::testing::Return(reinterpret_cast<void*>(0x1000)));
    OH_NN_ReturnCode retinput = nnExecutor->CreateInputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_SUCCESS, retinput);

    delete nnExecutor;

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nnexecutortest_getinputdimrange_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputdimrange_001, TestSize.Level0)
{
    LOGE("GetInputDimRange nnexecutortest_getinputdimrange_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};

    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    size_t min = 1;
    size_t max = 10;
    size_t *minInputDims = &min;
    size_t *maxInputDIms = &max;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = nnExecutor->GetInputDimRange(index, &minInputDims, &maxInputDIms, &shapeLength);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getinputdimrange_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputdimrange_002, TestSize.Level0)
{
    LOGE("GetInputDimRange nnexecutortest_getinputdimrange_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    size_t max = 10;
    size_t *maxInputDIms = &max;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = nnExecutor->GetInputDimRange(index, nullptr, &maxInputDIms, &shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getinputdimrange_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputdimrange_003, TestSize.Level0)
{
    LOGE("GetInputDimRange nnexecutortest_getinputdimrange_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    size_t min = 1;
    size_t *minInputDims = &min;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = nnExecutor->GetInputDimRange(index, &minInputDims, nullptr, &shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getinputdimrange_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputdimrange_004, TestSize.Level0)
{
    LOGE("GetInputDimRange nnexecutortest_getinputdimrange_004");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    size_t min = 1;
    size_t max = 10;
    size_t *minInputDims = &min;
    size_t *maxInputDIms = &max;
    OH_NN_ReturnCode ret = nnExecutor->GetInputDimRange(index, &minInputDims, &maxInputDIms, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getinputdimrange_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputdimrange_005, TestSize.Level0)
{
    LOGE("GetInputDimRange nnexecutortest_getinputdimrange_005");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    size_t min = 1;
    size_t max = 10;
    size_t *minInputDims = &min;
    size_t *maxInputDIms = &max;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = nnExecutor->GetInputDimRange(index, &minInputDims, &maxInputDIms, &shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getinputdimrange_006
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputdimrange_006, TestSize.Level0)
{
    LOGE("GetInputDimRange nnexecutortest_getinputdimrange_006");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();

    std::vector<std::vector<uint32_t>> minDims = {{1, 2, 3}};
    std::vector<std::vector<uint32_t>> maxDims = {{4, 5, 6}};
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillOnce(Invoke([&minDims, &maxDims](std::vector<std::vector<uint32_t>>& minInputDims,
            std::vector<std::vector<uint32_t>>& maxInputDims) {
                // 这里直接修改传入的引用参数
                minInputDims = minDims;
                maxInputDims = maxDims;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    size_t min = 1;
    size_t max = 10;
    size_t *minInputDims = &min;
    size_t *maxInputDIms = &max;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = nnExecutor->GetInputDimRange(index, &minInputDims, &maxInputDIms, &shapeLength);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getinputdimrange_007
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputdimrange_007, TestSize.Level0)
{
    LOGE("GetInputDimRange nnexecutortest_getinputdimrange_007");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();

    std::vector<std::vector<uint32_t>> minDims = {{1, 2}, {1, 2, 3}};
    std::vector<std::vector<uint32_t>> maxDims = {{4, 5, 6}};
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillOnce(Invoke([&minDims, &maxDims](std::vector<std::vector<uint32_t>>& minInputDims,
            std::vector<std::vector<uint32_t>>& maxInputDims) {
                // 这里直接修改传入的引用参数
                minInputDims = minDims;
                maxInputDims = maxDims;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    size_t min = 1;
    size_t max = 10;
    size_t *minInputDims = &min;
    size_t *maxInputDIms = &max;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = nnExecutor->GetInputDimRange(index, &minInputDims, &maxInputDIms, &shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getinputdimrange_008
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputdimrange_008, TestSize.Level0)
{
    LOGE("GetInputDimRange nnexecutortest_getinputdimrange_008");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();

    std::vector<std::vector<uint32_t>> minDims = {{1, 2}};
    std::vector<std::vector<uint32_t>> maxDims = {{4, 5, 6}};
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillOnce(Invoke([&minDims, &maxDims](std::vector<std::vector<uint32_t>>& minInputDims,
            std::vector<std::vector<uint32_t>>& maxInputDims) {
                // 这里直接修改传入的引用参数
                minInputDims = minDims;
                maxInputDims = maxDims;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    size_t min = 1;
    size_t max = 10;
    size_t *minInputDims = &min;
    size_t *maxInputDIms = &max;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = nnExecutor->GetInputDimRange(index, &minInputDims, &maxInputDIms, &shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getoutputshape_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getoutputshape_001, TestSize.Level0)
{
    LOGE("GetOutputShape nnexecutortest_getoutputshape_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    int32_t** dimensions = &ptr;
    uint32_t dimensionCount = 2;
    uint32_t* shapeNum = &dimensionCount;
    OH_NN_ReturnCode ret = nnExecutor->GetOutputShape(m_index, dimensions, shapeNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_getoutputshape_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getoutputshape_002, TestSize.Level0)
{
    LOGE("GetOutputShape nnexecutortest_getoutputshape_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    int32_t** dimensions = &ptr;
    uint32_t dimensionCount = 2;
    uint32_t* shapeNum = &dimensionCount;
    OH_NN_ReturnCode ret = nnExecutor->GetOutputShape(m_index, dimensions, shapeNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_getoutputshape_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getoutputshape_003, TestSize.Level0)
{
    LOGE("GetOutputShape nnexecutortest_getoutputshape_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    pair1.first = tensorDesr;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    int32_t** dimensions = &ptr;
    uint32_t dimensionCount = 2;
    uint32_t* shapeNum = &dimensionCount;
    OH_NN_ReturnCode ret = nnExecutor->GetOutputShape(m_index, dimensions, shapeNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_getoutputshape_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getoutputshape_004, TestSize.Level0)
{
    LOGE("GetOutputShape nnexecutortest_getoutputshape_004");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();

    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    int32_t expectDim2[2] = {3, 3};
    int32_t* ptr2 = expectDim2;
    int32_t** dimensions = &ptr2;
    uint32_t* shapeNum = &dimensionCount;
    *dimensions = nullptr;
    OH_NN_ReturnCode ret = nnExecutor->GetOutputShape(m_index, dimensions, shapeNum);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: nnexecutortest_getinputnum_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getinputnum_001, TestSize.Level0)
{
    LOGE("GetInputNum nnexecutortest_getinputnum_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t ret = nnExecutor->GetInputNum();
    EXPECT_EQ(0, ret);
}

/**
 * @tc.name: nnexecutortest_getoutputnum_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getoutputnum_001, TestSize.Level0)
{
    LOGE("GetOutputNum nnexecutortest_getoutputnum_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);


    size_t ret = nnExecutor->GetOutputNum();
    EXPECT_EQ(0, ret);
}

/**
 * @tc.name: nnexecutortest_createinputtensordesc_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createinputtensordesc_001, TestSize.Level0)
{
    LOGE("CreateInputTensorDesc nnexecutortest_createinputtensordesc_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 1;
    NN_TensorDesc* ret = nnExecutor->CreateInputTensorDesc(index);
    EXPECT_EQ(nullptr, ret);
}

/**
 * @tc.name: nnexecutortest_createinputtensordesc_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createinputtensordesc_002, TestSize.Level0)
{
    LOGE("CreateInputTensorDesc nnexecutortest_createinputtensordesc_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 1;
    NN_TensorDesc* ret = nnExecutor->CreateInputTensorDesc(index);
    EXPECT_EQ(nullptr, ret);
}

/**
 * @tc.name: nnexecutortest_createinputtensordesc_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createinputtensordesc_003, TestSize.Level0)
{
    LOGE("CreateInputTensorDesc nnexecutortest_createinputtensordesc_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 0;
    NN_TensorDesc* ret = nnExecutor->CreateInputTensorDesc(index);
    EXPECT_NE(nullptr, ret);
}

/**
 * @tc.name: nnexecutortest_createoutputtensordesc_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createoutputtensordesc_001, TestSize.Level0)
{
    LOGE("CreateOutputTensorDesc nnexecutortest_createoutputtensordesc_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 1;
    NN_TensorDesc* ret = nnExecutor->CreateOutputTensorDesc(index);
    EXPECT_EQ(nullptr, ret);
}

/**
 * @tc.name: nnexecutortest_createoutputtensordesc_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createoutputtensordesc_002, TestSize.Level0)
{
    LOGE("CreateOutputTensorDesc nnexecutortest_createoutputtensordesc_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 1;
    NN_TensorDesc* ret = nnExecutor->CreateOutputTensorDesc(index);
    EXPECT_EQ(nullptr, ret);
}

/**
 * @tc.name: nnexecutortest_createoutputtensordesc_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createoutputtensordesc_003, TestSize.Level0)
{
    LOGE("CreateOutputTensorDesc nnexecutortest_createoutputtensordesc_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t index = 1;
    NN_TensorDesc* ret = nnExecutor->CreateOutputTensorDesc(index);
    EXPECT_NE(nullptr, ret);
}

void MyOnRunDone(void *userData, OH_NN_ReturnCode errCode, void *outputTensor[], int32_t outputCount)
{
    LOGE("MyOnRunDone");
    // 在这里处理你的逻辑，例如：
    if (errCode != OH_NN_SUCCESS) {
        // 处理错误
        LOGE("Neural network execution failed with error code: %d", errCode);
    } else {
        // 使用 outputTensor[] 和 outputCount 处理成功的结果
        // 例如，outputTensor 可能指向了神经网络输出数据的内存位置
    }
    // 如果 userData 指向了需要清理的资源，在这里进行清理
}

/**
 * @tc.name: nnexecutortest_setonrundone_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setonrundone_001, TestSize.Level0)
{
    LOGE("SetOnRunDone nnexecutortest_setonrundone_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_ReturnCode ret = nnExecutor->SetOnRunDone(MyOnRunDone);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

void MyOnServiceDied(void *userData)
{
    LOGE("MyOnServiceDied");
}

/**
 * @tc.name: nnexecutortest_setonservicedied_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setonservicedied_001, TestSize.Level0)
{
    LOGE("SetOnServiceDied nnexecutortest_setonservicedied_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_ReturnCode ret = nnExecutor->SetOnServiceDied(MyOnServiceDied);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: nnexecutortest_runsync_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_runsync_001, TestSize.Level0)
{
    LOGE("RunSync nnexecutortest_runsync_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t inputSize = 1;
    size_t outputSize = 1;
    OH_NN_ReturnCode ret = nnExecutor->RunSync(nullptr, inputSize, nullptr, outputSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_runsync_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_runsync_002, TestSize.Level0)
{
    LOGE("RunAsync nnexecutortest_runsync_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t inputSize = 0;
    size_t outputSize = 1;
    OH_NN_ReturnCode ret = nnExecutor->RunSync(nullptr, inputSize, nullptr, outputSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_runsync_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_runsync_003, TestSize.Level0)
{
    LOGE("RunAsync nnexecutortest_runsync_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();

    std::vector<std::vector<uint32_t>> minDims = {{1, 2, 3}};
    std::vector<std::vector<uint32_t>> maxDims = {{4, 5, 6}};
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillOnce(Invoke([&minDims, &maxDims](std::vector<std::vector<uint32_t>>& minInputDims,
            std::vector<std::vector<uint32_t>>& maxInputDims) {
                // 这里直接修改传入的引用参数
                minInputDims = minDims;
                maxInputDims = maxDims;
                return OH_NN_OPERATION_FORBIDDEN; // 假设成功的状态码
            }));

    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    TensorDesc desc;
    TensorDesc* tensorDesc = &desc;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));

    size_t inputSize = 2;
    size_t outputSize = 2;
    OH_NN_ReturnCode ret = nnExecutor->RunSync(&tensor, inputSize, &tensor, outputSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nnexecutortest_runsync_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_runsync_004, TestSize.Level0)
{
    LOGE("RunAsync nnexecutortest_runsync_004");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();

    std::vector<std::vector<uint32_t>> minDims = {{1, 2, 3}};
    std::vector<std::vector<uint32_t>> maxDims = {{4, 5, 6}};
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillOnce(Invoke([&minDims, &maxDims](std::vector<std::vector<uint32_t>>& minInputDims,
            std::vector<std::vector<uint32_t>>& maxInputDims) {
                // 这里直接修改传入的引用参数
                minInputDims = minDims;
                maxInputDims = maxDims;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    TensorDesc desc;
    TensorDesc* tensorDesc = &desc;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));

    size_t inputSize = 2;
    size_t outputSize = 2;
    OH_NN_ReturnCode ret = nnExecutor->RunSync(&tensor, inputSize, &tensor, outputSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_runsync_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_runsync_005, TestSize.Level0)
{
    LOGE("RunAsync nnexecutortest_runsync_005");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();

    std::vector<std::vector<uint32_t>> minDims = {{1, 2, 3}, {1, 2, 3}};
    std::vector<std::vector<uint32_t>> maxDims = {{4, 5, 6}};
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillOnce(Invoke([&minDims, &maxDims](std::vector<std::vector<uint32_t>>& minInputDims,
            std::vector<std::vector<uint32_t>>& maxInputDims) {
                // 这里直接修改传入的引用参数
                minInputDims = minDims;
                maxInputDims = maxDims;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    TensorDesc desc;
    TensorDesc* tensorDesc = &desc;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));

    size_t inputSize = 2;
    size_t outputSize = 2;
    OH_NN_ReturnCode ret = nnExecutor->RunSync(&tensor, inputSize, &tensor, outputSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_runasync_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_runasync_001, TestSize.Level0)
{
    LOGE("RunAsync nnexecutortest_runasync_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    void* buffer = m_dataArry;
    size_t inputSize = 1;
    size_t outputSize = 1;
    int32_t timeout = 10;
    OH_NN_ReturnCode ret = nnExecutor->RunAsync(nullptr, inputSize, nullptr, outputSize, timeout, buffer);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: nnexecutortest_runasync_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_runasync_002, TestSize.Level0)
{
    LOGE("RunAsync nnexecutortest_runasync_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    void* buffer = m_dataArry;
    size_t inputSize = 0;
    size_t outputSize = 1;
    int32_t timeout = 10;
    OH_NN_ReturnCode ret = nnExecutor->RunAsync(nullptr, inputSize, nullptr, outputSize, timeout, buffer);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: nnexecutortest_runasync_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_runasync_003, TestSize.Level0)
{
    LOGE("RunAsync nnexecutortest_runasync_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();

    std::vector<std::vector<uint32_t>> minDims = {{1, 2, 3}};
    std::vector<std::vector<uint32_t>> maxDims = {{4, 5, 6}};
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillOnce(Invoke([&minDims, &maxDims](std::vector<std::vector<uint32_t>>& minInputDims,
            std::vector<std::vector<uint32_t>>& maxInputDims) {
                // 这里直接修改传入的引用参数
                minInputDims = minDims;
                maxInputDims = maxDims;
                return OH_NN_OPERATION_FORBIDDEN; // 假设成功的状态码
            }));

    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    TensorDesc desc;
    TensorDesc* tensorDesc = &desc;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));

    void* buffer = m_dataArry;
    size_t inputSize = 2;
    size_t outputSize = 2;
    int32_t timeout = 10;
    OH_NN_ReturnCode ret = nnExecutor->RunAsync(&tensor, inputSize, &tensor, outputSize, timeout, buffer);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_getbackendid_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_getbackendid_001, TestSize.Level0)
{
    LOGE("GetBackendID nnexecutortest_getbackendid_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t ret = nnExecutor->GetBackendID();
    EXPECT_EQ(0, ret);
}

/**
 * @tc.name: nnexecutortest_setinput_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setinput_001, TestSize.Level0)
{
    LOGE("SetInput nnexecutortest_setinput_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = nnExecutor->SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_setinput_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setinput_002, TestSize.Level0)
{
    LOGE("SetInput nnexecutortest_setinput_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_OPERATION_FORBIDDEN));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = nnExecutor->SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_setinput_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setinput_003, TestSize.Level0)
{
    LOGE("SetInput nnexecutortest_setinput_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_OPERATION_FORBIDDEN));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = nnExecutor->SetInput(m_index, tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_setinputfrommemory_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setinputfrommemory_001, TestSize.Level0)
{
    LOGE("SetInputFromMemory nnexecutortest_setinputfrommemory_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = nnExecutor->SetInputFromMemory(m_index, tensor, memory);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_setinputfrommemory_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setinputfrommemory_002, TestSize.Level0)
{
    LOGE("SetInputFromMemory nnexecutortest_setinputfrommemory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = nnExecutor->SetInputFromMemory(m_index, tensor, memory);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_setinputfrommemory_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setinputfrommemory_003, TestSize.Level0)
{
    LOGE("SetInputFromMemory nnexecutortest_setinputfrommemory_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = nnExecutor->SetInputFromMemory(m_index, tensor, memory);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_setoutput_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setoutput_001, TestSize.Level0)
{
    LOGE("SetOutput nnexecutortest_setoutput_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = nnExecutor->SetOutput(m_index, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_setoutput_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setoutput_002, TestSize.Level0)
{
    LOGE("SetOutput nnexecutortest_setoutput_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = nnExecutor->SetOutput(m_index, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_setoutput_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setoutput_003, TestSize.Level0)
{
    LOGE("SetOutput nnexecutortest_setoutput_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    void* buffer = m_dataArry;

    OH_NN_ReturnCode ret = nnExecutor->SetOutput(m_index, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_setoutputfrommemory_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setoutputfrommemory_001, TestSize.Level0)
{
    LOGE("SetOutputFromMemory nnexecutortest_setoutputfrommemory_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = nnExecutor->SetOutputFromMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_setoutputfrommemory_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setoutputfrommemory_002, TestSize.Level0)
{
    LOGE("SetOutputFromMemory nnexecutortest_setoutputfrommemory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = nnExecutor->SetOutputFromMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_setoutputfrommemory_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setoutputfrommemory_003, TestSize.Level0)
{
    LOGE("SetOutputFromMemory nnexecutortest_setoutputfrommemory_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    void* const data = m_dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = nnExecutor->SetOutputFromMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_createinputmemory_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createinputmemory_001, TestSize.Level0)
{
    LOGE("CreateInputMemory nnexecutortest_createinputmemory_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = nnExecutor->CreateInputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_createinputmemory_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createinputmemory_002, TestSize.Level0)
{
    LOGE("CreateInputMemory nnexecutortest_createinputmemory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = nnExecutor->CreateInputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_createinputmemory_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createinputmemory_003, TestSize.Level0)
{
    LOGE("CreateInputMemory nnexecutortest_createinputmemory_003");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);

    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_inputTensorDescs[m_index].first))
        .WillRepeatedly(::testing::Return(nullptr));

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* mPtr = &memoryPtr;
    memory = &mPtr;

    OH_NN_ReturnCode ret = nnExecutor->CreateInputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nnexecutortest_createinputmemory_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createinputmemory_004, TestSize.Level0)
{
    LOGE("CreateInputMemory nnexecutortest_createinputmemory_004");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);

    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_inputTensorDescs[m_index].first))
        .WillRepeatedly(::testing::Return(reinterpret_cast<void*>(0x1000)));

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* mPtr = &memoryPtr;
    memory = &mPtr;

    OH_NN_ReturnCode ret = nnExecutor->CreateInputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nnexecutortest_destroyinputmemory_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_destroyinputmemory_001, TestSize.Level0)
{
    LOGE("DestroyInputMemory nnexecutortest_destroyinputmemory_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;

    nnExecutor->CreateInputMemory(m_index, length, memory);
    OH_NN_ReturnCode ret = nnExecutor->DestroyInputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_destroyinputmemory_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_destroyinputmemory_002, TestSize.Level0)
{
    LOGE("DestroyInputMemory nnexecutortest_destroyinputmemory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;

    nnExecutor->CreateInputMemory(m_index, length, memory);
    OH_NN_ReturnCode ret = nnExecutor->DestroyInputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_destroyinputmemory_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_destroyinputmemory_003, TestSize.Level0)
{
    LOGE("DestroyInputMemory nnexecutortest_destroyinputmemory_003");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);

    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_inputTensorDescs[m_index].first))
        .WillRepeatedly(::testing::Return(reinterpret_cast<void*>(0x1000)));
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* mPtr = &memoryPtr;
    memory = &mPtr;

    nnExecutor->CreateInputMemory(m_index, length, memory);
    OH_NN_ReturnCode ret = nnExecutor->DestroyInputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nnexecutortest_createoutputmemory_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createoutputmemory_001, TestSize.Level0)
{
    LOGE("CreateOutputMemory nnexecutortest_createoutputmemory_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;

    OH_NN_ReturnCode ret = nnExecutor->CreateOutputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_createoutputmemory_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createoutputmemory_002, TestSize.Level0)
{
    LOGE("CreateInputMemory nnexecutortest_createoutputmemory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;
    size_t length = 9 * sizeof(float);

    OH_NN_ReturnCode ret = nnExecutor->CreateOutputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_createoutputmemory_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createoutputmemory_003, TestSize.Level0)
{
    LOGE("CreateInputMemory nnexecutortest_createoutputmemory_003");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_outputTensorDescs[m_index].first))
        .WillRepeatedly(::testing::Return(nullptr));

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* mPtr = &memoryPtr;
    memory = &mPtr;

    OH_NN_ReturnCode ret = nnExecutor->CreateOutputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nnexecutortest_createoutputmemory_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_createoutputmemory_004, TestSize.Level0)
{
    LOGE("CreateInputMemory nnexecutortest_createoutputmemory_004");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_outputTensorDescs[m_index].first))
        .WillRepeatedly(::testing::Return(reinterpret_cast<void*>(0x1000)));

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* mPtr = &memoryPtr;
    memory = &mPtr;

    OH_NN_ReturnCode ret = nnExecutor->CreateOutputMemory(m_index, length, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nnexecutortest_destroyoutputmemory_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_destroyoutputmemory_001, TestSize.Level0)
{
    LOGE("DestroyOutputMemory nnexecutortest_destroyoutputmemory_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;

    nnExecutor->CreateOutputMemory(m_index, length, memory);
    OH_NN_ReturnCode ret = nnExecutor->DestroyOutputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_destroyoutputmemory_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_destroyoutputmemory_002, TestSize.Level0)
{
    LOGE("DestroyInputMemory nnexecutortest_destroyoutputmemory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    OH_NN_Memory** memory = nullptr;
    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* ptr = &memoryPtr;
    memory = &ptr;

    nnExecutor->CreateOutputMemory(m_index, length, memory);
    OH_NN_ReturnCode ret = nnExecutor->DestroyOutputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nnexecutortest_destroyoutputmemory_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_destroyoutputmemory_003, TestSize.Level0)
{
    LOGE("DestroyInputMemory nnexecutortest_destroyoutputmemory_003");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    float dataArry[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_outputTensorDescs[m_index].first))
        .WillRepeatedly(::testing::Return(reinterpret_cast<void*>(0x1000)));
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NN_Memory** memory = nullptr;
    void* const data = dataArry;
    OH_NN_Memory memoryPtr = {data, 9 * sizeof(float)};
    OH_NN_Memory* mPtr = &memoryPtr;
    memory = &mPtr;

    nnExecutor->CreateOutputMemory(m_index, length, memory);
    OH_NN_ReturnCode ret = nnExecutor->DestroyOutputMemory(m_index, memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nnexecutortest_run_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_run_001, TestSize.Level0)
{
    LOGE("Run nnexecutortest_run_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;

    nnExecutor->SetInput(m_index, tensor, buffer, length);
    nnExecutor->SetOutput(m_index, buffer, length);
    OH_NN_ReturnCode ret = nnExecutor->Run();
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_run_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_run_002, TestSize.Level0)
{
    LOGE("Run nnexecutortest_run_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;

    nnExecutor->SetInput(m_index, tensor, buffer, length);
    nnExecutor->SetOutput(m_index, buffer, length);
    OH_NN_ReturnCode ret = nnExecutor->Run();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_run_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_run_003, TestSize.Level0)
{
    LOGE("Run nnexecutortest_run_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    size_t length = 9 * sizeof(float);
    OH_NN_Tensor tensor = SetTensor(OH_NN_FLOAT32, m_dimensionCount, m_dimArry, nullptr, OH_NN_TENSOR);
    void* buffer = m_dataArry;

    nnExecutor->SetInput(m_index, tensor, buffer, length);
    nnExecutor->SetOutput(m_index, buffer, length);
    OH_NN_ReturnCode ret = nnExecutor->Run();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: nnexecutortest_setextensionconfig_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNExecutorTest, nnexecutortest_setextensionconfig_001, TestSize.Level0)
{
    LOGE("SetExtensionConfig nnexecutortest_setextensionconfig_001");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);
    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);

    std::unordered_map<std::string, std::vector<char>> configMap;
    std::string callingPidStr = "callingPid";
    std::vector<char> vecCallingPid(callingPidStr.begin(), callingPidStr.end());
    configMap["callingPid"] = vecCallingPid;

    std::string hiaiModelIdStr = "hiaiModelId";
    std::vector<char> vechiaiModelId(hiaiModelIdStr.begin(), hiaiModelIdStr.end());
    configMap["hiaiModelId"] = vechiaiModelId;

    std::string vecNeedLatencyStr = "isNeedModelLatency";
    std::vector<char> vecNeedLatency(vecNeedLatencyStr.begin(), vecNeedLatencyStr.end());
    configMap["isNeedModelLatency"] = vecNeedLatency;
    OH_NN_ReturnCode retSetExtensionConfig = nnExecutor->SetExtensionConfig(configMap);
    EXPECT_EQ(OH_NN_SUCCESS, retSetExtensionConfig);

    ExecutorConfig* retGetExecutorConfig = nnExecutor->GetExecutorConfig();
    EXPECT_NE(nullptr, retGetExecutorConfig);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS