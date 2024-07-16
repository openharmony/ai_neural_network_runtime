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

#include "nntensor.h"
#include "nnexecutor.h"
#include "nncompiler.h"
#include "nnbackend.h"
#include "backend_manager.h"
#include "device.h"
#include "prepared_model.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"
#include "common/utils.h"
#include "common/log.h"
#include "hdi_device_v1_0.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace V1_0 = OHOS::HDI::Nnrt::V1_0;
namespace UnitTest {
class NNTensor2_0Test : public testing::Test {
public:
    NNTensor2_0Test() = default;
    ~NNTensor2_0Test() = default;
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

class MockBackend : public Backend {
public:
    MOCK_CONST_METHOD0(GetBackendID, size_t());
    MOCK_CONST_METHOD1(GetBackendName, OH_NN_ReturnCode(std::string&));
    MOCK_CONST_METHOD1(GetBackendType, OH_NN_ReturnCode(OH_NN_DeviceType&));
    MOCK_CONST_METHOD1(GetBackendStatus, OH_NN_ReturnCode(DeviceStatus&));
    MOCK_METHOD1(CreateCompiler, Compiler*(Compilation*));
    MOCK_METHOD1(DestroyCompiler, OH_NN_ReturnCode(Compiler*));
    MOCK_METHOD1(CreateExecutor, Executor*(Compilation*));
    MOCK_METHOD1(DestroyExecutor, OH_NN_ReturnCode(Executor*));
    MOCK_METHOD1(CreateTensor, Tensor*(TensorDesc*));
    MOCK_METHOD1(DestroyTensor, OH_NN_ReturnCode(Tensor*));

    std::shared_ptr<Device> GetDevice() {
        std::shared_ptr<Device> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));
        return device;
    }
    // MOCK_CONST_METHOD0(GetDevice, std::shared_ptr<Device>());
    MOCK_METHOD2(GetSupportedOperation, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
                                           std::vector<bool>&));
};

/**
 * @tc.name: nntensor2_0test_construct_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_construct_001, TestSize.Level0)
{
    LOGE("NNTensor2_0 nntensor2_0test_construct_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    delete nnTensor;
}

/**
 * @tc.name: nntensor2_0test_construct_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_construct_002, TestSize.Level0)
{
    LOGE("NNTensor2_0 nntensor2_0test_construct_002");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    nnTensor->SetSize(1);
    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* buffer = m_dataArry;
    nnTensor->SetData(buffer);
    nnTensor->SetFd(-1);
    delete nnTensor;
}

/**
 * @tc.name: nntensor2_0test_construct_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_construct_003, TestSize.Level0)
{
    LOGE("NNTensor2_0 nntensor2_0test_construct_003");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    nnTensor->SetSize(1);
    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* buffer = m_dataArry;
    nnTensor->SetData(buffer);
    nnTensor->SetFd(0);
    delete nnTensor;
}

/**
 * @tc.name: nntensor2_0test_settensordesc_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_settensordesc_001, TestSize.Level0)
{
    LOGE("SetTensorDesc nntensor2_0test_settensordesc_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc desc;
    TensorDesc* tensorDesc = &desc;
    OH_NN_ReturnCode setTensorDescRet = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, setTensorDescRet);

    OH_NN_ReturnCode ret = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_001, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    OH_NN_ReturnCode ret = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_002, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_002");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    std::shared_ptr<MockTensorDesc> tensorDesc = std::make_shared<MockTensorDesc>();
    EXPECT_CALL(*((MockTensorDesc *) tensorDesc.get()), GetByteSize(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_INVALID_PARAMETER));

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc.get());
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    OH_NN_ReturnCode retCreateData = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, retCreateData);

    testing::Mock::AllowLeak(tensorDesc.get());
}

std::shared_ptr<Backend> Creator() {
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceStatus(::testing::_))
        .WillOnce(Invoke([](DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = AVAILABLE;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    
    std::string backendName = "mock";
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVersion(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));

    std::shared_ptr<Backend> backend = std::make_unique<NNBackend>(device, backendID);
    return backend;
}

/**
 * @tc.name: nntensor2_0test_createdata_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_003, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_003");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    BackendManager& backendManager = BackendManager::GetInstance();

    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator;
    
    backendManager.RegisterBackend(backendName, creator);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    OH_NN_ReturnCode ret = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_004, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_004");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    std::shared_ptr<MockTensorDesc> tensorDesc = std::make_shared<MockTensorDesc>();
    EXPECT_CALL(*((MockTensorDesc *) tensorDesc.get()), GetByteSize(::testing::_))
        .WillRepeatedly(Invoke([](size_t* byteSize) {
                // 这里直接修改传入的引用参数
                *byteSize = ALLOCATE_BUFFER_LIMIT + 1;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc.get());
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    OH_NN_ReturnCode ret = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(tensorDesc.get());
}

/**
 * @tc.name: nntensor2_0test_createdata_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_005, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_005");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    std::shared_ptr<MockTensorDesc> tensorDesc = std::make_shared<MockTensorDesc>();
    EXPECT_CALL(*((MockTensorDesc *) tensorDesc.get()), GetByteSize(::testing::_))
        .WillRepeatedly(Invoke([](size_t* byteSize) {
                // 这里直接修改传入的引用参数
                *byteSize = 1;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc.get());
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    OH_NN_ReturnCode ret = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(tensorDesc.get());
}

std::shared_ptr<Backend> Creator2() {
    size_t backendID = 2;

    std::shared_ptr<Backend> backend = std::make_unique<NNBackend>(nullptr, backendID);
    return backend;
}

/**
 * @tc.name: nntensor2_0test_createdata_006
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_006, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_006");
    size_t backendId = 2;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    BackendManager& backendManager = BackendManager::GetInstance();

    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator2;
    
    backendManager.RegisterBackend(backendName, creator);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    OH_NN_ReturnCode ret = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
}

std::shared_ptr<Backend> Creator3() {
    size_t backendID = 3;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(Invoke([](DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = AVAILABLE;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    
    std::string backendName = "mock";
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVersion(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_MEMORY_ERROR));

    std::shared_ptr<Backend> backend = std::make_unique<NNBackend>(device, backendID);

    // LOGE("CreateData Creator [%{public}zu]",backend->GetBackendID());
    return backend;
}

/**
 * @tc.name: nntensor2_0test_createdata_007
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_007, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_007");
    size_t backendId = 3;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    BackendManager& backendManager = BackendManager::GetInstance();

    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator3;
    
    backendManager.RegisterBackend(backendName, creator);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    OH_NN_ReturnCode ret = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}

std::shared_ptr<Backend> Creator4() {
    size_t backendID = 4;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(Invoke([](DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = AVAILABLE;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    
    std::string backendName = "mock";
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVersion(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(Invoke([](size_t length, int& fd) {
                // 这里直接修改传入的引用参数
                fd = -1;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    std::shared_ptr<Backend> backend = std::make_unique<NNBackend>(device, backendID);

    // LOGE("CreateData Creator [%{public}zu]",backend->GetBackendID());
    return backend;
}

/**
 * @tc.name: nntensor2_0test_createdata_008
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_008, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_008");
    size_t backendId = 4;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    BackendManager& backendManager = BackendManager::GetInstance();

    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator4;
    
    backendManager.RegisterBackend(backendName, creator);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    OH_NN_ReturnCode ret = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_009
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_009, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_009");
    size_t backendId = 4;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* buffer = m_dataArry;
    nnTensor->SetData(buffer);

    OH_NN_ReturnCode ret = nnTensor->CreateData();
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_020
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_020, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_020");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    size_t size = 1;
    OH_NN_ReturnCode ret = nnTensor->CreateData(size);
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_021
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_021, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_021");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    size_t size = ALLOCATE_BUFFER_LIMIT + 1;
    OH_NN_ReturnCode ret = nnTensor->CreateData(size);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_022
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_022, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_022");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    size_t size = 1;
    OH_NN_ReturnCode ret = nnTensor->CreateData(size);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_023
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_023, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_023");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    std::shared_ptr<MockTensorDesc> tensorDesc = std::make_shared<MockTensorDesc>();
    EXPECT_CALL(*((MockTensorDesc *) tensorDesc.get()), GetByteSize(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_INVALID_PARAMETER));

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc.get());
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    size_t size = 1;
    OH_NN_ReturnCode ret = nnTensor->CreateData(size);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(tensorDesc.get());
}

/**
 * @tc.name: nntensor2_0test_createdata_024
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_024, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_024");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* buffer = m_dataArry;
    nnTensor->SetData(buffer);

    size_t size = 1;
    OH_NN_ReturnCode ret = nnTensor->CreateData(size);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_029
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_029, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_029");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* buffer = m_dataArry;
    nnTensor->SetData(buffer);

    int fd = 1;
    size_t size = 2;
    size_t offset = 3;
    OH_NN_ReturnCode ret = nnTensor->CreateData(fd, size, offset);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_030
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_030, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_030");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    int fd = 1;
    size_t size = 2;
    size_t offset = 3;
    OH_NN_ReturnCode ret = nnTensor->CreateData(fd, size, offset);
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_031
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_031, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_031");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    std::shared_ptr<MockTensorDesc> tensorDesc = std::make_shared<MockTensorDesc>();
    EXPECT_CALL(*((MockTensorDesc *) tensorDesc.get()), GetByteSize(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_INVALID_PARAMETER));

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc.get());
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    int fd = 1;
    size_t size = 2;
    size_t offset = 3;
    OH_NN_ReturnCode ret = nnTensor->CreateData(fd, size, offset);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(tensorDesc.get());
}

/**
 * @tc.name: nntensor2_0test_createdata_032
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_032, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_032");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    int fd = -1;
    size_t size = 2;
    size_t offset = 3;
    OH_NN_ReturnCode ret = nnTensor->CreateData(fd, size, offset);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_033
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_033, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_033");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    int fd = 0;
    size_t size = 0;
    size_t offset = 3;
    OH_NN_ReturnCode ret = nnTensor->CreateData(fd, size, offset);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_034
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_034, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_034");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    int fd = 0;
    size_t size = 1;
    size_t offset = 3;
    OH_NN_ReturnCode ret = nnTensor->CreateData(fd, size, offset);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_035
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_035, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_035");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    int fd = 0;
    size_t size = 3;
    size_t offset = 2;
    OH_NN_ReturnCode ret = nnTensor->CreateData(fd, size, offset);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_createdata_036
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_createdata_036, TestSize.Level0)
{
    LOGE("CreateData nntensor2_0test_createdata_036");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    int fd = 0;
    size_t size = 200;
    size_t offset = 1;
    OH_NN_ReturnCode ret = nnTensor->CreateData(fd, size, offset);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}


/**
 * @tc.name: nntensor2_0test_gettensordesc_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_gettensordesc_001, TestSize.Level0)
{
    LOGE("GetTensorDesc nntensor2_0test_gettensordesc_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc* ret = nnTensor->GetTensorDesc();
    EXPECT_EQ(nullptr, ret);
}

/**
 * @tc.name: nntensor2_0test_getdata_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_getdata_001, TestSize.Level0)
{
    LOGE("GetData nntensor2_0test_getdata_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    void* ret = nnTensor->GetData();
    EXPECT_EQ(nullptr, ret);
}

/**
 * @tc.name: nntensor2_0test_getfd_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_getfd_001, TestSize.Level0)
{
    LOGE("GetFd nntensor2_0test_getfd_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    int ret = nnTensor->GetFd();
    EXPECT_EQ(0, ret);
}

/**
 * @tc.name: nntensor2_0test_getsize_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_getsize_001, TestSize.Level0)
{
    LOGE("GetSize nntensor2_0test_getsize_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    size_t ret = nnTensor->GetSize();
    EXPECT_EQ(0, ret);
}

/**
 * @tc.name: nntensor2_0test_getoffset_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_getoffset_001, TestSize.Level0)
{
    LOGE("GetOffset nntensor2_0test_getoffset_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    size_t ret = nnTensor->GetOffset();
    EXPECT_EQ(0, ret);
}

/**
 * @tc.name: nntensor2_0test_getbackendid_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_getbackendid_001, TestSize.Level0)
{
    LOGE("GetBackendID nntensor2_0test_getbackendid_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    size_t ret = nnTensor->GetBackendID();
    EXPECT_EQ(1, ret);
}

/**
 * @tc.name: nntensor2_0test_checktensordata_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checktensordata_001, TestSize.Level0)
{
    LOGE("CheckTensorData nntensor2_0test_checktensordata_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    bool ret = nnTensor->CheckTensorData();
    EXPECT_EQ(false, ret);
}

/**
 * @tc.name: nntensor2_0test_checktensordata_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checktensordata_002, TestSize.Level0)
{
    LOGE("CheckTensorData nntensor2_0test_checktensordata_002");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    std::shared_ptr<MockTensorDesc> tensorDesc = std::make_shared<MockTensorDesc>();
    EXPECT_CALL(*((MockTensorDesc *) tensorDesc.get()), GetByteSize(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_INVALID_PARAMETER));

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc.get());
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    bool ret = nnTensor->CheckTensorData();
    EXPECT_EQ(false, ret);

    testing::Mock::AllowLeak(tensorDesc.get());
}

/**
 * @tc.name: nntensor2_0test_checktensordata_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checktensordata_003, TestSize.Level0)
{
    LOGE("CheckTensorData nntensor2_0test_checktensordata_003");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    bool ret = nnTensor->CheckTensorData();
    EXPECT_EQ(false, ret);
}

/**
 * @tc.name: nntensor2_0test_checktensordata_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checktensordata_004, TestSize.Level0)
{
    LOGE("CheckTensorData nntensor2_0test_checktensordata_004");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    nnTensor->SetSize(200);
    nnTensor->SetOffset(0);

    bool ret = nnTensor->CheckTensorData();
    EXPECT_EQ(false, ret);
}

/**
 * @tc.name: nntensor2_0test_checktensordata_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checktensordata_005, TestSize.Level0)
{
    LOGE("CheckTensorData nntensor2_0test_checktensordata_005");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    nnTensor->SetSize(200);
    nnTensor->SetOffset(0);
    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* buffer = m_dataArry;
    nnTensor->SetData(buffer);
    nnTensor->SetFd(-1);

    bool ret = nnTensor->CheckTensorData();
    EXPECT_EQ(false, ret);
}

/**
 * @tc.name: nntensor2_0test_checktensordata_006
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checktensordata_006, TestSize.Level0)
{
    LOGE("CheckTensorData nntensor2_0test_checktensordata_006");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    nnTensor->SetSize(200);
    nnTensor->SetOffset(0);
    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* buffer = m_dataArry;
    nnTensor->SetData(buffer);

    bool ret = nnTensor->CheckTensorData();
    EXPECT_EQ(true, ret);
}

/**
 * @tc.name: nntensor2_0test_checkdimranges_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checkdimranges_001, TestSize.Level0)
{
    LOGE("CheckDimRanges nntensor2_0test_checkdimranges_001");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    std::vector<uint32_t> minDimRanges;
    const std::vector<uint32_t> maxDimRanges;
    OH_NN_ReturnCode ret = nnTensor->CheckDimRanges(minDimRanges, maxDimRanges);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_checkdimranges_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checkdimranges_002, TestSize.Level0)
{
    LOGE("CheckDimRanges nntensor2_0test_checkdimranges_002");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    std::vector<uint32_t> minDimRanges;
    const std::vector<uint32_t> maxDimRanges;
    OH_NN_ReturnCode ret = nnTensor->CheckDimRanges(minDimRanges, maxDimRanges);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: nntensor2_0test_checkdimranges_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checkdimranges_003, TestSize.Level0)
{
    LOGE("CheckDimRanges nntensor2_0test_checkdimranges_003");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = -10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    std::vector<uint32_t> minDimRanges;
    const std::vector<uint32_t> maxDimRanges;
    OH_NN_ReturnCode ret = nnTensor->CheckDimRanges(minDimRanges, maxDimRanges);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nntensor2_0test_checkdimranges_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNTensor2_0Test, nntensor2_0test_checkdimranges_004, TestSize.Level0)
{
    LOGE("CheckDimRanges nntensor2_0test_checkdimranges_004");
    size_t backendId = 1;
    
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);
    
    TensorDesc desc;
    desc.SetDataType(OH_NN_INT64);
    size_t shapeNum = 1;
    int32_t index = 10;
    int32_t* shape = &index;
    desc.SetShape(shape, shapeNum);
    TensorDesc* tensorDesc = &desc;

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    std::vector<uint32_t> minDimRanges;
    minDimRanges.emplace_back(20);
    std::vector<uint32_t> maxDimRanges;
    maxDimRanges.emplace_back(20);
    OH_NN_ReturnCode ret = nnTensor->CheckDimRanges(minDimRanges, maxDimRanges);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS