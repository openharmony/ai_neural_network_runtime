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

#include "nnbackend.h"
#include "device.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"
#include "backend_manager.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class NNBackendTest : public testing::Test {
public:
    NNBackendTest() = default;
    ~NNBackendTest() = default;
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

/**
 * @tc.name: nnbackendtest_construct_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_construct_001, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_NE(hdiDevice, nullptr);
}

/**
 * @tc.name: nnbackendtest_getbackendname_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getbackendname_001, TestSize.Level0)
{
    size_t backendID = 1;
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    std::string backendName = "mock";
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetBackendName(backendName));
}

/**
 * @tc.name: nnbackendtest_getbackendname_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getbackendname_002, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::string backendName = "mock";
        
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceName(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_FAILED)));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetBackendName(backendName));
}

/**
 * @tc.name: nnbackendtest_getbackendname_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getbackendname_005, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::string backendName = "mock";
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceName(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVendorName(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_FAILED)));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetBackendName(backendName));
}

/**
 * @tc.name: nnbackendtest_getbackendname_007
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getbackendname_007, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::string backendName = "mock";
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceName(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVendorName(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVersion(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_FAILED)));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetBackendName(backendName));
}

/**
 * @tc.name: nnbackendtest_getbackendname_008
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getbackendname_008, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::string backendName = "mock";
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceName(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVendorName(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVersion(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_SUCCESS, hdiDevice->GetBackendName(backendName));
}

/**
 * @tc.name: nnbackendtest_getgackendtype_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getgackendtype_001, TestSize.Level0)
{
    size_t backendID = 1;

    OH_NN_DeviceType backendName = OH_NN_OTHERS;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetBackendType(backendName));
}

/**
 * @tc.name: nnbackendtest_getgackendtype_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getgackendtype_002, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    OH_NN_DeviceType backendName = OH_NN_OTHERS;
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceType(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_FAILED)));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetBackendType(backendName));
}

/**
 * @tc.name: nnbackendtest_getgackendtype_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getgackendtype_003, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    OH_NN_DeviceType backendName = OH_NN_OTHERS;
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceType(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_SUCCESS, hdiDevice->GetBackendType(backendName));
}

/**
 * @tc.name: nnbackendtest_getbackendstatus_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getbackendstatus_001, TestSize.Level0)
{
    size_t backendID = 1;

    DeviceStatus backendName = UNKNOWN;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetBackendStatus(backendName));
}

/**
 * @tc.name: nnbackendtest_getbackendstatus_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getbackendstatus_002, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    DeviceStatus backendName = UNKNOWN;
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceStatus(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_FAILED)));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetBackendStatus(backendName));
}

/**
 * @tc.name: nnbackendtest_getbackendstatus_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getbackendstatus_003, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    DeviceStatus backendName = UNKNOWN;
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceStatus(::testing::_))
    .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_SUCCESS, hdiDevice->GetBackendStatus(backendName));
}

/**
 * @tc.name: nnbackendtest_createcompiler_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_createcompiler_001, TestSize.Level0)
{
    size_t backendID = 1;

    Compilation backendName;
    Compilation* compilation = &backendName;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_NE(nullptr, hdiDevice->CreateCompiler(compilation));
}

/**
 * @tc.name: nnbackendtest_createcompiler_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_createcompiler_002, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    Compilation backendName;
    char a = 'a';
    backendName.offlineModelPath = &a;
    char b = 'b';
    backendName.offlineModelBuffer.first = &b;
    backendName.offlineModelBuffer.second = static_cast<size_t>(0);
    Compilation* compilation = &backendName;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(nullptr, hdiDevice->CreateCompiler(compilation));
}

/**
 * @tc.name: nnbackendtest_destroycompiler_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_destroycompiler_001, TestSize.Level0)
{
    size_t backendID = 1;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, hdiDevice->DestroyCompiler(nullptr));
}

/**
 * @tc.name: nnbackendtest_destroycompiler_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_destroycompiler_002, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_SUCCESS, hdiDevice->DestroyCompiler(nncompiler));
}

/**
 * @tc.name: nnbackendtest_CreateExecutor_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_CreateExecutor_001, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(nullptr, hdiDevice->CreateExecutor(nullptr));
}

/**
 * @tc.name: nnbackendtest_CreateExecutor_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_CreateExecutor_002, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    Compilation backendName;
    Compilation* compilation = &backendName;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(nullptr, hdiDevice->CreateExecutor(compilation));
}

/**
 * @tc.name: nnbackendtest_CreateExecutor_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_CreateExecutor_003, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    Compilation *compilation = new (std::nothrow) Compilation();

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(nullptr, hdiDevice->CreateExecutor(compilation));
}

/**
 * @tc.name: nnbackendtest_DestroyExecutor_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_DestroyExecutor_001, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, hdiDevice->DestroyExecutor(nullptr));
}

/**
 * @tc.name: nnbackendtest_createtensor_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_createtensor_001, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(nullptr, hdiDevice->CreateTensor(nullptr));
}

/**
 * @tc.name: nnbackendtest_createtensor_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_createtensor_002, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    TensorDesc desc;
    TensorDesc* tensorDesc = &desc;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_NE(nullptr, hdiDevice->CreateTensor(tensorDesc));
}

/**
 * @tc.name: nnbackendtest_destroytensor_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_destroytensor_001, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, hdiDevice->DestroyTensor(nullptr));
}

/**
 * @tc.name: nnbackendtest_getdevice_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getdevice_001, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_EQ(nullptr, hdiDevice->GetDevice());
}

/**
 * @tc.name: nnbackendtest_getsupportedoperation_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getsupportedoperation_001, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::shared_ptr<const mindspore::lite::LiteGraph> model = nullptr;
    std::vector<bool> ops;

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, hdiDevice->GetSupportedOperation(model, ops));
}

/**
 * @tc.name: nnbackendtest_getsupportedoperation_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getsupportedoperation_002, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::vector<bool> ops;
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetSupportedOperation(model, ops));
}

/**
 * @tc.name: nnbackendtest_getsupportedoperation_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getsupportedoperation_003, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::vector<bool> ops;
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    
    EXPECT_CALL(*((MockIDevice *) device.get()), GetSupportedOperation(::testing::_, ::testing::_))
    .WillRepeatedly(::testing::Return(OH_NN_FAILED));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetSupportedOperation(model, ops));
}

/**
 * @tc.name: nnbackendtest_getsupportedoperation_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNBackendTest, nnbackendtest_getsupportedoperation_004, TestSize.Level0)
{
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::vector<bool> ops;
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    
    EXPECT_CALL(*((MockIDevice *) device.get()), GetSupportedOperation(::testing::_, ::testing::_))
    .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));

    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(nullptr, backendID);
    EXPECT_EQ(OH_NN_FAILED, hdiDevice->GetSupportedOperation(model, ops));
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS