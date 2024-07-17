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

#include <sys/mman.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstdlib>

#include <hdf_base.h>
#include <refbase.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "hdi_device_v2_0.h"
#include "hdi_device_v2_1.h"
#include "hdi_prepared_model_v2_1.h"
#include "test/unittest/common/v2_0/mock_idevice.h"
#include "test/unittest/common/file_utils.h"

#include "lite_graph_to_hdi_model_v2_0.h"
#include "lite_graph_to_hdi_model_v2_1.h"
#include "device.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"
#include "common/log.h"
#include "nnbackend.h"
#include "ops_registry.h"
#include "transform.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;

namespace MSLITE = mindspore::lite;

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V2_1 {
class MockIDevice : public INnrtDevice {
public:
    MOCK_METHOD1(GetDeviceName, int32_t(std::string&));
    MOCK_METHOD1(GetVendorName, int32_t(std::string&));
    MOCK_METHOD1(GetDeviceType, int32_t(DeviceType&));
    MOCK_METHOD1(GetDeviceStatus, int32_t(DeviceStatus&));
    MOCK_METHOD2(GetSupportedOperation, int32_t(const Model&, std::vector<bool>&));
    MOCK_METHOD1(IsFloat16PrecisionSupported, int32_t(bool&));
    MOCK_METHOD1(IsPerformanceModeSupported, int32_t(bool&));
    MOCK_METHOD1(IsPrioritySupported, int32_t(bool&));
    MOCK_METHOD1(IsDynamicInputSupported, int32_t(bool&));
    MOCK_METHOD3(PrepareModel, int32_t(const Model&, const ModelConfig&, OHOS::sptr<IPreparedModel>&));
    MOCK_METHOD1(IsModelCacheSupported, int32_t(bool&));
    MOCK_METHOD3(PrepareModelFromModelCache, int32_t(const std::vector<SharedBuffer>&, const ModelConfig&,
        OHOS::sptr<IPreparedModel>&));
    MOCK_METHOD3(PrepareOfflineModel, int32_t(const std::vector<SharedBuffer>&, const ModelConfig&,
        sptr<OHOS::HDI::Nnrt::V2_1::IPreparedModel>&));
    MOCK_METHOD2(AllocateBuffer, int32_t(uint32_t, SharedBuffer&));
    MOCK_METHOD1(ReleaseBuffer, int32_t(const SharedBuffer&));
    MOCK_METHOD2(GetVersion, int32_t(uint32_t&, uint32_t&));
};

class MockIPreparedModel : public IPreparedModel {
public:
    MOCK_METHOD1(ExportModelCache, int32_t(std::vector<SharedBuffer>&));
    MOCK_METHOD3(Run, int32_t(const std::vector<IOTensor>&, const std::vector<IOTensor>&,
        std::vector<std::vector<int32_t>>&));
    MOCK_METHOD2(GetInputDimRanges, int32_t(std::vector<std::vector<uint32_t>>&, std::vector<std::vector<uint32_t>>&));
    MOCK_METHOD2(GetVersion, int32_t(uint32_t&, uint32_t&));

    static OH_NN_ReturnCode m_ExpectRetCode;
};


sptr<INnrtDevice> INnrtDevice::Get(bool isStub)
{
    return INnrtDevice::Get("device_service", isStub);
}

sptr<INnrtDevice> INnrtDevice::Get(const std::string& serviceName, bool isStub)
{
    if (isStub) {
        return nullptr;
    }

    sptr<INnrtDevice> mockIDevice = sptr<MockIDevice>(new (std::nothrow) MockIDevice());
    if (mockIDevice == nullptr) {
        return nullptr;
    }
    std::string deviceName = "MockDevice";
    EXPECT_CALL(*((V2_1::MockIDevice*)mockIDevice.GetRefPtr()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceName), ::testing::Return(HDF_SUCCESS)));

    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_1::MockIDevice*)mockIDevice.GetRefPtr()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(vendorName), ::testing::Return(HDF_SUCCESS)));

    V2_1::DeviceStatus deviceStatus = V2_1::DeviceStatus::AVAILABLE;
    EXPECT_CALL(*((V2_1::MockIDevice*)mockIDevice.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceStatus), ::testing::Return(HDF_SUCCESS)));

    return mockIDevice;
}
} // V2_1
} // Nnrt
} // HDI
} // OHOS

namespace mindspore {
namespace lite {
OHOS::HDI::Nnrt::V2_0::Model* MindIR_LiteGraph_To_Model(const LiteGraph* lite_graph,
    const OHOS::HDI::Nnrt::V2_0::SharedBuffer& buffer)
{
    return new (std::nothrow) OHOS::HDI::Nnrt::V2_0::Model();
}

void MindIR_Model_Destroy(OHOS::HDI::Nnrt::V2_0::Model** model)
{
    if ((model != nullptr) && (*model != nullptr)) {
        delete *model;
        *model = nullptr;
    }
}

size_t MindIR_LiteGraph_GetConstTensorSize(const mindspore::lite::LiteGraph* lite_graph)
{
    return 1;
}
}
}

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
static const int DATA_VALUE = 1;
static const int DATA_NUM = 36;
static const int DIM_NUM = 3;

void BuildLiteGraph(std::shared_ptr<mindspore::lite::LiteGraph>& model)
{
    model->name_ = "testGraph";
    model->input_indices_ = {0};
    model->output_indices_ = {1};
    model->all_tensors_ = {nullptr};
    const std::vector<mindspore::lite::QuantParam> quant_params {};
    const std::vector<uint8_t> data(DATA_NUM, DATA_VALUE);
    const std::vector<int32_t> dim = {DIM_NUM, DIM_NUM};

    for (size_t indexInput = 0; indexInput < model->input_indices_.size(); ++indexInput) {
        model->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    }

    for (size_t indexOutput = 0; indexOutput < model->output_indices_.size(); ++indexOutput) {
        model->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    }

    mindspore::lite::LiteGraph::Node node;
    node.name_ = "testNode";
    mindspore::lite::LiteGraph::Node* testNode = &node;
    model->all_nodes_.emplace_back(testNode);
    model->all_nodes_.emplace_back(testNode);
}

class HDIDeviceTest : public testing::Test {
protected:
    void GetBuffer(void*& buffer, size_t length);
    OH_NN_ReturnCode PrepareModel(int32_t allocBufferType, int32_t prepareType);
    OH_NN_ReturnCode PrepareModelv2(int32_t allocBufferType, int32_t prepareType);
};

void HDIDeviceTest::GetBuffer(void*& buffer, size_t length)
{
    std::string data = "ABCD";
    const size_t dataLength = 100;
    data.resize(dataLength, '+');

    std::string filename = "/data/log/memory-001.dat";
    FileUtils fileUtils(filename);
    fileUtils.WriteFile(data);

    int fd = open(filename.c_str(), O_RDWR);
    EXPECT_NE(fd, -1);

    const auto &memoryManager = MemoryManager::GetInstance();
    buffer = memoryManager->MapMemory(fd, length);
    EXPECT_NE(buffer, nullptr);

    const char* result = static_cast<const char*>(buffer);
    int index = 0;
    EXPECT_EQ('A', result[index++]);
    EXPECT_EQ('B', result[index++]);
    EXPECT_EQ('C', result[index++]);
    EXPECT_EQ('D', result[index++]);
    close(fd);
}

OH_NN_ReturnCode HDIDeviceTest::PrepareModel(int32_t allocBufferType, int32_t prepareType)
{
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    OHOS::sptr<V2_0::MockIDevice> sp = OHOS::sptr<V2_0::MockIDevice>(new (std::nothrow) V2_0::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    V2_0::SharedBuffer buffer {1, 1, 0, 1};
    EXPECT_CALL(*sp, AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(allocBufferType)));

    std::shared_ptr<PreparedModel> preparedModel;
    const int position = 2;
    OHOS::sptr<V2_0::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel());
    EXPECT_CALL(*sp, PrepareModel(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<position>(iPreparedModel),
        ::testing::Return(prepareType)));

    ModelConfig config;
    OH_NN_ReturnCode result = hdiDevice->PrepareModel(model, config, preparedModel);
    return result;
}

OH_NN_ReturnCode HDIDeviceTest::PrepareModelv2(int32_t allocBufferType, int32_t prepareType)
{
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    OHOS::sptr<V2_1::MockIDevice> sp = OHOS::sptr<V2_1::MockIDevice>(new (std::nothrow) V2_1::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    V2_1::SharedBuffer buffer {1, 1, 0, 1};
    EXPECT_CALL(*sp, AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(allocBufferType)));

    std::shared_ptr<PreparedModel> preparedModel;
    const int position = 2;
    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_CALL(*sp, PrepareModel(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<position>(iPreparedModel),
        ::testing::Return(prepareType)));

    ModelConfig config;
    OH_NN_ReturnCode result = hdiDevice->PrepareModel(model, config, preparedModel);
    return result;
}

class MockPreparedModel : public PreparedModel {
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
};

/* *
 * @tc.name: hdidevice_constructor_001
 * @tc.desc: Verify the Constructor function return object success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_constructor_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    EXPECT_NE(device, nullptr);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
}

/* *
 * @tc.name: hdidevice_getdevicename_001
 * @tc.desc: Verify the GetDeviceName function validate device name success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicename_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string deviceName = "MockDevice";
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceName), ::testing::Return(HDF_SUCCESS)));

    const std::string expectDeviceName = "MockDevice";
    std::string newDeviceName = "";
    OH_NN_ReturnCode result = hdiDevice->GetDeviceName(newDeviceName);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectDeviceName, newDeviceName);
}

/* *
 * @tc.name: hdidevice_getdevicename_002
 * @tc.desc: Verify the GetDeviceName function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicename_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string deviceName = "MockDevice";
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceName), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceName(deviceName);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_getvendorname_001
 * @tc.desc: Verify the GetVendorName function validate vendor name success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getvendorname_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(vendorName), ::testing::Return(HDF_SUCCESS)));

    const std::string expectDeviceName = "MockVendor";
    std::string newVendorName = "";
    OH_NN_ReturnCode result = hdiDevice->GetVendorName(newVendorName);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectDeviceName, newVendorName);
}

/* *
 * @tc.name: hdidevice_getvendorname_002
 * @tc.desc: Verify the GetVendorName function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getvendorname_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(vendorName), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->GetVendorName(vendorName);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_getversion_001
 * @tc.desc: Verify the GetVersion function validate vendor name success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getversion_001, TestSize.Level0)
{
    LOGE("GetVersion hdidevice_getversion_001");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetVersion(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(HDF_SUCCESS));

    const std::string expectDeviceName = "MockVendor";
    std::string newVendorName = "";
    OH_NN_ReturnCode result = hdiDevice->GetVersion(newVendorName);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_getversion_002
 * @tc.desc: Verify the GetVersion function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getversion_002, TestSize.Level0)
{
    LOGE("GetVersion hdidevice_getversion_002");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetVersion(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));
    OH_NN_ReturnCode result = hdiDevice->GetVersion(vendorName);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_getdevicetype_001
 * @tc.desc: Verify the GetDeviceType function validate device type success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicetype_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    V2_0::DeviceType iDeviceType = V2_0::DeviceType::CPU;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(iDeviceType), ::testing::Return(HDF_SUCCESS)));

    OH_NN_DeviceType expectDeviceType = OH_NN_CPU;
    OH_NN_DeviceType newDeviceType = OH_NN_CPU;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(newDeviceType);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectDeviceType, newDeviceType);
}

/* *
 * @tc.name: hdidevice_getdevicetype_002
 * @tc.desc: Verify the GetDeviceType function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicetype_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    OH_NN_DeviceType deviceType = OH_NN_CPU;
    V2_0::DeviceType iDeviceType = V2_0::DeviceType::CPU;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(iDeviceType), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(deviceType);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_getdevicetype_003
 * @tc.desc: Verify the GetDeviceType function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicetype_003, TestSize.Level0)
{
    LOGE("GetDeviceType hdidevice_getdevicetype_003");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    OH_NN_DeviceType deviceType = OH_NN_CPU;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(Invoke([](V2_0::DeviceType& type) {
                // 这里直接修改传入的引用参数
                type = V2_0::DeviceType::GPU;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(deviceType);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_getdevicetype_004
 * @tc.desc: Verify the GetDeviceType function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicetype_004, TestSize.Level0)
{
    LOGE("GetDeviceType hdidevice_getdevicetype_004");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    OH_NN_DeviceType deviceType = OH_NN_CPU;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(Invoke([](V2_0::DeviceType& type) {
                // 这里直接修改传入的引用参数
                type = V2_0::DeviceType::ACCELERATOR;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(deviceType);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_getdevicetype_005
 * @tc.desc: Verify the GetDeviceType function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicetype_005, TestSize.Level0)
{
    LOGE("GetDeviceType hdidevice_getdevicetype_005");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    OH_NN_DeviceType deviceType = OH_NN_CPU;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(Invoke([](V2_0::DeviceType& type) {
                // 这里直接修改传入的引用参数
                type = V2_0::DeviceType::OTHER;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(deviceType);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_getdevicestatus_001
 * @tc.desc: Verify the GetDeviceStatus function validate device status success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicestatus_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_0::DeviceStatus iDeviceStatus = V2_0::DeviceStatus::AVAILABLE;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(iDeviceStatus), ::testing::Return(HDF_SUCCESS)));

    const DeviceStatus expectDeviceStatus = AVAILABLE;
    DeviceStatus newDeviceStatus = AVAILABLE;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(newDeviceStatus);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectDeviceStatus, newDeviceStatus);
}

/* *
 * @tc.name: hdidevice_getdevicestatus_002
 * @tc.desc: Verify the GetDeviceStatus function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicestatus_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    DeviceStatus deviceStatus = AVAILABLE;
    V2_0::DeviceStatus iDeviceStatus = V2_0::DeviceStatus::AVAILABLE;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(iDeviceStatus), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(deviceStatus);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_getdevicestatus_003
 * @tc.desc: Verify the GetDeviceStatus function validate device status success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicestatus_003, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_getdevicestatus_003");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(Invoke([](V2_0::DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = V2_0::DeviceStatus::BUSY;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    DeviceStatus newDeviceStatus = AVAILABLE;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(newDeviceStatus);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_getdevicestatus_004
 * @tc.desc: Verify the GetDeviceStatus function validate device status success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicestatus_004, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_getdevicestatus_004");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(Invoke([](V2_0::DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = V2_0::DeviceStatus::OFFLINE;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    DeviceStatus newDeviceStatus = AVAILABLE;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(newDeviceStatus);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_getdevicestatus_005
 * @tc.desc: Verify the GetDeviceStatus function validate device status success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getdevicestatus_005, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_getdevicestatus_005");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(Invoke([](V2_0::DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = V2_0::DeviceStatus::UNKNOWN;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    DeviceStatus newDeviceStatus = AVAILABLE;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(newDeviceStatus);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_getsupportedoperation_001
 * @tc.desc: Verify the GetSupportedOperation function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getsupportedoperation_001, TestSize.Level0)
{
    std::vector<bool> ops {true};
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    EXPECT_NE(nullptr, model);
    BuildLiteGraph(model);

    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_0::SharedBuffer buffer {1, 1, 0, 1};
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(HDF_SUCCESS)));

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetSupportedOperation(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(ops), ::testing::Return(HDF_SUCCESS)));

    std::vector<bool> newOps {true};
    const std::vector<bool> expectOps {true};
    OH_NN_ReturnCode result = hdiDevice->GetSupportedOperation(model, newOps);
    EXPECT_EQ(OH_NN_FAILED, result);
    auto expectOpsSize = expectOps.size();
    for (size_t i = 0; i < expectOpsSize; ++i) {
        EXPECT_EQ(expectOps[i], newOps[i]);
    }
}

/* *
 * @tc.name: hdidevice_getsupportedoperation_002
 * @tc.desc: Verify the GetSupportedOperation function return failed in case of allocate buffer failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getsupportedoperation_002, TestSize.Level0)
{
    std::vector<bool> ops;
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    EXPECT_NE(nullptr, model);
    BuildLiteGraph(model);

    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_0::SharedBuffer buffer {1, 1, 0, 1};
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(HDF_FAILURE)));

    OH_NN_ReturnCode result = hdiDevice->GetSupportedOperation(model, ops);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_getsupportedoperation_003
 * @tc.desc: Verify the GetSupportedOperation function return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getsupportedoperation_003, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    std::shared_ptr<const mindspore::lite::LiteGraph> model = nullptr;
    std::vector<bool> ops;
    OH_NN_ReturnCode result = hdiDevice->GetSupportedOperation(model, ops);
    EXPECT_EQ(OH_NN_NULL_PTR, result);
}

/* *
 * @tc.name: hdidevice_getsupportedoperation_004
 * @tc.desc: Verify the GetSupportedOperation function return unavalidable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_getsupportedoperation_004, TestSize.Level0)
{
    std::vector<bool> ops {true};
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    EXPECT_NE(nullptr, model);
    BuildLiteGraph(model);

    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_0::SharedBuffer buffer {2, 1, 0, 1};
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(HDF_SUCCESS)));

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), GetSupportedOperation(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(ops), ::testing::Return(HDF_FAILURE)));

    std::vector<bool> newOps {true};
    OH_NN_ReturnCode result = hdiDevice->GetSupportedOperation(model, newOps);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_isfloat16precisionsupported_001
 * @tc.desc: Verify the IsFloat16PrecisionSupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_isfloat16precisionsupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsFloat16PrecisionSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));
    OH_NN_ReturnCode result = hdiDevice->IsFloat16PrecisionSupported(isSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_isfloat16precisionsupported_002
 * @tc.desc: Verify the IsFloat16PrecisionSupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_isfloat16precisionsupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsFloat16PrecisionSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsFloat16PrecisionSupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_isperformancemodesupported_001
 * @tc.desc: Verify the IsPerformanceModeSupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_isperformancemodesupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsPerformanceModeSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));

    bool newIsSupported = false;
    const bool expectIsSupported = false;
    OH_NN_ReturnCode result = hdiDevice->IsPerformanceModeSupported(newIsSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectIsSupported, newIsSupported);
}

/* *
 * @tc.name: hdidevice_isperformancemodesupported_002
 * @tc.desc: Verify the IsPerformanceModeSupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_isperformancemodesupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsPerformanceModeSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsPerformanceModeSupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_isprioritysupported_001
 * @tc.desc: Verify the IsPrioritySupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_isprioritysupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsPrioritySupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));

    bool newIsSupported = false;
    bool expectIsSupported = false;
    OH_NN_ReturnCode result = hdiDevice->IsPrioritySupported(newIsSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(newIsSupported, expectIsSupported);
}

/* *
 * @tc.name: hdidevice_isprioritysupported_002
 * @tc.desc: Verify the IsPrioritySupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_isprioritysupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsPrioritySupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsPrioritySupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_isdynamicinputsupported_001
 * @tc.desc: Verify the IsDynamicInputSupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_isdynamicinputsupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsDynamicInputSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));

    bool newIsSupported = false;
    bool expectIsSupported = false;
    OH_NN_ReturnCode result = hdiDevice->IsDynamicInputSupported(newIsSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(newIsSupported, expectIsSupported);
}

/* *
 * @tc.name: hdidevice_isdynamicinputsupported_002
 * @tc.desc: Verify the IsDynamicInputSupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_isdynamicinputsupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsDynamicInputSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsDynamicInputSupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_isdynamicinputsupported_001
 * @tc.desc: Verify the IsModelCacheSupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_ismodelcachesupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsModelCacheSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));

    bool newIsSupported = false;
    bool expectIsSupported = false;
    OH_NN_ReturnCode result = hdiDevice->IsModelCacheSupported(newIsSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectIsSupported, newIsSupported);
}

/* *
 * @tc.name: hdidevice_isdynamicinputsupported_002
 * @tc.desc: Verify the IsModelCacheSupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_ismodelcachesupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), IsModelCacheSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsModelCacheSupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_preparemodel_001
 * @tc.desc: Verify the PrepareModel function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodel_001, TestSize.Level0)
{
    int32_t allocBufferType = HDF_SUCCESS;
    int32_t prepareType = HDF_SUCCESS;
    OH_NN_ReturnCode result = PrepareModel(allocBufferType, prepareType);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_preparemodel_002
 * @tc.desc: Verify the PrepareModel function return invalid parameter.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodel_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    std::shared_ptr<const mindspore::lite::LiteGraph> model = nullptr;
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode result = hdiDevice->PrepareModel(model, config, preparedModel);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/* *
 * @tc.name: hdidevice_preparemodel_003
 * @tc.desc: Verify the PrepareModel function return failed.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodel_003, TestSize.Level0)
{
    int32_t allocBufferType = HDF_SUCCESS;
    int32_t prepareType = HDF_FAILURE;
    OH_NN_ReturnCode result = PrepareModel(allocBufferType, prepareType);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_preparemodel_004
 * @tc.desc: Verify the PrepareModel function return failed.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodel_004, TestSize.Level0)
{
    int32_t allocBufferType = HDF_FAILURE;
    int32_t prepareType = HDF_FAILURE;
    OH_NN_ReturnCode result = PrepareModel(allocBufferType, prepareType);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_preparemodel_005
 * @tc.desc: Verify the PrepareModel function return failed.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodel_005, TestSize.Level0)
{
    LOGE("PrepareModel hdidevice_preparemodel_005");
    OHOS::sptr<V2_0::MockIDevice> sp = OHOS::sptr<V2_0::MockIDevice>(new (std::nothrow) V2_0::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* data = dataArry;
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode result = hdiDevice->PrepareModel(data, config, preparedModel);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, result);
}

/* *
 * @tc.name: hdidevice_preparemodelfrommodelcache_001
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodelfrommodelcache_001, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;

    OHOS::sptr<V2_0::MockIDevice> sp = OHOS::sptr<V2_0::MockIDevice>(new (std::nothrow) V2_0::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::shared_ptr<PreparedModel> preparedModel;

    OHOS::sptr<V2_0::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel());
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_SUCCESS)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    const auto &memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_preparemodelfrommodelcache_002
 * @tc.desc: Verify the PrepareModelFromModelCache function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodelfrommodelcache_002, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_0::MockIDevice> sp = OHOS::sptr<V2_0::MockIDevice>(new (std::nothrow) V2_0::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    OHOS::sptr<V2_0::IPreparedModel> preModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_0>(preModel);

    OHOS::sptr<V2_0::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_preparemodelfrommodelcache_003
 * @tc.desc: Verify the PrepareModelFromModelCache function return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodelfrommodelcache_003, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { nullptr, 0 } };
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_NULL_PTR, result);
}

/* *
 * @tc.name: hdidevice_preparemodelfrommodelcache_004
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodelfrommodelcache_004, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_getdevicestatus_005");
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_0::MockIDevice> sp = OHOS::sptr<V2_0::MockIDevice>(new (std::nothrow) V2_0::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    config.mode = OH_NN_PERFORMANCE_LOW;
    config.priority = OH_NN_PRIORITY_LOW;
    OHOS::sptr<V2_0::IPreparedModel> preModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_0>(preModel);

    OHOS::sptr<V2_0::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_preparemodelfrommodelcache_005
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodelfrommodelcache_005, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_preparemodelfrommodelcache_005");
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_0::MockIDevice> sp = OHOS::sptr<V2_0::MockIDevice>(new (std::nothrow) V2_0::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    config.mode = OH_NN_PERFORMANCE_MEDIUM;
    config.priority = OH_NN_PRIORITY_MEDIUM;
    OHOS::sptr<V2_0::IPreparedModel> preModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_0>(preModel);

    OHOS::sptr<V2_0::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_preparemodelfrommodelcache_006
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodelfrommodelcache_006, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_preparemodelfrommodelcache_006");
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_0::MockIDevice> sp = OHOS::sptr<V2_0::MockIDevice>(new (std::nothrow) V2_0::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    config.mode = OH_NN_PERFORMANCE_HIGH;
    config.priority = OH_NN_PRIORITY_HIGH;
    OHOS::sptr<V2_0::IPreparedModel> preModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_0>(preModel);

    OHOS::sptr<V2_0::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_preparemodelfrommodelcache_007
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodelfrommodelcache_007, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_preparemodelfrommodelcache_007");
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_0::MockIDevice> sp = OHOS::sptr<V2_0::MockIDevice>(new (std::nothrow) V2_0::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    config.mode = OH_NN_PERFORMANCE_EXTREME;
    OHOS::sptr<V2_0::IPreparedModel> preModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_0>(preModel);

    OHOS::sptr<V2_0::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_allocatebuffer_001
 * @tc.desc: Verify the AllocateBuffer function return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_allocatebuffer_001, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_0::SharedBuffer buffer;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(HDF_FAILURE)));

    size_t length = 8;
    void *result = hdiDevice->AllocateBuffer(length);
    EXPECT_EQ(nullptr, result);
    hdiDevice->ReleaseBuffer(result);
}

/* *
 * @tc.name: hdidevice_allocatebuffer_002
 * @tc.desc: Verify the AllocateBuffer function return nullptr and HDF_FAILURE.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_allocatebuffer_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 8;
    void *result = hdiDevice->AllocateBuffer(length);
    EXPECT_EQ(nullptr, result);
    hdiDevice->ReleaseBuffer(result);
}

/* *
 * @tc.name: hdidevice_allocatebuffer_003
 * @tc.desc: Verify the AllocateBuffer function return nullptr in case of 0 size.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_allocatebuffer_003, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 0;
    void *result = hdiDevice->AllocateBuffer(length);
    EXPECT_EQ(nullptr, result);
}

/* *
 * @tc.name: hdidevice_allocatebuffer_004
 * @tc.desc: Verify the AllocateBuffer function return nullptr in case of 0 size.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_allocatebuffer_004, TestSize.Level0)
{
    LOGE("AllocateBuffer hdidevice_preparemodelfrommodelcache_007");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 0;
    int fd = 0;
    OH_NN_ReturnCode result = hdiDevice->AllocateBuffer(length, fd);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/* *
 * @tc.name: hdidevice_allocatebuffer_005
 * @tc.desc: Verify the AllocateBuffer function return nullptr in case of 0 size.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_allocatebuffer_005, TestSize.Level0)
{
    LOGE("AllocateBuffer hdidevice_allocatebuffer_005");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 1;
    int fd = 0;
    OH_NN_ReturnCode result = hdiDevice->AllocateBuffer(length, fd);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, result);
}

/* *
 * @tc.name: hdidevice_allocatebuffer_006
 * @tc.desc: Verify the AllocateBuffer function return nullptr in case of 0 size.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_allocatebuffer_006, TestSize.Level0)
{
    LOGE("AllocateBuffer hdidevice_allocatebuffer_006");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(HDF_SUCCESS));

    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 1;
    int fd = 0;
    OH_NN_ReturnCode result = hdiDevice->AllocateBuffer(length, fd);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_releasebuffer_001
 * @tc.desc: Verify the ReleaseBuffer function validate buffer success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_releasebuffer_001, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_SUCCESS));

    EXPECT_NE(hdiDevice, nullptr);
    hdiDevice->ReleaseBuffer(buffer);
    const auto &memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}

/* *
 * @tc.name: hdidevice_releasebuffer_002
 * @tc.desc: Verify the ReleaseBuffer function validate AllocateBuffer return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_releasebuffer_002, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_0::SharedBuffer sharedbuffer;
    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(sharedbuffer), ::testing::Return(HDF_FAILURE)));

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));

    size_t length = 8;
    void *buffer = hdiDevice->AllocateBuffer(length);
    hdiDevice->ReleaseBuffer(buffer);
}

/* *
 * @tc.name: hdidevice_releasebuffer_003
 * @tc.desc: Verify the ReleaseBuffer function validate param buffer is nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_releasebuffer_003, TestSize.Level0)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    void *buffer = nullptr;
    hdiDevice->ReleaseBuffer(buffer);
}

/* *
 * @tc.name: hdidevice_releasebuffer_004
 * @tc.desc: Verify the ReleaseBuffer function validate invalid buffer.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_releasebuffer_004, TestSize.Level0)
{
    const size_t length = 100;
    auto* buffer = new(std::nothrow) char[length];
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    hdiDevice->ReleaseBuffer(buffer);
    delete[] buffer;
    buffer = nullptr;
}

/* *
 * @tc.name: hdidevice_releasebuffer_005
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_releasebuffer_005, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));

    hdiDevice->ReleaseBuffer(buffer);
    const auto &memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}

/* *
 * @tc.name: hdidevice_releasebuffer_007
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_releasebuffer_007, TestSize.Level0)
{
    LOGE("ReleaseBuffer hdidevice_releasebuffer_007");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));

    int fd = 0;
    size_t length = 1;
    OH_NN_ReturnCode ret = hdiDevice->ReleaseBuffer(fd, length);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/* *
 * @tc.name: hdidevice_releasebuffer_008
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_releasebuffer_008, TestSize.Level0)
{
    LOGE("ReleaseBuffer hdidevice_releasebuffer_008");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_0::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_SUCCESS));

    int fd = 0;
    size_t length = 1;
    OH_NN_ReturnCode ret = hdiDevice->ReleaseBuffer(fd, length);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/* *
 * @tc.name: hdidevice_allocatetensorbuffer_001
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_allocatetensorbuffer_001, TestSize.Level0)
{
    LOGE("AllocateTensorBuffer hdidevice_allocatetensorbuffer_001");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 0;
    std::shared_ptr<TensorDesc> tensor;
    void* ret = hdiDevice->AllocateTensorBuffer(length, tensor);
    EXPECT_EQ(nullptr, ret);
}

/* *
 * @tc.name: hdidevice_allocatetensorbuffer_002
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_allocatetensorbuffer_002, TestSize.Level0)
{
    LOGE("AllocateTensorBuffer hdidevice_allocatetensorbuffer_002");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 0;
    std::shared_ptr<NNTensor> tensor;
    void* ret = hdiDevice->AllocateTensorBuffer(length, tensor);
    EXPECT_EQ(nullptr, ret);
}

/* *
 * @tc.name: hdidevice_prepareofflinemodel_001
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_prepareofflinemodel_001, TestSize.Level0)
{
    LOGE("PrepareOfflineModel hdidevice_prepareofflinemodel_001");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode ret = hdiDevice->PrepareOfflineModel(nullptr, config, preparedModel);
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
}

/* *
 * @tc.name: hdidevice_prepareofflinemodel_002
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_prepareofflinemodel_002, TestSize.Level0)
{
    LOGE("PrepareOfflineModel hdidevice_prepareofflinemodel_002");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    mindspore::lite::LiteGraph::Node node;
    mindspore::lite::LiteGraph::Node* testNode = &node;
    model->all_nodes_.emplace_back(testNode);
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode ret = hdiDevice->PrepareOfflineModel(model, config, preparedModel);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/* *
 * @tc.name: hdidevice_prepareofflinemodel_003
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_prepareofflinemodel_003, TestSize.Level0)
{
    LOGE("PrepareOfflineModel hdidevice_prepareofflinemodel_003");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    mindspore::lite::LiteGraph::Node node;
    uint32_t indice = 0;
    node.input_indices_.emplace_back(indice);
    node.input_indices_.emplace_back(indice);
    mindspore::lite::LiteGraph::Node* testNode = &node;
    model->all_nodes_.emplace_back(testNode);
    model->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    model->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode ret = hdiDevice->PrepareOfflineModel(model, config, preparedModel);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/* *
 * @tc.name: hdidevice_prepareofflinemodel_004
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_prepareofflinemodel_004, TestSize.Level0)
{
    LOGE("PrepareOfflineModel hdidevice_prepareofflinemodel_004");
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_0> hdiDevice = std::make_unique<HDIDeviceV2_0>(device);
    EXPECT_NE(hdiDevice, nullptr);
    
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    mindspore::lite::LiteGraph::Node node;
    uint32_t indice = 0;
    node.input_indices_.emplace_back(indice);
    node.input_indices_.emplace_back(indice);
    mindspore::lite::LiteGraph::Node* testNode = &node;
    model->all_nodes_.emplace_back(testNode);

    char a = 'a';
    mindspore::lite::DataType data_type = mindspore::lite::DataType::DATA_TYPE_INT32;
    int dim = 1;
    int32_t *dims = &dim;
    uint32_t dims_size = 1;
    mindspore::lite::Format format = mindspore::lite::Format::FORMAT_HWCK;
    uint8_t datas = 0;
    uint8_t *data = &datas;
    uint32_t data_size = 2;
    mindspore::lite::QuantParam quant_params;
    uint32_t quant_params_size = 0;
    mindspore::lite::TensorPtr ptr2 = mindspore::lite::MindIR_Tensor_Create(&a, data_type, dims, dims_size,
                               format, data, data_size,
                               &quant_params, quant_params_size);
    std::vector<uint8_t> offlineModel2 = mindspore::lite::MindIR_Tensor_GetData(ptr2);

    model->all_tensors_.emplace_back(ptr2);
    model->all_tensors_.emplace_back(ptr2);
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode ret = hdiDevice->PrepareOfflineModel(model, config, preparedModel);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);

    testing::Mock::AllowLeak(device.GetRefPtr());
}

/* *
 * @tc.name: hdidevice_V2_1_constructor_001
 * @tc.desc: Verify the Constructor function return object success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_constructor_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    EXPECT_NE(device, nullptr);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicename_001
 * @tc.desc: Verify the GetDeviceName function validate device name success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicename_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string deviceName = "MockDevice";
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceName), ::testing::Return(HDF_SUCCESS)));

    const std::string expectDeviceName = "MockDevice";
    std::string newDeviceName = "";
    OH_NN_ReturnCode result = hdiDevice->GetDeviceName(newDeviceName);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectDeviceName, newDeviceName);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicename_002
 * @tc.desc: Verify the GetDeviceName function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicename_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string deviceName = "MockDevice";
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceName), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceName(deviceName);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getvendorname_001
 * @tc.desc: Verify the GetVendorName function validate vendor name success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getvendorname_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(vendorName), ::testing::Return(HDF_SUCCESS)));

    const std::string expectDeviceName = "MockVendor";
    std::string newVendorName = "";
    OH_NN_ReturnCode result = hdiDevice->GetVendorName(newVendorName);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectDeviceName, newVendorName);
}

/* *
 * @tc.name: hdidevice_V2_1_getvendorname_002
 * @tc.desc: Verify the GetVendorName function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getvendorname_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(vendorName), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->GetVendorName(vendorName);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getversion_001
 * @tc.desc: Verify the GetVersion function validate vendor name success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getversion_001, TestSize.Level0)
{
    LOGE("GetVersion hdidevice_V2_1_getversion_001");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetVersion(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(HDF_SUCCESS));

    const std::string expectDeviceName = "MockVendor";
    std::string newVendorName = "";
    OH_NN_ReturnCode result = hdiDevice->GetVersion(newVendorName);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getversion_002
 * @tc.desc: Verify the GetVersion function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getversion_002, TestSize.Level0)
{
    LOGE("GetVersion hdidevice_V2_1_getversion_002");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    std::string vendorName = "MockVendor";
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetVersion(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));
    OH_NN_ReturnCode result = hdiDevice->GetVersion(vendorName);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicetype_001
 * @tc.desc: Verify the GetDeviceType function validate device type success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicetype_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    V2_1::DeviceType iDeviceType = V2_1::DeviceType::CPU;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(iDeviceType), ::testing::Return(HDF_SUCCESS)));

    OH_NN_DeviceType expectDeviceType = OH_NN_CPU;
    OH_NN_DeviceType newDeviceType = OH_NN_CPU;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(newDeviceType);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectDeviceType, newDeviceType);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicetype_002
 * @tc.desc: Verify the GetDeviceType function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicetype_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    OH_NN_DeviceType deviceType = OH_NN_CPU;
    V2_1::DeviceType iDeviceType = V2_1::DeviceType::CPU;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(iDeviceType), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(deviceType);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicetype_003
 * @tc.desc: Verify the GetDeviceType function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicetype_003, TestSize.Level0)
{
    LOGE("GetDeviceType hdidevice_V2_1_getdevicetype_003");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    OH_NN_DeviceType deviceType = OH_NN_CPU;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(Invoke([](V2_1::DeviceType& type) {
                // 这里直接修改传入的引用参数
                type = V2_1::DeviceType::GPU;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(deviceType);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicetype_004
 * @tc.desc: Verify the GetDeviceType function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicetype_004, TestSize.Level0)
{
    LOGE("GetDeviceType hdidevice_V2_1_getdevicetype_004");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    OH_NN_DeviceType deviceType = OH_NN_CPU;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(Invoke([](V2_1::DeviceType& type) {
                // 这里直接修改传入的引用参数
                type = V2_1::DeviceType::ACCELERATOR;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(deviceType);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicetype_005
 * @tc.desc: Verify the GetDeviceType function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicetype_005, TestSize.Level0)
{
    LOGE("GetDeviceType hdidevice_V2_1_getdevicetype_005");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    OH_NN_DeviceType deviceType = OH_NN_CPU;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceType(::testing::_))
        .WillRepeatedly(Invoke([](V2_1::DeviceType& type) {
                // 这里直接修改传入的引用参数
                type = V2_1::DeviceType::OTHER;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceType(deviceType);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicestatus_001
 * @tc.desc: Verify the GetDeviceStatus function validate device status success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicestatus_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_1::DeviceStatus iDeviceStatus = V2_1::DeviceStatus::AVAILABLE;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(iDeviceStatus), ::testing::Return(HDF_SUCCESS)));

    const DeviceStatus expectDeviceStatus = AVAILABLE;
    DeviceStatus newDeviceStatus = AVAILABLE;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(newDeviceStatus);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectDeviceStatus, newDeviceStatus);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicestatus_002
 * @tc.desc: Verify the GetDeviceStatus function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicestatus_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    DeviceStatus deviceStatus = AVAILABLE;
    V2_1::DeviceStatus iDeviceStatus = V2_1::DeviceStatus::AVAILABLE;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(iDeviceStatus), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(deviceStatus);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicestatus_003
 * @tc.desc: Verify the GetDeviceStatus function validate device status success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicestatus_003, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_V2_1_getdevicestatus_003");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(Invoke([](V2_1::DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = V2_1::DeviceStatus::BUSY;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    DeviceStatus newDeviceStatus = AVAILABLE;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(newDeviceStatus);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicestatus_004
 * @tc.desc: Verify the GetDeviceStatus function validate device status success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicestatus_004, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_V2_1_getdevicestatus_004");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(Invoke([](V2_1::DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = V2_1::DeviceStatus::OFFLINE;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    DeviceStatus newDeviceStatus = AVAILABLE;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(newDeviceStatus);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getdevicestatus_005
 * @tc.desc: Verify the GetDeviceStatus function validate device status success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getdevicestatus_005, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_V2_1_getdevicestatus_005");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(Invoke([](V2_1::DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = V2_1::DeviceStatus::UNKNOWN;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    DeviceStatus newDeviceStatus = AVAILABLE;
    OH_NN_ReturnCode result = hdiDevice->GetDeviceStatus(newDeviceStatus);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getsupportedoperation_001
 * @tc.desc: Verify the GetSupportedOperation function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getsupportedoperation_001, TestSize.Level0)
{
    std::vector<bool> ops {true};
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    EXPECT_NE(nullptr, model);
    BuildLiteGraph(model);

    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_1::SharedBuffer buffer {1, 1, 0, 1};
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(HDF_SUCCESS)));

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetSupportedOperation(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(ops), ::testing::Return(HDF_SUCCESS)));

    std::vector<bool> newOps {true};
    const std::vector<bool> expectOps {true};
    OH_NN_ReturnCode result = hdiDevice->GetSupportedOperation(model, newOps);
    EXPECT_EQ(OH_NN_FAILED, result);
    auto expectOpsSize = expectOps.size();
    for (size_t i = 0; i < expectOpsSize; ++i) {
        EXPECT_EQ(expectOps[i], newOps[i]);
    }
}

/* *
 * @tc.name: hdidevice_V2_1_getsupportedoperation_002
 * @tc.desc: Verify the GetSupportedOperation function return failed in case of allocate buffer failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getsupportedoperation_002, TestSize.Level0)
{
    std::vector<bool> ops;
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    EXPECT_NE(nullptr, model);
    BuildLiteGraph(model);

    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_1::SharedBuffer buffer {1, 1, 0, 1};
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(HDF_FAILURE)));

    OH_NN_ReturnCode result = hdiDevice->GetSupportedOperation(model, ops);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getsupportedoperation_003
 * @tc.desc: Verify the GetSupportedOperation function return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getsupportedoperation_003, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    std::shared_ptr<const mindspore::lite::LiteGraph> model = nullptr;
    std::vector<bool> ops;
    OH_NN_ReturnCode result = hdiDevice->GetSupportedOperation(model, ops);
    EXPECT_EQ(OH_NN_NULL_PTR, result);
}

/* *
 * @tc.name: hdidevice_V2_1_getsupportedoperation_004
 * @tc.desc: Verify the GetSupportedOperation function return unavalidable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_getsupportedoperation_004, TestSize.Level0)
{
    std::vector<bool> ops {true};
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    EXPECT_NE(nullptr, model);
    BuildLiteGraph(model);

    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_1::SharedBuffer buffer {2, 1, 0, 1};
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(HDF_SUCCESS)));

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), GetSupportedOperation(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(ops), ::testing::Return(HDF_FAILURE)));

    std::vector<bool> newOps {true};
    OH_NN_ReturnCode result = hdiDevice->GetSupportedOperation(model, newOps);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_isfloat16precisionsupported_001
 * @tc.desc: Verify the IsFloat16PrecisionSupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_isfloat16precisionsupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsFloat16PrecisionSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));
    OH_NN_ReturnCode result = hdiDevice->IsFloat16PrecisionSupported(isSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_isfloat16precisionsupported_002
 * @tc.desc: Verify the IsFloat16PrecisionSupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_isfloat16precisionsupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsFloat16PrecisionSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsFloat16PrecisionSupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_isperformancemodesupported_001
 * @tc.desc: Verify the IsPerformanceModeSupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_isperformancemodesupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsPerformanceModeSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));

    bool newIsSupported = false;
    const bool expectIsSupported = false;
    OH_NN_ReturnCode result = hdiDevice->IsPerformanceModeSupported(newIsSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectIsSupported, newIsSupported);
}

/* *
 * @tc.name: hdidevice_V2_1_isperformancemodesupported_002
 * @tc.desc: Verify the IsPerformanceModeSupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_isperformancemodesupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsPerformanceModeSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsPerformanceModeSupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_isprioritysupported_001
 * @tc.desc: Verify the IsPrioritySupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_isprioritysupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsPrioritySupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));

    bool newIsSupported = false;
    bool expectIsSupported = false;
    OH_NN_ReturnCode result = hdiDevice->IsPrioritySupported(newIsSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(newIsSupported, expectIsSupported);
}

/* *
 * @tc.name: hdidevice_V2_1_isprioritysupported_002
 * @tc.desc: Verify the IsPrioritySupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_isprioritysupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsPrioritySupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsPrioritySupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_isdynamicinputsupported_001
 * @tc.desc: Verify the IsDynamicInputSupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_isdynamicinputsupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsDynamicInputSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));

    bool newIsSupported = false;
    bool expectIsSupported = false;
    OH_NN_ReturnCode result = hdiDevice->IsDynamicInputSupported(newIsSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(newIsSupported, expectIsSupported);
}

/* *
 * @tc.name: hdidevice_V2_1_isdynamicinputsupported_002
 * @tc.desc: Verify the IsDynamicInputSupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_isdynamicinputsupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsDynamicInputSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsDynamicInputSupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_isdynamicinputsupported_001
 * @tc.desc: Verify the IsModelCacheSupported function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_ismodelcachesupported_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsModelCacheSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_SUCCESS)));

    bool newIsSupported = false;
    bool expectIsSupported = false;
    OH_NN_ReturnCode result = hdiDevice->IsModelCacheSupported(newIsSupported);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_EQ(expectIsSupported, newIsSupported);
}

/* *
 * @tc.name: hdidevice_V2_1_isdynamicinputsupported_002
 * @tc.desc: Verify the IsModelCacheSupported function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_ismodelcachesupported_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    bool isSupported = false;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), IsModelCacheSupported(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(isSupported), ::testing::Return(HDF_FAILURE)));
    OH_NN_ReturnCode result = hdiDevice->IsModelCacheSupported(isSupported);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodel_001
 * @tc.desc: Verify the PrepareModel function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodel_001, TestSize.Level0)
{
    int32_t allocBufferType = HDF_SUCCESS;
    int32_t prepareType = HDF_SUCCESS;
    OH_NN_ReturnCode result = PrepareModelv2(allocBufferType, prepareType);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodel_002
 * @tc.desc: Verify the PrepareModel function return invalid parameter.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodel_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    std::shared_ptr<const mindspore::lite::LiteGraph> model = nullptr;
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode result = hdiDevice->PrepareModel(model, config, preparedModel);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodel_003
 * @tc.desc: Verify the PrepareModel function return failed.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodel_003, TestSize.Level0)
{
    int32_t allocBufferType = HDF_SUCCESS;
    int32_t prepareType = HDF_FAILURE;
    OH_NN_ReturnCode result = PrepareModelv2(allocBufferType, prepareType);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodel_004
 * @tc.desc: Verify the PrepareModel function return failed.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodel_004, TestSize.Level0)
{
    int32_t allocBufferType = HDF_FAILURE;
    int32_t prepareType = HDF_FAILURE;
    OH_NN_ReturnCode result = PrepareModelv2(allocBufferType, prepareType);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodel_005
 * @tc.desc: Verify the PrepareModel function return failed.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodel_005, TestSize.Level0)
{
    LOGE("PrepareModel hdidevice_V2_1_preparemodel_005");
    OHOS::sptr<V2_1::MockIDevice> sp = OHOS::sptr<V2_1::MockIDevice>(new (std::nothrow) V2_1::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* data = dataArry;
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode result = hdiDevice->PrepareModel(data, config, preparedModel);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodelfrommodelcache_001
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodelfrommodelcache_001, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;

    OHOS::sptr<V2_1::MockIDevice> sp = OHOS::sptr<V2_1::MockIDevice>(new (std::nothrow) V2_1::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::shared_ptr<PreparedModel> preparedModel;

    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_SUCCESS)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    const auto &memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodelfrommodelcache_002
 * @tc.desc: Verify the PrepareModelFromModelCache function return unavailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodelfrommodelcache_002, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_1::MockIDevice> sp = OHOS::sptr<V2_1::MockIDevice>(new (std::nothrow) V2_1::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    OHOS::sptr<V2_1::IPreparedModel> preModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_1>(preModel);

    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodelfrommodelcache_003
 * @tc.desc: Verify the PrepareModelFromModelCache function return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodelfrommodelcache_003, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { nullptr, 0 } };
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_NULL_PTR, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodelfrommodelcache_004
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodelfrommodelcache_004, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_V2_1_getdevicestatus_005");
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_1::MockIDevice> sp = OHOS::sptr<V2_1::MockIDevice>(new (std::nothrow) V2_1::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    config.mode = OH_NN_PERFORMANCE_LOW;
    config.priority = OH_NN_PRIORITY_LOW;
    OHOS::sptr<V2_1::IPreparedModel> preModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_1>(preModel);

    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodelfrommodelcache_005
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodelfrommodelcache_005, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_V2_1_preparemodelfrommodelcache_005");
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_1::MockIDevice> sp = OHOS::sptr<V2_1::MockIDevice>(new (std::nothrow) V2_1::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    config.mode = OH_NN_PERFORMANCE_MEDIUM;
    config.priority = OH_NN_PRIORITY_MEDIUM;
    OHOS::sptr<V2_1::IPreparedModel> preModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_1>(preModel);

    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodelfrommodelcache_006
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodelfrommodelcache_006, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_V2_1_preparemodelfrommodelcache_006");
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_1::MockIDevice> sp = OHOS::sptr<V2_1::MockIDevice>(new (std::nothrow) V2_1::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    config.mode = OH_NN_PERFORMANCE_HIGH;
    config.priority = OH_NN_PRIORITY_HIGH;
    OHOS::sptr<V2_1::IPreparedModel> preModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_1>(preModel);

    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_preparemodelfrommodelcache_007
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_preparemodelfrommodelcache_007, TestSize.Level0)
{
    LOGE("GetDeviceStatus hdidevice_V2_1_preparemodelfrommodelcache_007");
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_1::MockIDevice> sp = OHOS::sptr<V2_1::MockIDevice>(new (std::nothrow) V2_1::MockIDevice());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(sp);
    EXPECT_NE(hdiDevice, nullptr);

    std::vector<Buffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    config.mode = OH_NN_PERFORMANCE_EXTREME;
    OHOS::sptr<V2_1::IPreparedModel> preModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_1>(preModel);

    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    bool isUpdatable = false;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel, isUpdatable);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/* *
 * @tc.name: hdidevice_V2_1_allocatebuffer_001
 * @tc.desc: Verify the AllocateBuffer function return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_allocatebuffer_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_1::SharedBuffer buffer;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(buffer), ::testing::Return(HDF_FAILURE)));

    size_t length = 8;
    void *result = hdiDevice->AllocateBuffer(length);
    EXPECT_EQ(nullptr, result);
    hdiDevice->ReleaseBuffer(result);
}

/* *
 * @tc.name: hdidevice_V2_1_allocatebuffer_002
 * @tc.desc: Verify the AllocateBuffer function return nullptr and HDF_FAILURE.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_allocatebuffer_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 8;
    void *result = hdiDevice->AllocateBuffer(length);
    EXPECT_EQ(nullptr, result);
    hdiDevice->ReleaseBuffer(result);
}

/* *
 * @tc.name: hdidevice_V2_1_allocatebuffer_003
 * @tc.desc: Verify the AllocateBuffer function return nullptr in case of 0 size.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_allocatebuffer_003, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 0;
    void *result = hdiDevice->AllocateBuffer(length);
    EXPECT_EQ(nullptr, result);
}

/* *
 * @tc.name: hdidevice_V2_1_allocatebuffer_004
 * @tc.desc: Verify the AllocateBuffer function return nullptr in case of 0 size.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_allocatebuffer_004, TestSize.Level0)
{
    LOGE("AllocateBuffer hdidevice_V2_1_preparemodelfrommodelcache_007");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 0;
    int fd = 0;
    OH_NN_ReturnCode result = hdiDevice->AllocateBuffer(length, fd);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/* *
 * @tc.name: hdidevice_V2_1_allocatebuffer_005
 * @tc.desc: Verify the AllocateBuffer function return nullptr in case of 0 size.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_allocatebuffer_005, TestSize.Level0)
{
    LOGE("AllocateBuffer hdidevice_V2_1_allocatebuffer_005");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 1;
    int fd = 0;
    OH_NN_ReturnCode result = hdiDevice->AllocateBuffer(length, fd);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, result);
}

/* *
 * @tc.name: hdidevice_V2_1_allocatebuffer_006
 * @tc.desc: Verify the AllocateBuffer function return nullptr in case of 0 size.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_allocatebuffer_006, TestSize.Level0)
{
    LOGE("AllocateBuffer hdidevice_V2_1_allocatebuffer_006");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(HDF_SUCCESS));

    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 1;
    int fd = 0;
    OH_NN_ReturnCode result = hdiDevice->AllocateBuffer(length, fd);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/* *
 * @tc.name: hdidevice_V2_1_releasebuffer_001
 * @tc.desc: Verify the ReleaseBuffer function validate buffer success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_releasebuffer_001, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_SUCCESS));

    EXPECT_NE(hdiDevice, nullptr);
    hdiDevice->ReleaseBuffer(buffer);
    const auto &memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}

/* *
 * @tc.name: hdidevice_V2_1_releasebuffer_002
 * @tc.desc: Verify the ReleaseBuffer function validate AllocateBuffer return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_releasebuffer_002, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    V2_1::SharedBuffer sharedbuffer;
    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<1>(sharedbuffer), ::testing::Return(HDF_FAILURE)));

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));

    size_t length = 8;
    void *buffer = hdiDevice->AllocateBuffer(length);
    hdiDevice->ReleaseBuffer(buffer);
}

/* *
 * @tc.name: hdidevice_V2_1_releasebuffer_003
 * @tc.desc: Verify the ReleaseBuffer function validate param buffer is nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_releasebuffer_003, TestSize.Level0)
{
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    void *buffer = nullptr;
    hdiDevice->ReleaseBuffer(buffer);
}

/* *
 * @tc.name: hdidevice_V2_1_releasebuffer_004
 * @tc.desc: Verify the ReleaseBuffer function validate invalid buffer.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_releasebuffer_004, TestSize.Level0)
{
    const size_t length = 100;
    auto* buffer = new(std::nothrow) char[length];
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    hdiDevice->ReleaseBuffer(buffer);
    delete[] buffer;
    buffer = nullptr;
}

/* *
 * @tc.name: hdidevice_V2_1_releasebuffer_005
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_releasebuffer_005, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));

    hdiDevice->ReleaseBuffer(buffer);
    const auto &memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}

/* *
 * @tc.name: hdidevice_V2_1_releasebuffer_007
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_releasebuffer_007, TestSize.Level0)
{
    LOGE("ReleaseBuffer hdidevice_V2_1_releasebuffer_007");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_FAILURE));

    int fd = 0;
    size_t length = 1;
    OH_NN_ReturnCode ret = hdiDevice->ReleaseBuffer(fd, length);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/* *
 * @tc.name: hdidevice_V2_1_releasebuffer_008
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_releasebuffer_008, TestSize.Level0)
{
    LOGE("ReleaseBuffer hdidevice_V2_1_releasebuffer_008");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    EXPECT_CALL(*((V2_1::MockIDevice *)device.GetRefPtr()), ReleaseBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(HDF_SUCCESS));

    int fd = 0;
    size_t length = 1;
    OH_NN_ReturnCode ret = hdiDevice->ReleaseBuffer(fd, length);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/* *
 * @tc.name: hdidevice_V2_1_allocatetensorbuffer_001
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_allocatetensorbuffer_001, TestSize.Level0)
{
    LOGE("AllocateTensorBuffer hdidevice_V2_1_allocatetensorbuffer_001");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 0;
    std::shared_ptr<TensorDesc> tensor;
    void* ret = hdiDevice->AllocateTensorBuffer(length, tensor);
    EXPECT_EQ(nullptr, ret);
}

/* *
 * @tc.name: hdidevice_V2_1_allocatetensorbuffer_002
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_allocatetensorbuffer_002, TestSize.Level0)
{
    LOGE("AllocateTensorBuffer hdidevice_V2_1_allocatetensorbuffer_002");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);

    size_t length = 0;
    std::shared_ptr<NNTensor> tensor;
    void* ret = hdiDevice->AllocateTensorBuffer(length, tensor);
    EXPECT_EQ(nullptr, ret);
}

/* *
 * @tc.name: hdidevice_V2_1_prepareofflinemodel_001
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_prepareofflinemodel_001, TestSize.Level0)
{
    LOGE("PrepareOfflineModel hdidevice_V2_1_prepareofflinemodel_001");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode ret = hdiDevice->PrepareOfflineModel(nullptr, config, preparedModel);
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
}

/* *
 * @tc.name: hdidevice_V2_1_prepareofflinemodel_002
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_prepareofflinemodel_002, TestSize.Level0)
{
    LOGE("PrepareOfflineModel hdidevice_V2_1_prepareofflinemodel_002");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    mindspore::lite::LiteGraph::Node node;
    mindspore::lite::LiteGraph::Node* testNode = &node;
    model->all_nodes_.emplace_back(testNode);
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode ret = hdiDevice->PrepareOfflineModel(model, config, preparedModel);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/* *
 * @tc.name: hdidevice_V2_1_prepareofflinemodel_003
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_prepareofflinemodel_003, TestSize.Level0)
{
    LOGE("PrepareOfflineModel hdidevice_V2_1_prepareofflinemodel_003");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    mindspore::lite::LiteGraph::Node node;
    uint32_t indice = 0;
    node.input_indices_.emplace_back(indice);
    node.input_indices_.emplace_back(indice);
    mindspore::lite::LiteGraph::Node* testNode = &node;
    model->all_nodes_.emplace_back(testNode);
    model->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    model->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode ret = hdiDevice->PrepareOfflineModel(model, config, preparedModel);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/* *
 * @tc.name: hdidevice_V2_1_prepareofflinemodel_004
 * @tc.desc: Verify the ReleaseBuffer function validate moc object's ReleaseBuffer return failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_V2_1_prepareofflinemodel_004, TestSize.Level0)
{
    LOGE("PrepareOfflineModel hdidevice_V2_1_prepareofflinemodel_004");
    OHOS::sptr<V2_1::INnrtDevice> device = V2_1::INnrtDevice::Get(false);
    std::unique_ptr<HDIDeviceV2_1> hdiDevice = std::make_unique<HDIDeviceV2_1>(device);
    EXPECT_NE(hdiDevice, nullptr);
    
    std::shared_ptr<mindspore::lite::LiteGraph> model = std::make_shared<mindspore::lite::LiteGraph>();
    mindspore::lite::LiteGraph::Node node;
    uint32_t indice = 0;
    node.input_indices_.emplace_back(indice);
    node.input_indices_.emplace_back(indice);
    mindspore::lite::LiteGraph::Node* testNode = &node;
    model->all_nodes_.emplace_back(testNode);

    char a = 'a';
    mindspore::lite::DataType data_type = mindspore::lite::DataType::DATA_TYPE_INT32;
    int dim = 1;
    int32_t *dims = &dim;
    uint32_t dims_size = 1;
    mindspore::lite::Format format = mindspore::lite::Format::FORMAT_HWCK;
    uint8_t datas = 0;
    uint8_t *data = &datas;
    uint32_t data_size = 2;
    mindspore::lite::QuantParam quant_params;
    uint32_t quant_params_size = 0;
    mindspore::lite::TensorPtr ptr2 = mindspore::lite::MindIR_Tensor_Create(&a, data_type, dims, dims_size,
                               format, data, data_size,
                               &quant_params, quant_params_size);
    std::vector<uint8_t> offlineModel2 = mindspore::lite::MindIR_Tensor_GetData(ptr2);

    model->all_tensors_.emplace_back(ptr2);
    model->all_tensors_.emplace_back(ptr2);
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode ret = hdiDevice->PrepareOfflineModel(model, config, preparedModel);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);

    testing::Mock::AllowLeak(device.GetRefPtr());
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace NNRt_V2_1 {
namespace UnitTest {
class LiteGraphToHDIModelTest : public testing::Test {
public:
    LiteGraphToHDIModelTest() = default;
    ~LiteGraphToHDIModelTest() = default;
public:
    std::vector<uint32_t> m_inputs{0, 1};
    std::vector<uint32_t> m_outputs{2};
    std::vector<uint32_t> m_param{3};
    std::vector<int32_t> m_input_dim{3, 3};
    std::vector<int32_t> m_output_dim{3, 3};
    std::vector<int32_t> m_param_dim{};
};

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_001, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_001");
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(nullptr, tensorBuffer);
    EXPECT_EQ(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_002, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_002");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {0, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_EQ(nullptr, model);

    uint8_t *mmapPtr = static_cast<uint8_t *>(mmap(nullptr,
        tensorBuffer.bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, tensorBuffer.fd, 0));
    EXPECT_EQ(MAP_FAILED, mmapPtr);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_003, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_003");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::SubGraph* subGraph = new (std::nothrow) MSLITE::LiteGraph::SubGraph();
    subGraph->name_ = "NNRt_SubGraph";
    subGraph->input_indices_ = {1, 1, 1, 1};
    subGraph->output_indices_ = {1, 1, 1, 1};
    subGraph->node_indices_ = {1, 1, 1, 1};

    void* tp = MSLITE::MindIR_Tensor_Create();

    liteGraph.get()->all_tensors_.emplace_back(tp);
    liteGraph.get()->sub_graphs_.emplace_back(subGraph);

    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 1, 1, 1};

    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);

    uint8_t *mmapPtr = static_cast<uint8_t *>(mmap(nullptr,
        tensorBuffer.bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, tensorBuffer.fd, 0));
    EXPECT_EQ(MAP_FAILED, mmapPtr);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_004, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_004");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    liteGraph.get()->all_nodes_.emplace_back(nullptr);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_EQ(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_005, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_005");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();
    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_EQ(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_006
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_006, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_006");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_ABS};

    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_007
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_007, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_007");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int8_t num = 1;
    int8_t* fuseData = &num;
    mindspore::lite::ActivationType type = NNToMS::TransfromFusionType(static_cast<OH_NN_FuseType>(*fuseData));
    void* primitive = mindspore::lite::MindIR_AddFusion_CreatePrimitive(type);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_008
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_008, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_008");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t keepDims {0};
    void* primitive = mindspore::lite::MindIR_All_CreatePrimitive(keepDims);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_009
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_009, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_009");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis {-1};
    int64_t topK {1};
    bool keepDims {false};
    bool outMaxValue {false};
    void* primitive = mindspore::lite::MindIR_ArgMaxFusion_CreatePrimitive(axis, topK, keepDims, outMaxValue);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_010
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_010, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_010");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t summarize {0};
    void* primitive = mindspore::lite::MindIR_Assert_CreatePrimitive(summarize);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_011
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_011, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_011");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> kernelSize;
    std::vector<int64_t> pad;
    std::vector<int64_t> strides;
    mindspore::lite::PadMode padMode {mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    mindspore::lite::RoundMode roundMode {mindspore::lite::ROUND_MODE_FLOOR};
    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    bool global {false};
    void* primitive = mindspore::lite::MindIR_AvgPoolFusion_CreatePrimitive(kernelSize, strides, pad,
        padMode, roundMode, format, global, activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_012
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_012, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_012");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> blockSize;
    std::vector<std::vector<int64_t>> crops;
    void* primitive = mindspore::lite::MindIR_BatchToSpaceND_CreatePrimitive(blockSize, crops);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_013
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_013, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_013");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float epsilon {0.0001f};
    void* primitive = mindspore::lite::MindIR_FusedBatchNorm_CreatePrimitive(epsilon);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_014
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_014, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_014");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float epsilon {0.0001f};
    void* primitive = mindspore::lite::MindIR_FusedBatchNorm_CreatePrimitive(epsilon);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_015
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_015, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_015");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_BiasAdd_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_016
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_016, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_016");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> shape;
    void* primitive = mindspore::lite::MindIR_BroadcastTo_CreatePrimitive(shape);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_017
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_017, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_017");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Cast_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_018
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_018, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_018");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Ceil_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_019
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_019, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_019");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float max {0.0f};
    float min {0.0f};
    void* primitive = mindspore::lite::MindIR_Clip_CreatePrimitive(max, min);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_020
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_020, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_020");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis{0};
    void* primitive = mindspore::lite::MindIR_Concat_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_021
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_021, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_021");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t dataType {0};
    std::vector<float> value;
    void* primitive = mindspore::lite::MindIR_ConstantOfShape_CreatePrimitive(dataType, value);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_022
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_022, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_022");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t group {1};
    int64_t inChannel {0};
    int64_t outChannel {0};
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> strides;
    std::vector<int64_t> padList;
    std::vector<int64_t> dilation;
    std::vector<int64_t> outputPaddings;
    mindspore::lite::PadMode padMode{mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = MindIR_Conv2dTransposeFusion_CreatePrimitive(kernelSize,
        strides, dilation, padMode, padList, group, inChannel, outChannel,
        activationType, outputPaddings);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_023
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_023, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_023");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Cos_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_024
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_024, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_024");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis {0};
    std::vector<int64_t> offset;
    void* primitive = mindspore::lite::MindIR_Crop_CreatePrimitive(axis, offset);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_025
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_025, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_025");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t blockSize {0};
    std::string mode;
    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    void* primitive = mindspore::lite::MindIR_DepthToSpace_CreatePrimitive(blockSize, format, mode);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_026
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_026, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_026");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t inputSize {0};
    std::vector<float> scale;
    float nmsIoUThreshold {0.0f};
    float nmsScoreThreshold {0.0f};
    int64_t maxDetections {0};
    int64_t detectionsPerClass {0};
    int64_t maxClassesPerDetection {0};
    int64_t numClasses {0};
    bool useRegularNms {false};
    bool outQuantized {false};
    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    void* primitive = mindspore::lite::MindIR_DetectionPostProcess_CreatePrimitive(format, inputSize, scale,
        nmsIoUThreshold, nmsScoreThreshold, maxDetections, detectionsPerClass, maxClassesPerDetection,
        numClasses, useRegularNms, outQuantized);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_027
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_027, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_027");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::EltwiseMode mode {mindspore::lite::ELTWISE_MODE_PROD};
    void* primitive = mindspore::lite::MindIR_Eltwise_CreatePrimitive(mode);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_028
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_028, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_028");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Equal_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_029
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_029, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_029");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Erf_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_030
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_030, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_030");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float base {-1.0f};
    float scale {1.0f};
    float shift {0.0f};
    void* primitive = mindspore::lite::MindIR_ExpFusion_CreatePrimitive(base, scale, shift);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_031
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_031, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_031");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_ExpandDims_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_032
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_032, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_032");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Fill_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_033
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_033, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_033");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis {1};
    void* primitive = mindspore::lite::MindIR_Flatten_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_034
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_034, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_034");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Floor_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_035
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_035, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_035");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    bool hasBias {false};
    bool useAxis {false};
    int64_t axis {0};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_FullConnection_CreatePrimitive(hasBias, useAxis,
        axis, activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_036
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_036, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_036");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Gather_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_037
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_037, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_037");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_GatherNd_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_038
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_038, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_038");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::ActivationType activationType = mindspore::lite::ACTIVATION_TYPE_GELU;
    float alpha = 0.0f;
    float minVal = 0.0f;
    float maxVal = 0.0f;
    bool approximate = false;
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType,
        alpha, minVal, maxVal, approximate);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_039
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_039, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_039");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Greater_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_040
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_040, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_040");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_GreaterEqual_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_041
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_041, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_041");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::ActivationType activationType = mindspore::lite::ACTIVATION_TYPE_HSIGMOID;
    float alpha = 0.0f;
    float minVal = 0.0f;
    float maxVal = 0.0f;
    bool approximate = false;
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType,
        alpha, minVal, maxVal, approximate);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_042
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_042, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_042");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float epsilon {0.0f};
    void* primitive = mindspore::lite::MindIR_InstanceNorm_CreatePrimitive(epsilon);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_043
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_043, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_043");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float epsilon {0.0f};
    void* primitive = mindspore::lite::MindIR_InstanceNorm_CreatePrimitive(epsilon);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_044
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_044, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_044");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> axis;
    float epsilon {1e-6};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_L2NormalizeFusion_CreatePrimitive(axis, epsilon, activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_045
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_045, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_045");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t beginNormAxis {1};
    float epsilon {1e-7};
    bool elementwiseAffine {true};
    int64_t beginParamsAxis {1};
    void* primitive = mindspore::lite::MindIR_LayerNormFusion_CreatePrimitive(beginNormAxis,
        epsilon, elementwiseAffine, beginParamsAxis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_046
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_046, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_046");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_LEAKY_RELU};
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_047
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_047, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_047");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Less_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_048
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_048, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_048");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_LessEqual_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_049
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_049, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_049");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Log_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_050
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_050, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_050");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis {0};
    void* primitive = mindspore::lite::MindIR_LogSoftmax_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_051
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_051, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_051");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_LogicalAnd_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_052
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_052, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_052");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_LogicalNot_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_053
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_053, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_053");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_LogicalOr_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_054
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_054, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_054");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t depthRadius {0};
    float bias {0.0f};
    float alpha {0.0f};
    float beta {0.0f};
    std::string normRegion {"ACROSS_CHANNELS"};
    void* primitive = mindspore::lite::MindIR_LRN_CreatePrimitive(depthRadius, bias, alpha,
        beta, normRegion);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_055
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_055, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_055");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    bool bidirectional {false};
    bool hasBias {false};
    int64_t inputSize {0};
    int64_t hiddenSize {0};
    int64_t numLayers {0};
    int64_t numDirections {0};
    float dropout {0.0f};
    float zoneoutCell {0.0f};
    float zoneoutHidden {0.0f};
    int64_t projSize {0};
    void* primitive = mindspore::lite::MindIR_LSTM_CreatePrimitive(bidirectional, hasBias, inputSize,
        hiddenSize, numLayers, numDirections, dropout, zoneoutCell, zoneoutHidden, projSize);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_056
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_056, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_056");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Maximum_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_057
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_057, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_057");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> kernelSize;
    std::vector<int64_t> pad;
    std::vector<int64_t> strides;
    mindspore::lite::PadMode padMode {mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    bool global {false};
    void* primitive = MindIR_MaxPoolFusion_CreatePrimitive(kernelSize, strides, pad,
        padMode, format, global, activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_058
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_058, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_058");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Minimum_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_059
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_059, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_059");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Mod_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_060
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_060, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_060");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_MulFusion_CreatePrimitive(activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_061
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_061, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_061");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Neg_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_062
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_062, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_062");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_NotEqual_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_063
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_063, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_063");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis {-1};
    void* primitive = mindspore::lite::MindIR_OneHot_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_064
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_064, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_064");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<std::vector<int64_t>> paddings;
    float constantValue {0.0f};
    mindspore::lite::PaddingMode paddingMode {mindspore::lite::PADDING_MODE_CONSTANT};
    void* primitive = MindIR_PadFusion_CreatePrimitive(paddings, paddingMode, constantValue);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_065
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_065, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_065");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float scale {1.0f};
    float shift {0.0f};
    void* primitive = mindspore::lite::MindIR_PowFusion_CreatePrimitive(scale, shift);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_066
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_066, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_066");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    bool channelShared{false};
    void* primitive = mindspore::lite::MindIR_PReLUFusion_CreatePrimitive(channelShared);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_067
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_067, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_067");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    const uint64_t* srcT{nullptr};
    const uint64_t* dstT{nullptr};
    int64_t axis {0};
    void* primitive = mindspore::lite::MindIR_QuantDTypeCast_CreatePrimitive(*srcT, *dstT, axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_068
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_068, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_068");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t dType {0.0f};
    int64_t start {0};
    int64_t limit {0};
    int64_t delta {1};
    void* primitive = mindspore::lite::MindIR_Range_CreatePrimitive(dType, start, limit, delta);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_069
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_069, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_069");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Rank_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_070
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_070, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_070");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Reciprocal_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_071
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_071, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_071");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::ReduceMode mode {mindspore::lite::REDUCE_MODE_ALL};
    float coeff {0.0f};
    bool reduceToEnd {false};
    bool keepDims {false};
    void* primitive = mindspore::lite::MindIR_ReduceFusion_CreatePrimitive(keepDims, mode, reduceToEnd, coeff);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_072
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_072, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_072");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float alpha{0.0f};
    float minVal{0.0f};
    float maxVal{0.0f};
    bool approximate{false};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_RELU6};
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_073
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_073, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_073");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Reshape_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_074
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_074, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_074");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float cubicCoeff{0.0f};
    float extrapolationValue{0.0f};
    mindspore::lite::NearestMode nearestMode{mindspore::lite::NEAREST_MODE_NORMAL};
    mindspore::lite::ResizeMethod method {mindspore::lite::RESIZE_METHOD_LINEAR};
    uint64_t newHeight{0};
    uint64_t newWidth{0};
    bool preserveAspectRatio{false};
    mindspore::lite::CoordinateTransformMode coordinateTransformMode {
        mindspore::lite::COORDINATE_TRANSFORM_MODE_ASYMMETRIC};
    uint64_t excludeOutside{0};
    void* primitive = mindspore::lite::MindIR_Resize_CreatePrimitive(method, newHeight, newWidth,
        preserveAspectRatio, coordinateTransformMode, cubicCoeff, excludeOutside,
        extrapolationValue, nearestMode);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_075
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_075, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_075");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Round_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_076
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_076, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_076");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Rsqrt_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_077
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_077, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_077");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    const uint64_t* axis{nullptr};
    void* primitive = mindspore::lite::MindIR_ScaleFusion_CreatePrimitive(*axis, activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_078
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_078, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_078");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_ScatterNd_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_079
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_079, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_079");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Select_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_080
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_080, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_080");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float alpha{0.0f};
    float minVal{0.0f};
    float maxVal{0.0f};
    bool approximate{false};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_SIGMOID};
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha, minVal,
        maxVal, approximate);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_081
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_081, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_081");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Sin_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_082
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_082, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_082");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    int64_t blockSize {0};
    void* primitive = mindspore::lite::MindIR_SpaceToDepth_CreatePrimitive(blockSize, format);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_083
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_083, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_083");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_SparseToDense_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_084
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_084, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_084");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Square_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_085
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_085, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_085");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_SWISH};
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_086
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_086, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_086");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis {0};
    void* primitive = mindspore::lite::MindIR_Unstack_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_087
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_087, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_087");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Where_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_088
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_088, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_088");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Shape_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_089
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_089, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_089");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Unsqueeze_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_090
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_090, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_090");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t inChannel{0};
    int64_t outChannel{0};
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> strides;
    std::vector<int64_t> pad;
    std::vector<int64_t> dilation;
    mindspore::lite::PadMode padMode{mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_Conv2DFusion_CreatePrimitive(kernelSize, strides,
        dilation, padMode, pad, inChannel, inChannel, outChannel, activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_091
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_091, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_091");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_DivFusion_CreatePrimitive(activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_092
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_092, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_092");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    bool transposeA{false};
    bool transposeB{false};
    void* primitive = mindspore::lite::MindIR_MatMulFusion_CreatePrimitive(transposeA, transposeB, activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_093
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_093, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_093");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> axes;
    void* primitive = mindspore::lite::MindIR_SliceFusion_CreatePrimitive(axes);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_094
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_094, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_094");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Softmax_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_095
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_095, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_095");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<std::vector<int64_t>> paddings;
    std::vector<int64_t> block_shape {};
    void* primitive = mindspore::lite::MindIR_SpaceToBatchND_CreatePrimitive(block_shape, paddings);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_096
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_096, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_096");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t outputNum {0};
    std::vector<int64_t> sizeSplits;
    int64_t axis {0};
    void* primitive = mindspore::lite::MindIR_Split_CreatePrimitive(outputNum, sizeSplits, axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_097
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_097, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_097");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Sqrt_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_098
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_098, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_098");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_SquaredDifference_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_099
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_099, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_099");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Squeeze_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_100
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_100, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_100");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis = {0};
    void* primitive = mindspore::lite::MindIR_Stack_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_101
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_101, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_101");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t beginMask = {0};
    int64_t endMask = {0};
    int64_t ellipsisMask = {0};
    int64_t newAxisMask = {0};
    int64_t shrinkAxisMask = {0};
    void* primitive = mindspore::lite::MindIR_StridedSlice_CreatePrimitive(beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_102
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_102, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_102");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    mindspore::lite::ActivationType  activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_SubFusion_CreatePrimitive(activationType);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_103
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_103, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_103");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> dims {0};
    void* primitive = mindspore::lite::MindIR_TileFusion_CreatePrimitive(dims);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_104
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_104, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_104");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    int64_t axis {0};
    bool sorted {true};
    void* primitive = mindspore::lite::MindIR_TopKFusion_CreatePrimitive(sorted, axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_105
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_105, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_105");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    void* primitive = mindspore::lite::MindIR_Transpose_CreatePrimitive();

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_106
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_106, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_106");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Unsqueeze_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_107
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_107, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_107");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Unsqueeze_CreatePrimitive(axis);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_108
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_108, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_108");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::SubGraph* subGraph = new (std::nothrow) MSLITE::LiteGraph::SubGraph();
    subGraph->name_ = "NNRt_SubGraph";
    subGraph->input_indices_ = {1, 1, 1, 1};
    subGraph->output_indices_ = {1, 1, 1, 1};
    subGraph->node_indices_ = {1, 1, 1, 1};

    void* tp = MSLITE::MindIR_Tensor_Create();

    liteGraph.get()->all_tensors_.emplace_back(tp);
    liteGraph.get()->all_tensors_.emplace_back(nullptr);
    liteGraph.get()->sub_graphs_.emplace_back(subGraph);

    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 1, 1, 1};

    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);

    uint8_t *mmapPtr = static_cast<uint8_t *>(mmap(nullptr,
        tensorBuffer.bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, tensorBuffer.fd, 0));
    EXPECT_EQ(MAP_FAILED, mmapPtr);
}

/**
 * @tc.name: litegraphtohdimodeltest_hdimodel_destroy_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_hdimodel_destroy_001, TestSize.Level0)
{
    LOGE("HDIModel_Destroy litegraphtohdimodeltest_hdimodel_destroy_001");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_ABS};

    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};

    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V2_1::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
    HDIModel_Destroy(&model);
}
} // namespace UnitTest
} // namespace V1
} // namespace NeuralNetworkRuntime
} // namespace OHOS