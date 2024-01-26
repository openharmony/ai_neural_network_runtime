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
#include "test/unittest/common/v2_0/mock_idevice.h"
#include "test/unittest/common/file_utils.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
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
        model->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create(model->name_,
            mindspore::lite::DATA_TYPE_FLOAT32, dim, mindspore::lite::FORMAT_NCHW, data, quant_params));
    }

    for (size_t indexOutput = 0; indexOutput < model->output_indices_.size(); ++indexOutput) {
        model->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create(model->name_,
            mindspore::lite::DATA_TYPE_FLOAT32, dim, mindspore::lite::FORMAT_NCHW, data, quant_params));
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
    EXPECT_EQ(OH_NN_SUCCESS, result);
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
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
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
    EXPECT_EQ(OH_NN_SUCCESS, result);
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
 * @tc.name: hdidevice_preparemodelfrommodelcache_001
 * @tc.desc: Verify the PrepareModelFromModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIDeviceTest, hdidevice_preparemodelfrommodelcache_001, TestSize.Level0)
{
    size_t length = 100;
    void *buffer = nullptr;
    GetBuffer(buffer, length);

    std::vector<ModelBuffer> modelCache = { { buffer, 100 } };
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

    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel);
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

    std::vector<ModelBuffer> modelCache = { { buffer, 100 } };
    ModelConfig config;
    OHOS::sptr<V2_0::IPreparedModel> preModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel());
    EXPECT_NE(preModel, nullptr);

    std::shared_ptr<PreparedModel> preparedModel = std::make_shared<HDIPreparedModelV2_0>(preModel);

    OHOS::sptr<V2_0::IPreparedModel> iPreparedModel =
        OHOS::sptr<V2_0::MockIPreparedModel>(new (std::nothrow) V2_0::MockIPreparedModel);
    EXPECT_CALL(*sp, PrepareModelFromModelCache(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<2>(iPreparedModel), ::testing::Return(HDF_FAILURE)));

    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel);
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

    std::vector<ModelBuffer> modelCache = { { nullptr, 0 } };
    ModelConfig config;
    std::shared_ptr<PreparedModel> preparedModel;
    OH_NN_ReturnCode result = hdiDevice->PrepareModelFromModelCache(modelCache, config, preparedModel);
    EXPECT_EQ(OH_NN_NULL_PTR, result);
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
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
