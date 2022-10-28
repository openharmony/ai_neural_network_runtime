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

#include "common/log.h"
#include "frameworks/native/device_manager.h"
#include "frameworks/native/hdi_device.h"
#include "test/unittest/common/mock_idevice.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class DeviceManagerTest : public testing::Test {
protected:
    void MockInit(OHOS::sptr<V1_0::MockIDevice> device, const std::vector<int32_t>& typeVect,
        const std::string& deviceName, const std::string& vendorName);
};

void DeviceManagerTest::MockInit(OHOS::sptr<V1_0::MockIDevice> device, const std::vector<int32_t>& typeVect,
    const std::string& deviceName, const std::string& vendorName)
{
    const size_t typeSize = 4;
    int index = 0;
    EXPECT_EQ(typeSize, typeVect.size());
    EXPECT_CALL(*device, GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceName),
        ::testing::Return(typeVect[index++])));

    EXPECT_CALL(*device, GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(vendorName),
        ::testing::Return(typeVect[index++])));

    V1_0::DeviceStatus deviceStatus = V1_0::DeviceStatus::AVAILABLE;
    EXPECT_CALL(*device, GetDeviceStatus(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceStatus),
        ::testing::Return(typeVect[index++])));

    uint32_t majorVer = 1;
    uint32_t minorVer = 0;
    EXPECT_CALL(*device, GetVersion(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(majorVer), ::testing::SetArgReferee<1>(minorVer),
        ::testing::Return(typeVect[index++])));
}

/**
 * @tc.name: devicemanager_getalldeviceid_001
 * @tc.desc: Verify the GetAllDeviceId function return deviceid list is not null.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_getalldeviceid_001, TestSize.Level0)
{
    auto &deviceManager = DeviceManager::GetInstance();
    std::vector<size_t> idVect = deviceManager.GetAllDeviceId();
    EXPECT_NE((size_t)0, idVect.size());

    const size_t expectDeviceId {std::hash<std::string>{}("MockDevice_MockVendor")};
    EXPECT_EQ(expectDeviceId, idVect[0]);

    const std::string expectDeviceName = "MockDevice";
    std::string deviceName = "";
    std::shared_ptr<Device> retDevice = deviceManager.GetDevice(idVect[0]);
    retDevice->GetDeviceName(deviceName);
    EXPECT_EQ(deviceName, expectDeviceName);
}

/**
 * @tc.name: devicemanager_getdevice_001
 * @tc.desc: Verify the GetDevice function return nullptr in case of deviceId invalid.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_getdevice_001, TestSize.Level0)
{
    auto &deviceManager = DeviceManager::GetInstance();
    const size_t deviceId = 1;
    std::shared_ptr<Device> result = deviceManager.GetDevice(deviceId);
    EXPECT_EQ(nullptr, result);
}

/**
 * @tc.name: devicemanager_getdevice_002
 * @tc.desc: Verify the GetDevice function validate device name return specified device name.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_getdevice_002, TestSize.Level0)
{
    auto &deviceManager = DeviceManager::GetInstance();
    std::vector<size_t> idVect = deviceManager.GetAllDeviceId();
    EXPECT_EQ((size_t)1, idVect.size());
    size_t deviceId = idVect[0];
    std::shared_ptr<Device> result = deviceManager.GetDevice(deviceId);
    EXPECT_NE(nullptr, result);

    const std::string expectDeviceNameA = "MockDevice";
    std::string deviceName = "";
    result->GetDeviceName(deviceName);
    EXPECT_EQ(deviceName, expectDeviceNameA);
}

/**
 * @tc.name: devicemanager_registerdevice_001
 * @tc.desc: Verify the RegisterDevice function register repeatly.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_registerdevice_001, TestSize.Level0)
{
    std::vector<int32_t> typeVect = {HDF_SUCCESS, HDF_SUCCESS, HDF_SUCCESS, HDF_SUCCESS};
    OHOS::sptr<V1_0::MockIDevice> device = OHOS::sptr<V1_0::MockIDevice>(new (std::nothrow) V1_0::MockIDevice());
    EXPECT_NE(device.GetRefPtr(), nullptr);

    std::string deviceName = "MockDevice";
    std::string vendorName = "MockVendor";
    MockInit(device, typeVect, deviceName, vendorName);

    std::function<std::shared_ptr<HDIDevice>()> creator =
        [&device]()->std::shared_ptr<HDIDevice> {return std::make_shared<HDIDevice>(device);};
    auto& deviceManager = DeviceManager::GetInstance();
    OH_NN_ReturnCode result = deviceManager.RegisterDevice(creator);
    EXPECT_EQ(OH_NN_FAILED, result);
}

/**
 * @tc.name: devicemanager_registerdevice_002
 * @tc.desc: Verify the RegisterDevice function return invalid parameter.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_registerdevice_002, TestSize.Level0)
{
    std::function<std::shared_ptr<HDIDevice>()> creator =
        []()->std::shared_ptr<HDIDevice> {return nullptr;};
    auto& deviceManager = DeviceManager::GetInstance();
    OH_NN_ReturnCode result = deviceManager.RegisterDevice(creator);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: devicemanager_registerdevice_003
 * @tc.desc: Verify the RegisterDevice function return unavailable device in case of device name invalid param.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_registerdevice_003, TestSize.Level0)
{
    std::vector<int32_t> typeVect = {HDF_FAILURE, HDF_SUCCESS, HDF_SUCCESS, HDF_SUCCESS};
    OHOS::sptr<V1_0::MockIDevice> device = OHOS::sptr<V1_0::MockIDevice>(new (std::nothrow) V1_0::MockIDevice());
    EXPECT_NE(device.GetRefPtr(), nullptr);

    std::string deviceName = "MockDevice";
    std::string vendorName = "MockVendor";
    MockInit(device, typeVect, deviceName, vendorName);

    std::function<std::shared_ptr<HDIDevice>()> creator =
        [&device]()->std::shared_ptr<HDIDevice> {return std::make_shared<HDIDevice>(device);};
    auto& deviceManager = DeviceManager::GetInstance();
    OH_NN_ReturnCode result = deviceManager.RegisterDevice(creator);
    EXPECT_EQ(OH_NN_UNAVALIDABLE_DEVICE, result);
}

/**
 * @tc.name: devicemanager_registerdevice_004
 * @tc.desc: Verify the RegisterDevice function return unavailable device in case of vendor name failure.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_registerdevice_004, TestSize.Level0)
{
    std::vector<int32_t> typeVect = {HDF_SUCCESS, HDF_FAILURE, HDF_SUCCESS, HDF_SUCCESS};
    OHOS::sptr<V1_0::MockIDevice> device = OHOS::sptr<V1_0::MockIDevice>(new (std::nothrow) V1_0::MockIDevice());
    EXPECT_NE(device.GetRefPtr(), nullptr);

    std::string deviceName = "MockDevice";
    std::string vendorName = "MockVendor";
    MockInit(device, typeVect, deviceName, vendorName);

    std::function<std::shared_ptr<HDIDevice>()> creator =
        [&device]()->std::shared_ptr<HDIDevice> {return std::make_shared<HDIDevice>(device);};
    auto& deviceManager = DeviceManager::GetInstance();
    OH_NN_ReturnCode result = deviceManager.RegisterDevice(creator);
    EXPECT_EQ(OH_NN_UNAVALIDABLE_DEVICE, result);
}

/**
 * @tc.name: devicemanager_registerdevice_005
 * @tc.desc: Verify the RegisterDevice function return success.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_registerdevice_005, TestSize.Level0)
{
    std::vector<int32_t> typeVect = {HDF_SUCCESS, HDF_SUCCESS, HDF_SUCCESS, HDF_SUCCESS};
    OHOS::sptr<V1_0::MockIDevice> device = OHOS::sptr<V1_0::MockIDevice>(new (std::nothrow) V1_0::MockIDevice());
    EXPECT_NE(device.GetRefPtr(), nullptr);

    std::string deviceName = "MockDeviceA";
    std::string vendorName = "MockVendorA";
    MockInit(device, typeVect, deviceName, vendorName);

    std::function<std::shared_ptr<HDIDevice>()> creator =
        [&device]()->std::shared_ptr<HDIDevice> {return std::make_shared<HDIDevice>(device);};
    auto& deviceManager = DeviceManager::GetInstance();
    OH_NN_ReturnCode result = deviceManager.RegisterDevice(creator);
    EXPECT_EQ(OH_NN_SUCCESS, result);

    std::vector<size_t> idVect = deviceManager.GetAllDeviceId();
    EXPECT_NE((size_t)0, idVect.size());

    const size_t expectDeviceId {std::hash<std::string>{}("MockDeviceA_MockVendorA")};
    EXPECT_EQ(expectDeviceId, idVect[0]);

    const std::string expectDeviceName = "MockDeviceA_MockVendorA";
    const std::string retDeviceName = deviceManager.GetDeviceName(idVect[0]);
    EXPECT_EQ(retDeviceName, expectDeviceName);
}

/**
 * @tc.name: devicemanager_getdevicename_001
 * @tc.desc: Verify the GetDevice function return empty string in case of deviceid invalid.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceManagerTest, devicemanager_getdevicename_001, TestSize.Level0)
{
    auto &deviceManager = DeviceManager::GetInstance();
    const size_t deviceId = 1;
    std::string result = deviceManager.GetDeviceName(deviceId);
    EXPECT_EQ("", result);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
