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

#include <hdf_base.h>
#include <refbase.h>
#include <gtest/gtest.h>

#include "common/log.h"
#include "device_registrar.h"
#include "hdi_device_v2_0.h"
#include "device_manager.h"
#include "test/unittest/common/v2_0/mock_idevice.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
constexpr uint32_t INNRT_DEVICE_MAJOR_VERSION = 2;
constexpr uint32_t INNRT_DEVICE_MINOR_VERSION = 0;

class IRegisterDevice : public HDI::HdiBase {
public:
    DECLARE_HDI_DESCRIPTOR(u"ohos.hdi.nnrt.v2_0.IRegisterDevice");

    virtual ~IRegisterDevice() = default;

    static sptr<IRegisterDevice> Get(bool isStub = false);
    static sptr<IRegisterDevice> Get(const std::string& serviceName, bool isStub = false);

    virtual int32_t GetDeviceName(std::string& name) = 0;

    virtual int32_t GetVendorName(std::string& name) = 0;

    virtual int32_t GetDeviceType(V2_0::DeviceType& deviceType) = 0;

    virtual int32_t GetDeviceStatus(V2_0::DeviceStatus& status) = 0;

    virtual int32_t GetSupportedOperation(const V2_0::Model& model, std::vector<bool>& ops) = 0;

    virtual int32_t IsFloat16PrecisionSupported(bool& isSupported) = 0;

    virtual int32_t IsPerformanceModeSupported(bool& isSupported) = 0;

    virtual int32_t IsPrioritySupported(bool& isSupported) = 0;

    virtual int32_t IsDynamicInputSupported(bool& isSupported) = 0;

    virtual int32_t PrepareModel(const V2_0::Model& model, const V2_0::ModelConfig& config,
        sptr<V2_0::IPreparedModel>& preparedModel) = 0;

    virtual int32_t IsModelCacheSupported(bool& isSupported) = 0;

    virtual int32_t PrepareModelFromModelCache(const std::vector<V2_0::SharedBuffer>& modelCache,
        const V2_0::ModelConfig& config, sptr<V2_0::IPreparedModel>& preparedModel) = 0;

    virtual int32_t AllocateBuffer(uint32_t length, V2_0::SharedBuffer& buffer) = 0;

    virtual int32_t ReleaseBuffer(const V2_0::SharedBuffer& buffer) = 0;

    virtual int32_t GetVersion(uint32_t& majorVer, uint32_t& minorVer)
    {
        majorVer = INNRT_DEVICE_MAJOR_VERSION;
        minorVer = INNRT_DEVICE_MINOR_VERSION;
        return HDF_SUCCESS;
    }
};

class SimulationDevice : public Device {
public:
    explicit SimulationDevice(OHOS::sptr<IRegisterDevice> device) {};

    OH_NN_ReturnCode GetDeviceName(std::string& name) override
    {
        name = "MockIDeviceA";
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode GetVendorName(std::string& name) override
    {
        name = "MockVendorA";
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode GetVersion(std::string& version) override
    {
        version = "MockVersionA";
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode GetDeviceType(OH_NN_DeviceType& deviceType) override
    {
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode GetDeviceStatus(DeviceStatus& status) override
    {
        status = DeviceStatus::AVAILABLE;
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
        std::vector<bool>& ops) override
    {
        return OH_NN_SUCCESS;
    };

    OH_NN_ReturnCode IsFloat16PrecisionSupported(bool& isSupported) override
    {
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode IsPerformanceModeSupported(bool& isSupported) override
    {
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode IsPrioritySupported(bool& isSupported) override
    {
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode IsDynamicInputSupported(bool& isSupported) override
    {
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode IsModelCacheSupported(bool& isSupported) override
    {
        return OH_NN_SUCCESS;
    };

    OH_NN_ReturnCode PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model, const ModelConfig& config,
        std::shared_ptr<PreparedModel>& preparedModel) override
    {
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode PrepareModelFromModelCache(const std::vector<ModelBuffer>& modelCache,
        const ModelConfig& config, std::shared_ptr<PreparedModel>& preparedModel) override
    {
        return OH_NN_SUCCESS;
    };
    OH_NN_ReturnCode PrepareOfflineModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
        const ModelConfig& config, std::shared_ptr<PreparedModel>& preparedModel) override
    {
        return OH_NN_SUCCESS;
    };

    void *AllocateBuffer(size_t length) override
    {
        return nullptr;
    };
    OH_NN_ReturnCode ReleaseBuffer(const void* buffer) override
    {
        return OH_NN_SUCCESS;
    };
};

class MockIDeviceImp : public IRegisterDevice {
public:
    MOCK_METHOD1(GetDeviceName, int32_t(std::string&));
    MOCK_METHOD1(GetVendorName, int32_t(std::string&));
    MOCK_METHOD1(GetDeviceType, int32_t(V2_0::DeviceType&));
    MOCK_METHOD1(GetDeviceStatus, int32_t(V2_0::DeviceStatus&));
    MOCK_METHOD2(GetSupportedOperation, int32_t(const V2_0::Model&, std::vector<bool>&));
    MOCK_METHOD1(IsFloat16PrecisionSupported, int32_t(bool&));
    MOCK_METHOD1(IsPerformanceModeSupported, int32_t(bool&));
    MOCK_METHOD1(IsPrioritySupported, int32_t(bool&));
    MOCK_METHOD1(IsDynamicInputSupported, int32_t(bool&));
    MOCK_METHOD3(PrepareModel,
        int32_t(const V2_0::Model&, const V2_0::ModelConfig&, OHOS::sptr<V2_0::IPreparedModel>&));
    MOCK_METHOD1(IsModelCacheSupported, int32_t(bool&));
    MOCK_METHOD3(PrepareModelFromModelCache, int32_t(const std::vector<V2_0::SharedBuffer>&, const V2_0::ModelConfig&,
        OHOS::sptr<V2_0::IPreparedModel>&));
    MOCK_METHOD2(AllocateBuffer, int32_t(uint32_t, V2_0::SharedBuffer&));
    MOCK_METHOD1(ReleaseBuffer, int32_t(const V2_0::SharedBuffer&));
    MOCK_METHOD2(GetVersion, int32_t(uint32_t&, uint32_t&));
};

sptr<IRegisterDevice> IRegisterDevice::Get(bool isStub)
{
    return IRegisterDevice::Get("device_service", isStub);
}

sptr<IRegisterDevice> IRegisterDevice::Get(const std::string& serviceName, bool isStub)
{
    if (isStub) {
        return nullptr;
    }

    sptr<IRegisterDevice> mockIDevice = sptr<MockIDeviceImp>(new (std::nothrow) MockIDeviceImp());
    if (mockIDevice.GetRefPtr() == nullptr) {
        LOGE("Failed to new MockIDeviceImp object.");
        return nullptr;
    }

    std::string deviceName = "MockIDeviceA";
    EXPECT_CALL(*((MockIDeviceImp *)mockIDevice.GetRefPtr()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceName), ::testing::Return(HDF_SUCCESS)));

    std::string vendorName = "MockVendorA";
    EXPECT_CALL(*((MockIDeviceImp *)mockIDevice.GetRefPtr()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(vendorName), ::testing::Return(HDF_SUCCESS)));

    V2_0::DeviceStatus deviceStatus = V2_0::DeviceStatus::AVAILABLE;
    EXPECT_CALL(*((MockIDeviceImp *)mockIDevice.GetRefPtr()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(deviceStatus), ::testing::Return(HDF_SUCCESS)));
    return mockIDevice;
}

class DeviceRegistrarTest : public testing::Test {
public:
    DeviceRegistrarTest() = default;
    ~DeviceRegistrarTest() = default;
};

std::shared_ptr<Device> CreateDeviceObjectCallback()
{
    OHOS::sptr<IRegisterDevice> device = IRegisterDevice::Get(false);
    EXPECT_NE(device, nullptr);
    std::shared_ptr<Device> m_mockDevice = std::make_shared<SimulationDevice>(device);
    return m_mockDevice;
}

std::shared_ptr<Device> CreateNullObjectCallback()
{
    return nullptr;
}

/* *
 * @tc.name: devicemanager_getalldeviceid_001
 * @tc.desc: Verify the Constructor function register object success.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceRegistrarTest, deviceregistrar_constructor_001, TestSize.Level0)
{
    CreateDevice creator = CreateDeviceObjectCallback;
    std::unique_ptr<DeviceRegistrar> deviceRegister = std::make_unique<DeviceRegistrar>(creator);
    EXPECT_NE(deviceRegister, nullptr);
    auto &deviceManager = DeviceManager::GetInstance();
    std::vector<size_t> idVect = deviceManager.GetAllDeviceId();
    EXPECT_EQ((size_t)2, idVect.size());

    const std::string expectDeviceNameA = "MockDevice";
    std::string deviceName = "";
    std::shared_ptr<Device> retDevice = deviceManager.GetDevice(idVect[1]);
    retDevice->GetDeviceName(deviceName);
    EXPECT_EQ(deviceName, expectDeviceNameA);

    const std::string expectDeviceNameB = "MockDevice_MockVendor_v0_0";
    std::string queryDeviceName = deviceManager.GetDeviceName(idVect[1]);
    EXPECT_EQ(queryDeviceName, expectDeviceNameB);
}

/* *
 * @tc.name: devicemanager_getalldeviceid_002
 * @tc.desc: Verify the Constructor function register object creator return nullptr, used for branch coverage.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceRegistrarTest, deviceregistrar_constructor_002, TestSize.Level0)
{
    CreateDevice creator = CreateNullObjectCallback;
    std::unique_ptr<DeviceRegistrar> deviceRegister = std::make_unique<DeviceRegistrar>(creator);
    EXPECT_NE(deviceRegister, nullptr);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
