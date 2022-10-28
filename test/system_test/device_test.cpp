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
#include <iostream>
#include <functional>
#include <string>

#include "interfaces/kits/c/neural_network_runtime.h"

using namespace testing;
using namespace testing::ext;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace SystemTest {
class DeviceTest : public testing::Test {
public:
    void SetUp() {}
    void TearDown() {}

public:
    std::string m_deviceName {"RK3568-CPU_Rockchip"};
    size_t m_deviceId {std::hash<std::string>{}("RK3568-CPU_Rockchip")};
    OH_NN_DeviceType m_deviceType {OH_NN_CPU};
};

/*
 * @tc.name: device_001
 * @tc.desc: Get all devices id successfully.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_001, testing::ext::TestSize.Level1)
{
    const size_t* allDeviceIds = nullptr;
    uint32_t count {0};

    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(&allDeviceIds, &count);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    uint32_t expectCount = 1;
    EXPECT_EQ(expectCount, count);
    EXPECT_EQ(m_deviceId, *allDeviceIds);
}

/*
 * @tc.name: device_002
 * @tc.desc: Get all devices id with nullptr deviceId parameter.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_002, testing::ext::TestSize.Level1)
{
    uint32_t count {0};

    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(nullptr, &count);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_003
 * @tc.desc: Get all devices id with nullptr count parameter.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_003, testing::ext::TestSize.Level1)
{
    const size_t* allDeviceIds = nullptr;

    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(&allDeviceIds, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_004
 * @tc.desc: Get all devices id with not nullptr deviceId pointer.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_004, testing::ext::TestSize.Level1)
{
    const size_t allDeviceIds = 0;
    const size_t* pAllDeviceIds = &allDeviceIds;
    uint32_t count {0};

    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(&pAllDeviceIds, &count);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_005
 * @tc.desc: Get device name successfully.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_005, testing::ext::TestSize.Level1)
{
    const char* name = nullptr;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(m_deviceId, &name);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
    std::string sName(name);
    EXPECT_EQ(m_deviceName, sName);
}

/*
 * @tc.name: device_006
 * @tc.desc: Get device name with invalid deviceId.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_006, testing::ext::TestSize.Level1)
{
    const size_t deviceId = 0;
    const char* name = nullptr;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceId, &name);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: device_007
 * @tc.desc: Get device name without nullptr name pointer.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_007, testing::ext::TestSize.Level1)
{
    const size_t deviceId = 0;
    const char* name = "name";
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceId, &name);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_008
 * @tc.desc: Get device name with nullptr name parameter.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_008, testing::ext::TestSize.Level1)
{
    const size_t deviceId = 0;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceId, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_009
 * @tc.desc: Get device type successfully.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_009, testing::ext::TestSize.Level1)
{
    OH_NN_DeviceType type {OH_NN_OTHERS};
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(m_deviceId, &type);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
    EXPECT_EQ(m_deviceType, type);
}

/*
 * @tc.name: device_010
 * @tc.desc: Get device type with invalid deviceId.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_010, testing::ext::TestSize.Level1)
{
    const size_t deviceId = 0;
    OH_NN_DeviceType type {OH_NN_OTHERS};
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceId, &type);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_011
 * @tc.desc: Get device type with nullptr type.
 * @tc.type: FUNC
 */
HWTEST_F(DeviceTest, device_011, testing::ext::TestSize.Level1)
{
    const size_t deviceId = 0;
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceId, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}
} // namespace SystemTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS