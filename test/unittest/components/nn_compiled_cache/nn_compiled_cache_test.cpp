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

#include "nncompiled_cache.h"
#include "device.h"
#include "nnbackend.h"
#include "backend_manager.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"
#include "common/utils.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class NNCompiledCacheTest : public testing::Test {
public:
    NNCompiledCacheTest() = default;
    ~NNCompiledCacheTest() = default;
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
 * @tc.name: nncompiledcachetest_save_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_save_001, TestSize.Level0)
{
    LOGE("Save nncompiledcachetest_save_001");
    NNCompiledCache nncompiledCache;

    std::vector<Buffer> caches;
    std::string m_cachePath = "a";
    uint32_t m_cacheVersion = 1;

    OH_NN_ReturnCode ret = nncompiledCache.Save(caches, m_cachePath, m_cacheVersion);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nncompiledcachetest_save_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_save_002, TestSize.Level0)
{
    LOGE("Save nncompiledcachetest_save_002");
    NNCompiledCache nncompiledCache;

    Buffer buffer;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* data = dataArry;
    buffer.data = data;
    buffer.length = 1;
    std::vector<Buffer> caches;
    caches.emplace_back(buffer);
    std::string m_cachePath = "a";
    uint32_t m_cacheVersion = 1;

    OH_NN_ReturnCode ret = nncompiledCache.Save(caches, m_cachePath, m_cacheVersion);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

std::shared_ptr<Backend> Creator()
{
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
 * @tc.name: nncompiledcachetest_save_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_save_003, TestSize.Level0)
{
    LOGE("Save nncompiledcachetest_save_003");
    NNCompiledCache nncompiledCache;

    size_t backendID = 1;
    BackendManager& backendManager = BackendManager::GetInstance();

    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator;
    
    backendManager.RegisterBackend(backendName, creator);

    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    Buffer buffer;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* data = dataArry;
    buffer.data = data;
    buffer.length = 1;
    std::vector<Buffer> caches;
    caches.emplace_back(buffer);
    std::string m_cachePath = "a";
    uint32_t m_cacheVersion = 1;

    OH_NN_ReturnCode retSave = nncompiledCache.Save(caches, m_cachePath, m_cacheVersion);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, retSave);
}

std::shared_ptr<Backend> Creator2()
{
    size_t backendID = 2;
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

    char ptr = 'a';
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateBuffer(::testing::_))
        .WillRepeatedly(::testing::Return(&ptr));

    std::shared_ptr<Backend> backend = std::make_unique<NNBackend>(device, backendID);
    return backend;
}

/**
 * @tc.name: nncompiledcachetest_save_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_save_004, TestSize.Level0)
{
    LOGE("Save nncompiledcachetest_save_004");
    NNCompiledCache nncompiledCache;

    size_t backendID = 1;
    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    Buffer buffer;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* data = dataArry;
    buffer.data = data;
    buffer.length = 1;
    std::vector<Buffer> caches;
    caches.emplace_back(buffer);
    std::string m_cachePath = "/data/data";
    uint32_t m_cacheVersion = 1;

    OH_NN_ReturnCode retSave = nncompiledCache.Save(caches, m_cachePath, m_cacheVersion);
    EXPECT_EQ(OH_NN_SUCCESS, retSave);

    size_t backendID2 = 2;
    BackendManager& backendManager = BackendManager::GetInstance();

    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator2;
    
    backendManager.RegisterBackend(backendName, creator);

    OH_NN_ReturnCode retSetBackend = nncompiledCache.SetBackend(backendID2);
    EXPECT_EQ(OH_NN_SUCCESS, retSetBackend);

    std::string m_modelName = "test";
    nncompiledCache.SetModelName(m_modelName);

    OH_NN_ReturnCode retSave2 = nncompiledCache.Save(caches, m_cachePath, m_cacheVersion);
    EXPECT_EQ(OH_NN_SUCCESS, retSave2);
}

/**
 * @tc.name: nncompiledcachetest_restore_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_001, TestSize.Level0)
{
    LOGE("Restore nncompiledcachetest_restore_001");
    NNCompiledCache nncompiledCache;

    std::string m_cachePath = "a";
    uint32_t m_cacheVersion = 1;
    std::vector<Buffer> caches;

    OH_NN_ReturnCode ret = nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nncompiledcachetest_restore_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_002, TestSize.Level0)
{
    LOGE("Restore nncompiledcachetest_restore_002");
    NNCompiledCache nncompiledCache;

    std::string m_cachePath;
    uint32_t m_cacheVersion = 1;
    std::vector<Buffer> caches;

    OH_NN_ReturnCode ret = nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nncompiledcachetest_restore_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_003, TestSize.Level0)
{
    LOGE("Restore nncompiledcachetest_restore_003");
    NNCompiledCache nncompiledCache;

    std::string m_cachePath = "a";
    uint32_t m_cacheVersion = 1;
    Buffer buffer;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* data = dataArry;
    buffer.data = data;
    buffer.length = 1;
    std::vector<Buffer> caches;
    caches.emplace_back(buffer);

    OH_NN_ReturnCode ret = nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nncompiledcachetest_restore_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_004, TestSize.Level0)
{
    LOGE("Restore nncompiledcachetest_restore_004");
    NNCompiledCache nncompiledCache;

    size_t backendID = 1;
    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    std::string m_cachePath = "a";
    uint32_t m_cacheVersion = 1;
    std::vector<Buffer> caches;

    OH_NN_ReturnCode retRestore = nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, retRestore);
}

/**
 * @tc.name: nncompiledcachetest_restore_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_005, TestSize.Level0)
{
    LOGE("Restore nncompiledcachetest_restore_005");
    NNCompiledCache nncompiledCache;

    size_t backendID = 1;
    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    std::string m_modelName = "test";
    nncompiledCache.SetModelName(m_modelName);

    std::string m_cachePath = "/data";
    uint32_t m_cacheVersion = 1;
    std::vector<Buffer> caches;

    OH_NN_ReturnCode retRestore = nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, retRestore);
}

/**
 * @tc.name: nncompiledcachetest_restore_006
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_006, TestSize.Level0)
{
    LOGE("Restore nncompiledcachetest_restore_006");
    NNCompiledCache nncompiledCache;

    size_t backendID = 1;
    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    std::string m_cachePath = "/data/data";
    uint32_t m_cacheVersion = 1;
    std::vector<Buffer> caches;

    OH_NN_ReturnCode retRestore = nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, retRestore);
}

/**
 * @tc.name: nncompiledcachetest_restore_007
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_007, TestSize.Level0)
{
    LOGE("Restore nncompiledcachetest_restore_007");
    NNCompiledCache nncompiledCache;

    size_t backendID = 1;
    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    std::string m_cachePath = "/data/data";
    uint32_t m_cacheVersion = 1;
    std::vector<Buffer> caches;

    OH_NN_ReturnCode retRestore = nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, retRestore);
}

/**
 * @tc.name: nncompiledcachetest_restore_008
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_008, TestSize.Level0)
{
    LOGE("Restore nncompiledcachetest_restore_008");
    NNCompiledCache nncompiledCache;

    size_t backendID = 2;
    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    std::string m_cachePath = "/data/data";
    uint32_t m_cacheVersion = 1;
    std::vector<Buffer> caches;

    OH_NN_ReturnCode retRestore = nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, retRestore);
}

/**
 * @tc.name: nncompiledcachetest_setbackend_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_setbackend_001, TestSize.Level0)
{
    LOGE("SetBackend nncompiledcachetest_setbackend_001");
    NNCompiledCache nncompiledCache;

    size_t backendID = 3;

    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nncompiledcachetest_setbackend_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_setbackend_002, TestSize.Level0)
{
    LOGE("SetBackend nncompiledcachetest_setbackend_002");
    NNCompiledCache nncompiledCache;

    size_t backendID = 1;
    BackendManager& backendManager = BackendManager::GetInstance();

    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator;
    
    backendManager.RegisterBackend(backendName, creator);

    OH_NN_ReturnCode ret = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: nncompiledcachetest_setmodelname_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_setmodelname_001, TestSize.Level0)
{
    LOGE("SetModelName nncompiledcachetest_setmodelname_001");
    NNCompiledCache nncompiledCache;
    std::string m_modelName;
    nncompiledCache.SetModelName(m_modelName);
}

/**
 * @tc.name: nncompiledcachetest_writecacheinfo_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_writecacheinfo_001, TestSize.Level0)
{
    LOGE("WriteCacheInfo nncompiledcachetest_writecacheinfo_001");
    NNCompiledCache nncompiledCache;

    uint32_t cacheSize = 1;
    std::unique_ptr<int64_t[]> cacheInfo = std::make_unique<int64_t[]>(cacheSize);
    std::string cacheDir = "mock";

    OH_NN_ReturnCode ret = nncompiledCache.WriteCacheInfo(cacheSize, cacheInfo, cacheDir);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nncompiledcachetest_writecacheinfo_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_writecacheinfo_002, TestSize.Level0)
{
    LOGE("WriteCacheInfo nncompiledcachetest_writecacheinfo_002");
    NNCompiledCache nncompiledCache;

    uint32_t cacheSize = 1;
    std::unique_ptr<int64_t[]> cacheInfo = std::make_unique<int64_t[]>(cacheSize);
    std::string cacheDir = "/data/data";

    OH_NN_ReturnCode ret = nncompiledCache.WriteCacheInfo(cacheSize, cacheInfo, cacheDir);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: nncompiledcachetest_checkcacheinfo_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_checkcacheinfo_001, TestSize.Level0)
{
    LOGE("CheckCacheInfo nncompiledcachetest_checkcacheinfo_001");
    NNCompiledCache nncompiledCache;

    NNCompiledCacheInfo modelCacheInfo;
    std::string cacheInfoPath = "MOCK";

    OH_NN_ReturnCode ret = nncompiledCache.CheckCacheInfo(modelCacheInfo, cacheInfoPath);
    EXPECT_EQ(OH_NN_INVALID_FILE, ret);
}

/**
 * @tc.name: nncompiledcachetest_checkcacheinfo_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_checkcacheinfo_002, TestSize.Level0)
{
    LOGE("CheckCacheInfo nncompiledcachetest_checkcacheinfo_002");
    NNCompiledCache nncompiledCache;

    NNCompiledCacheInfo modelCacheInfo;
    std::string cacheInfoPath = "/data/data/0.nncache";

    OH_NN_ReturnCode ret = nncompiledCache.CheckCacheInfo(modelCacheInfo, cacheInfoPath);
    EXPECT_EQ(OH_NN_INVALID_FILE, ret);
}

/**
 * @tc.name: nncompiledcachetest_checkcacheinfo_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_checkcacheinfo_003, TestSize.Level0)
{
    LOGE("CheckCacheInfo nncompiledcachetest_checkcacheinfo_003");
    NNCompiledCache nncompiledCache;

    NNCompiledCacheInfo modelCacheInfo;
    std::string cacheInfoPath = "/data/data/testcache_info.nncache";

    OH_NN_ReturnCode ret = nncompiledCache.CheckCacheInfo(modelCacheInfo, cacheInfoPath);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: nncompiledcachetest_checkcacheinfo_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_checkcacheinfo_004, TestSize.Level0)
{
    LOGE("CheckCacheInfo nncompiledcachetest_checkcacheinfo_004");
    NNCompiledCache nncompiledCache;
    
    size_t backendID = 2;
    OH_NN_ReturnCode retSetBackend = nncompiledCache.SetBackend(backendID);
    EXPECT_EQ(OH_NN_SUCCESS, retSetBackend);

    std::string m_modelName = "test";
    nncompiledCache.SetModelName(m_modelName);

    NNCompiledCacheInfo modelCacheInfo;
    std::string cacheInfoPath = "/data/data/testcache_info.nncache";

    OH_NN_ReturnCode ret = nncompiledCache.CheckCacheInfo(modelCacheInfo, cacheInfoPath);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS