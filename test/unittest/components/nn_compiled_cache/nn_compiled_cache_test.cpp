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

/**
 * @tc.name: nncompiledcachetest_save_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_save_001, TestSize.Level0)
{
    NNCompiledCache nncompiledCache;
    std::vector<Buffer> caches;
    std::string m_cachePath = "a";
    uint32_t m_cacheVersion = 1;

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nncompiledCache.Save(caches, m_cachePath, m_cacheVersion));
}

/**
 * @tc.name: nncompiledcachetest_restore_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_restore_001, TestSize.Level0)
{
    NNCompiledCache nncompiledCache;
    std::string m_cachePath = "a";
    uint32_t m_cacheVersion = 1;
    std::vector<Buffer> caches;

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nncompiledCache.Restore(m_cachePath, m_cacheVersion, caches));
}

/**
 * @tc.name: nncompiledcachetest_setbackend_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_setbackend_001, TestSize.Level0)
{
    NNCompiledCache nncompiledCache;
    size_t backendID = 1;

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nncompiledCache.SetBackend(backendID));
}

/**
 * @tc.name: nncompiledcachetest_setmodelname_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompiledCacheTest, nncompiledcachetest_setmodelname_001, TestSize.Level0)
{
    NNCompiledCache nncompiledCache;
    std::string m_modelName;
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS