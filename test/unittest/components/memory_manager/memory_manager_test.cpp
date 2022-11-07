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

#include <cstdlib>

#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <gtest/gtest.h>

#include "frameworks/native/cpp_type.h"
#include "frameworks/native/memory_manager.h"
#include "test/unittest/common/file_utils.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class MemoryManagerTest : public testing::Test {
public:
    MemoryManagerTest() = default;
    ~MemoryManagerTest() = default;
};

/**
 * @tc.name: memorymanagertest_mapmemory_001
 * @tc.desc: Verify the MapMemory function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_mapmemory_001, TestSize.Level0)
{
    const auto& memoryManager = MemoryManager::GetInstance();
    int fd = -1;
    size_t length = 0;
    void* result = memoryManager->MapMemory(fd, length);
    EXPECT_EQ(nullptr, result);
}

/**
 * @tc.name: memorymanagertest_mapmemory_002
 * @tc.desc: Verify the MapMemory function return nullptr in case of length 0.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_mapmemory_002, TestSize.Level0)
{
    const auto& memoryManager = MemoryManager::GetInstance();
    int fd = 0;
    size_t length = 0;
    void* result = memoryManager->MapMemory(fd, length);
    EXPECT_EQ(nullptr, result);
}

/**
 * @tc.name: memorymanagertest_mapmemory_003
 * @tc.desc: Verify the MapMemory function return nullptr in case of fd 0.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_mapmemory_003, TestSize.Level0)
{
    const auto& memoryManager = MemoryManager::GetInstance();
    int fd = 0;
    size_t length = 1;
    void* result = memoryManager->MapMemory(fd, length);
    EXPECT_EQ(nullptr, result);
}

/**
 * @tc.name: memorymanagertest_mapmemory_004
 * @tc.desc: Verify the MapMemory function validate mapmemory content success.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_mapmemory_004, TestSize.Level0)
{
    std::string data = "ABCD";
    const size_t dataLength = 100;
    data.resize(dataLength, '*');

    std::string filename = "/data/log/memory-001.dat";
    FileUtils fileUtils(filename);
    fileUtils.WriteFile(data);

    int fd = open(filename.c_str(), O_RDWR);
    EXPECT_NE(-1, fd);

    size_t length = 4;
    const auto& memoryManager = MemoryManager::GetInstance();
    char* result = static_cast<char*>(memoryManager->MapMemory(fd, length));
    EXPECT_NE(nullptr, result);
    EXPECT_EQ('A', static_cast<char>(result[0]));
    EXPECT_EQ('B', static_cast<char>(result[1]));
    EXPECT_EQ('C', static_cast<char>(result[2]));
    EXPECT_EQ('D', static_cast<char>(result[3]));
    memoryManager->UnMapMemory(result);
    close(fd);
}

/**
 * @tc.name: memorymanagertest_unmapmemory_001
 * @tc.desc: Verify the UnMapMemory function validate behavior.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_unmapmemory_001, TestSize.Level0)
{
    const auto& memoryManager = MemoryManager::GetInstance();
    void* memory = nullptr;
    memoryManager->UnMapMemory(memory);
}

/**
 * @tc.name: memorymanagertest_unmapmemory_002
 * @tc.desc: Verify the UnMapMemory function validate behavior
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_unmapmemory_002, TestSize.Level0)
{
    const auto& memoryManager = MemoryManager::GetInstance();
    void* memory = malloc(10);
    memoryManager->UnMapMemory(memory);
    free(memory);
}

/**
 * @tc.name: memorymanagertest_unmapmemory_003
 * @tc.desc: Verify the UnMapMemory function pairwise behavior.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_unmapmemory_003, TestSize.Level0)
{
    std::string data = "ABCD";
    const size_t dataLength = 100;
    data.resize(dataLength, '/');

    std::string filename = "/data/log/memory-001.dat";
    FileUtils fileUtils(filename);
    fileUtils.WriteFile(data);

    int fd = 0;
    fd = open(filename.c_str(), O_RDWR);
    EXPECT_NE(-1, fd);

    size_t length = 10;
    const auto& memoryManager = MemoryManager::GetInstance();
    void* buffer = memoryManager->MapMemory(fd, length);
    memoryManager->UnMapMemory(buffer);
    close(fd);
}

/**
 * @tc.name: memorymanagertest_getmemory_001
 * @tc.desc: Verify the GetMemory function return nullptr.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_getmemory_001, TestSize.Level0)
{
    const auto& memoryManager = MemoryManager::GetInstance();
    void* buffer = nullptr;
    Memory memory;
    OH_NN_ReturnCode result = memoryManager->GetMemory(buffer, memory);
    EXPECT_EQ(OH_NN_NULL_PTR, result);
}

/**
 * @tc.name: memorymanagertest_getmemory_002
 * @tc.desc: Verify the GetMemory function return invalid parameter.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_getmemory_002, TestSize.Level0)
{
    const auto& memoryManager = MemoryManager::GetInstance();
    void* buffer = malloc(10);
    Memory memory;
    OH_NN_ReturnCode result = memoryManager->GetMemory(buffer, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
    free(buffer);
}

/**
 * @tc.name: memorymanagertest_getmemory_003
 * @tc.desc: Verify the GetMemory function validate memory content success.
 * @tc.type: FUNC
 */
HWTEST_F(MemoryManagerTest, memorymanagertest_getmemory_003, TestSize.Level0)
{
    std::string data = "ABCD";
    const size_t dataLength = 100;
    data.resize(dataLength, '%');

    std::string filename = "/data/log/memory-001.dat";
    FileUtils fileUtils(filename);
    fileUtils.WriteFile(data);

    int fd = 0;
    fd = open(filename.c_str(), O_RDWR);
    EXPECT_NE(-1, fd);

    size_t length = 4;
    const auto& memoryManager = MemoryManager::GetInstance();
    void* buffer = memoryManager->MapMemory(fd, length);
    close(fd);

    Memory memory;
    OH_NN_ReturnCode result = memoryManager->GetMemory(buffer, memory);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    EXPECT_NE(nullptr, memory.data);

    const char* tmpData = static_cast<const char*>(memory.data);
    EXPECT_EQ('A', static_cast<char>(tmpData[0]));
    EXPECT_EQ('B', static_cast<char>(tmpData[1]));
    EXPECT_EQ('C', static_cast<char>(tmpData[2]));
    EXPECT_EQ('D', static_cast<char>(tmpData[3]));
    memoryManager->UnMapMemory(buffer);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
