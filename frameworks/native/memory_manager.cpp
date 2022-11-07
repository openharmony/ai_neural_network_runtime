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


#include "memory_manager.h"

#include <sys/mman.h>
#include <unistd.h>

#include "cpp_type.h"
#include "common/log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
void* MemoryManager::MapMemory(int fd, size_t length)
{
    if (fd < 0) {
        LOGE("Invalid fd, fd must greater than 0.");
        return nullptr;
    }

    if (length <= 0 || length > ALLOCATE_BUFFER_LIMIT) {
        LOGE("Invalid buffer size, it must greater than 0 and less than 1Gb. length=%zu", length);
        return nullptr;
    }

    void* addr = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (addr == MAP_FAILED) {
        LOGE("Map fd to address failed.");
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(m_mtx);
    Memory memory {fd, addr, length};
    m_memorys.emplace(addr, memory);
    return addr;
}

OH_NN_ReturnCode MemoryManager::UnMapMemory(const void* buffer)
{
    if (buffer == nullptr) {
        LOGE("Buffer is nullptr, no need to release.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto iter = m_memorys.find(buffer);
    if (iter == m_memorys.end()) {
        LOGE("This buffer is not found, cannot release.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto& memory = m_memorys[buffer];
    auto unmapResult = munmap(const_cast<void*>(memory.data), memory.length);
    if (unmapResult != 0) {
        LOGE("Unmap memory failed. Please try again.");
        return OH_NN_MEMORY_ERROR;
    }
    memory.data = nullptr;

    if (close(memory.fd) != 0) {
        LOGE("Close memory fd failed. fd=%d", memory.fd);
        return OH_NN_MEMORY_ERROR;
    }

    std::lock_guard<std::mutex> lock(m_mtx);
    m_memorys.erase(iter);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode MemoryManager::GetMemory(const void* buffer, Memory& memory) const
{
    if (buffer == nullptr) {
        LOGE("Memory is nullptr.");
        return OH_NN_NULL_PTR;
    }

    auto iter = m_memorys.find(buffer);
    if (iter == m_memorys.end()) {
        LOGE("Memory is not found.");
        return OH_NN_INVALID_PARAMETER;
    }

    memory.fd = iter->second.fd;
    memory.data = buffer;
    memory.length = iter->second.length;

    return OH_NN_SUCCESS;
}
} // NeuralNetworkRuntime
} // OHOS