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

#ifndef NEURAL_NETWORK_RUNTIME_MEMORY_MANAGER_H
#define NEURAL_NETWORK_RUNTIME_MEMORY_MANAGER_H

#include <unordered_map>
#include <mutex>

#include "interfaces/kits/c/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
const int INVALID_FD = -1;

struct Memory {
    int fd;
    const void* data;
    size_t length;
};

class MemoryManager {
public:
    ~MemoryManager() = default;

    void* MapMemory(int fd, size_t length);
    OH_NN_ReturnCode UnMapMemory(const void* buffer);
    OH_NN_ReturnCode GetMemory(const void* buffer, Memory& memory) const;

    static MemoryManager* GetInstance()
    {
        static MemoryManager instance;
        return &instance;
    }

private:
    MemoryManager() {};
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;

private:
    // key: OH_NN_Memory, value: fd
    std::unordered_map<const void*, Memory> m_memorys;
    std::mutex m_mtx;
};
} // namespace NeuralNetworkRuntime
} // OHOS
#endif // NEURAL_NETWORK_RUNTIME_MEMORY_MANAGER_H