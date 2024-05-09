/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#ifndef NEURAL_NETWORK_BACKEND_NNCOMPILED_CACHE_H
#define NEURAL_NETWORK_BACKEND_NNCOMPILED_CACHE_H

#include <vector>
#include <fstream>
#include <memory>

#include "device.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime.h"
#include "tensor_desc.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
const uint32_t INVALID_CAHCE_VERSION = UINT32_MAX; // UINT32_MAX is reserved for invalid cache version.

struct NNCompiledCacheInfo {
    uint64_t fileNumber{0};
    uint64_t version{0};
    uint64_t deviceId{0};
    std::vector<unsigned short> modelCheckSum;
};

class NNCompiledCache {
public:
    NNCompiledCache() = default;
    ~NNCompiledCache() = default;

    OH_NN_ReturnCode Save(const std::vector<Buffer>& caches,
                          const std::string& cacheDir,
                          uint32_t version);
    OH_NN_ReturnCode Restore(const std::string& cacheDir,
                             uint32_t version,
                             std::vector<Buffer>& caches);

    OH_NN_ReturnCode SetBackend(size_t backendID);
    void SetModelName(const std::string& modelName);

private:
    OH_NN_ReturnCode GenerateCacheFiles(const std::vector<Buffer>& caches,
                                        const std::string& cacheDir,
                                        uint32_t version) const;
    OH_NN_ReturnCode GenerateCacheModel(const std::vector<Buffer>& caches,
                                        std::unique_ptr<uint64_t[]>& cacheInfo,
                                        const std::string& cacheDir,
                                        uint32_t version) const;
    OH_NN_ReturnCode WriteCacheInfo(uint32_t cacheSize,
                                    std::unique_ptr<uint64_t[]>& cacheInfo,
                                    const std::string& cacheDir) const;
    OH_NN_ReturnCode CheckCacheInfo(NNCompiledCacheInfo& modelCacheInfo, const std::string& cacheInfoPath) const;
    OH_NN_ReturnCode ReadCacheModelFile(const std::string& file, Buffer& cache) const;
    unsigned short GetCrc16(char* buffer, size_t length) const;
    OH_NN_ReturnCode GetCacheFileLength(std::ifstream& ifs, int& fileSize) const;
    OH_NN_ReturnCode VerifyCachePath(const std::string& cachePath) const;

private:
    size_t m_backendID {0};
    std::string m_modelName;
    std::shared_ptr<Device> m_device {nullptr};
};

} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_BACKEND_NNCOMPILED_CACHE_H
