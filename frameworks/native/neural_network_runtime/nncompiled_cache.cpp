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

#include "nncompiled_cache.h"

#include <unistd.h>
#include <functional>
#include <memory>
#include <limits>

#include "common/utils.h"
#include "backend_manager.h"
#include "nnbackend.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
constexpr int MAX_MODEL_SIZE = 200 * 1024 * 1024; // 200MB
constexpr int NULL_PTR_LENGTH = 0;
constexpr int NUMBER_CACHE_INFO_MEMBERS = 3;
constexpr int HEX_UNIT = 16;
constexpr char ROOT_DIR_STR = '/';
constexpr char DOUBLE_SLASH_STR[] = "//";

OH_NN_ReturnCode NNCompiledCache::Save(const std::vector<OHOS::NeuralNetworkRuntime::Buffer>& caches,
                                       const std::string& cacheDir,
                                       uint32_t version)
{
    if (caches.empty()) {
        LOGE("[NNCompiledCache] Save failed, caches is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_device == nullptr) {
        LOGE("[NNCompiledCache] Save failed, m_device is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = GenerateCacheFiles(caches, cacheDir, version);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] Save failed, error happened when calling GenerateCacheFiles.");
        return ret;
    }

    LOGI("[NNCompiledCache] Save success. %zu caches are saved.", caches.size());
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::Restore(const std::string& cacheDir,
                                          uint32_t version,
                                          std::vector<OHOS::NeuralNetworkRuntime::Buffer>& caches)
{
    if (cacheDir.empty()) {
        LOGE("[NNCompiledCache] Restore failed, cacheDir is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!caches.empty()) {
        LOGE("[NNCompiledCache] Restore failed, caches is not empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_device == nullptr) {
        LOGE("[NNCompiledCache] Restore failed, m_device is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::string cacheInfoPath = cacheDir + "/" + m_modelName + "cache_info.nncache";
    char path[PATH_MAX];
    if (realpath(cacheInfoPath.c_str(), path) == nullptr) {
        LOGE("[NNCompiledCache] Restore failed, fail to get the real path of cacheInfoPath.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (access(cacheInfoPath.c_str(), F_OK) != 0) {
        LOGE("[NNCompiledCache] Restore failed, cacheInfoPath is not exist.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNCompiledCacheInfo cacheInfo;
    OH_NN_ReturnCode ret = CheckCacheInfo(cacheInfo, cacheInfoPath);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] Restore failed, error happened when calling CheckCacheInfo.");
        return ret;
    }

    if (static_cast<uint64_t>(version) > cacheInfo.version) {
        LOGE("[NNCompiledCache] Restore failed, version is not match. The current version is %{public}u, "
             "but the cache files version is %{public}zu.",
             version,
             static_cast<size_t>(cacheInfo.version));
        return OH_NN_INVALID_PARAMETER;
    }

    if (static_cast<uint64_t>(version) < cacheInfo.version) {
        LOGE("[NNCompiledCache] Restore failed, the current version is lower than the cache files, "
             "please set a higher version.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    for (uint32_t i = 0; i < cacheInfo.fileNumber; ++i) {
        std::string cacheModelPath = cacheDir + "/" + m_modelName + std::to_string(i) + ".nncache";
        if (access(cacheModelPath.c_str(), 0) != 0) {
            LOGE("[NNCompiledCache] Restore failed, %{public}s is not exist.", cacheModelPath.c_str());
            return OH_NN_INVALID_PARAMETER;
        }

        OHOS::NeuralNetworkRuntime::Buffer modelBuffer;
        ret = ReadCacheModelFile(cacheModelPath, modelBuffer);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiledCache] Restore failed, error happened when calling ReadCacheModelFile.");
            return ret;
        }

        if (GetCrc16(static_cast<char*>(modelBuffer.data), modelBuffer.length) !=
            cacheInfo.modelCheckSum[i]) {
            LOGE("[NNCompiledCache] Restore failed, the cache model file %{public}s has been changed.",
                 cacheModelPath.c_str());
            return OH_NN_INVALID_FILE;
        }

        caches.emplace_back(std::move(modelBuffer));
    }

    return ret;
}

OH_NN_ReturnCode NNCompiledCache::SetBackend(size_t backendID)
{
    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(backendID);
    if (backend == nullptr) {
        LOGE("[NNCompiledCache] SetBackend failed, backend with backendID %{public}zu is not exist.", backendID);
        return OH_NN_INVALID_PARAMETER;
    }

    std::shared_ptr<NNBackend> nnBackend = std::reinterpret_pointer_cast<NNBackend>(backend);
    m_device = nnBackend->GetDevice();
    if (m_device == nullptr) {
        LOGE("[NNCompiledCache] SetBackend failed, device with backendID %{public}zu is not exist.", backendID);
        return OH_NN_FAILED;
    }

    m_backendID = backendID;
    return OH_NN_SUCCESS;
}

void NNCompiledCache::SetModelName(const std::string& modelName)
{
    m_modelName = modelName;
}

OH_NN_ReturnCode NNCompiledCache::GenerateCacheFiles(const std::vector<OHOS::NeuralNetworkRuntime::Buffer>& caches,
                                                     const std::string& cacheDir,
                                                     uint32_t version) const
{
    const size_t cacheNumber = caches.size();
    uint32_t cacheSize = NUMBER_CACHE_INFO_MEMBERS + cacheNumber;
    std::unique_ptr<uint64_t[]> cacheInfo = CreateUniquePtr<uint64_t[]>(cacheSize);
    if (cacheInfo == nullptr) {
        LOGE("[NNCompiledCache] GenerateCacheFiles failed, fail to create cacheInfo instance.");
        return OH_NN_MEMORY_ERROR;
    }

    OH_NN_ReturnCode ret = GenerateCacheModel(caches, cacheInfo, cacheDir, version);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] GenerateCacheFiles failed, error happened when calling GenerateCacheModel.");
        return ret;
    }

    uint32_t infoCharNumber = cacheSize * sizeof(uint64_t);
    ret = WriteCacheInfo(infoCharNumber, cacheInfo, cacheDir);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] GenerateCacheFiles failed, error happened when calling WriteCacheInfo.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::GenerateCacheModel(const std::vector<OHOS::NeuralNetworkRuntime::Buffer>& caches,
                                                     std::unique_ptr<uint64_t[]>& cacheInfo,
                                                     const std::string& cacheDir,
                                                     uint32_t version) const
{
    size_t cacheNumber = caches.size();

    auto cacheInfoPtr = cacheInfo.get();
    *cacheInfoPtr++ = static_cast<uint64_t>(cacheNumber);
    *cacheInfoPtr++ = static_cast<uint64_t>(version);
    *cacheInfoPtr++ = static_cast<uint64_t>(m_backendID); // Should call SetBackend first.

    // standardize the input dir
    OH_NN_ReturnCode ret = OH_NN_SUCCESS;
    char path[PATH_MAX];
    if (realpath(cacheDir.c_str(), path) == nullptr) {
        LOGE("[NNCompiledCache] GenerateCacheModel failed, fail to get the real path of cacheDir.");
        return OH_NN_INVALID_PARAMETER;
    }

    // verify the Standardized path available
    ret = VerifyCachePath(path);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] GenerateCacheModel failed, fail to verify the file path of cacheDir.");
        return ret;
    }

    for (size_t i = 0; i < cacheNumber; ++i) {
        std::string cacheModelFile = cacheDir + "/" + m_modelName + std::to_string(i) + ".nncache";
        std::ofstream cacheModelStream(cacheModelFile, std::ios::binary | std::ios::out | std::ios::trunc);
        if (cacheModelStream.fail()) {
            LOGE("[NNCompiledCache] GenerateCacheModel failed, model cache file is invalid.");
            return OH_NN_INVALID_PARAMETER;
        }

        uint64_t checkSum =
            static_cast<uint64_t>(GetCrc16(static_cast<char*>(caches[i].data), caches[i].length));
        *cacheInfoPtr++ = checkSum;
        if (!cacheModelStream.write(static_cast<const char*>(caches[i].data), caches[i].length)) {
            LOGE("[NNCompiledCache] GenerateCacheModel failed, fail to write cache model.");
            cacheModelStream.close();
            return OH_NN_SAVE_CACHE_EXCEPTION;
        };

        cacheModelStream.close();
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::WriteCacheInfo(uint32_t cacheSize,
                                                 std::unique_ptr<uint64_t[]>& cacheInfo,
                                                 const std::string& cacheDir) const
{
    // standardize the input dir
    char path[PATH_MAX];
    if (realpath(cacheDir.c_str(), path) == nullptr) {
        LOGE("[NNCompiledCache] WriteCacheInfo failed, fail to get the real path of cacheDir.");
        return OH_NN_INVALID_PARAMETER;
    }

    // verify the Standardized path available
    OH_NN_ReturnCode ret = VerifyCachePath(path);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] WriteCacheInfo failed, fail to verify the file path of cacheDir.");
        return ret;
    }

    std::string cacheInfoPath = cacheDir + "/" + m_modelName + "cache_info.nncache";
    std::ofstream cacheInfoStream(cacheInfoPath, std::ios::binary | std::ios::out | std::ios::trunc);
    if (cacheInfoStream.fail()) {
        LOGE("[NNCompiledCache] WriteCacheInfo failed, model cache info file is invalid.");
        return OH_NN_INVALID_FILE;
    }

    if (!cacheInfoStream.write(reinterpret_cast<const char*>(cacheInfo.get()), cacheSize)) {
        LOGE("[NNCompiledCache] WriteCacheInfo failed, fail to write cache info.");
        cacheInfoStream.close();
        return OH_NN_SAVE_CACHE_EXCEPTION;
    }

    cacheInfoStream.close();
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::CheckCacheInfo(NNCompiledCacheInfo& modelCacheInfo,
                                                 const std::string& cacheInfoPath) const
{
    // cacheInfoPath is validated outside.
    std::ifstream infoCacheFile(cacheInfoPath.c_str(), std::ios::in | std::ios::binary);
    if (!infoCacheFile) {
        LOGE("[NNCompiledCache] CheckCacheInfo failed, error happened when opening cache info file.");
        return OH_NN_INVALID_FILE;
    }

    int charNumber = NUMBER_CACHE_INFO_MEMBERS * sizeof(uint64_t);
    if (!infoCacheFile.read(reinterpret_cast<char*>(&(modelCacheInfo)), charNumber)) {
        LOGE("[NNCompiledCache] CheckCacheInfo failed, error happened when reading cache info file.");
        infoCacheFile.close();
        return OH_NN_INVALID_FILE;
    }

    // modelCacheInfo.deviceId type is int64_t,
    // it is transformed from size_t value, so the transform here will not truncate value.
    size_t deviceId = static_cast<size_t>(modelCacheInfo.deviceId);
    if (deviceId != m_backendID) {
        LOGE("[NNCompiledCache] CheckCacheInfo failed. The deviceId=%{public}zu in the cache files "
             "is different from current deviceId=%{public}zu,"
             "please change the cache directory or current deviceId.",
             deviceId,
             m_backendID);
        infoCacheFile.close();
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<uint64_t> modelCheckSum;
    modelCheckSum.resize(modelCacheInfo.fileNumber);
    modelCacheInfo.modelCheckSum.resize(modelCacheInfo.fileNumber);
    if (!infoCacheFile.read(reinterpret_cast<char*>(&modelCheckSum[0]),
        modelCacheInfo.fileNumber * sizeof(uint64_t))) {
        LOGE("[NNCompiledCache] CheckCacheInfo failed. The info cache file has been changed.");
        infoCacheFile.close();
        return OH_NN_INVALID_FILE;
    }

    for (uint32_t i = 0; i < modelCacheInfo.fileNumber; ++i) {
        modelCacheInfo.modelCheckSum[i] = static_cast<unsigned short>(modelCheckSum[i]);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::ReadCacheModelFile(const std::string& filePath,
                                                     OHOS::NeuralNetworkRuntime::Buffer& cache) const
{
    // filePath is validate in NNCompiledCache::Restore, no need to check again.
    std::ifstream ifs(filePath.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, file is invalid.");
        return OH_NN_INVALID_FILE;
    }

    int fsize{-1};
    OH_NN_ReturnCode ret = GetCacheFileLength(ifs, fsize);
    if (ret != OH_NN_SUCCESS) {
        ifs.close();
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, get file %{public}s length fialed.", filePath.c_str());
        return ret;
    }

    ifs.seekg(0, std::ios::beg);
    if (!ifs.good()) {
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, file is invalid.");
        ifs.close();
        return OH_NN_INVALID_FILE;
    }

    char* ptr = static_cast<char*>(m_device->AllocateBuffer(fsize));
    if (ptr == nullptr) {
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, failed to allocate memory.");
        ifs.close();
        return OH_NN_MEMORY_ERROR;
    }

    ifs.read(ptr, fsize);
    if (!ifs.good()) {
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, failed to read file.");
        ifs.close();
        m_device->ReleaseBuffer(ptr);
        ptr = nullptr;
        return OH_NN_INVALID_FILE;
    }

    ifs.close();
    cache.data = ptr;
    cache.length = static_cast<size_t>(fsize); // fsize should be non-negative, safe to cast.
    return OH_NN_SUCCESS;
}

unsigned short NNCompiledCache::GetCrc16(char* buffer, size_t length) const
{
    unsigned int sum = 0;
    while (length > 1) {
        sum += *(reinterpret_cast<unsigned short*>(buffer));
        length -= sizeof(unsigned short);
        buffer += sizeof(unsigned short);
    }

    if (length > 0) {
        sum += *(reinterpret_cast<unsigned char*>(buffer));
    }

    while (sum >> HEX_UNIT) {
        sum = (sum >> HEX_UNIT) + (sum & 0xffff);
    }

    return static_cast<unsigned short>(~sum);
}

OH_NN_ReturnCode NNCompiledCache::GetCacheFileLength(std::ifstream& ifs, int& fileSize) const
{
    ifs.seekg(0, std::ios::end);
    if (!ifs.good()) {
        LOGE("[NNCompiledCache] GetCacheFileLength failed, fail to set the position of the next character "
             "to be extracted from the input stream.");
        return OH_NN_FAILED;
    }

    int handleValue = ifs.tellg();
    if (handleValue == -1) {
        LOGE("[NNCompiledCache] GetCacheFileLength failed, fail to get position of the input stream.");
        return OH_NN_INVALID_FILE;
    }

    if ((handleValue > MAX_MODEL_SIZE) || (handleValue == NULL_PTR_LENGTH)) {
        LOGE("[NNCompiledCache] GetCacheFileLength failed, unable to read huge or empty input stream, "
             "get cache file size=%{public}d",
             handleValue);
        return OH_NN_INVALID_FILE;
    }

    fileSize = handleValue;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::VerifyCachePath(const std::string& cachePath) const
{
    // exception: input path is not start with '/'.
    if (cachePath.find(ROOT_DIR_STR) != size_t(0)) {
        LOGE("[NNCompiledCache] VerifyCachePath failed, input file dir=%{public}s is invalid, "
             "should start with '/'.",
             cachePath.c_str());
        return OH_NN_INVALID_FILE;
    }

    // exception: input path contains continuous double '/'.
    if (cachePath.find(DOUBLE_SLASH_STR) != std::string::npos) {
        LOGE("[NNCompiledCache] VerifyCachePath failed, input file dir=%{public}s is invalid, "
             "containing double '/'.",
             cachePath.c_str());
        return OH_NN_INVALID_FILE;
    }

    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS
