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
#include <cstdio>
#include <securec.h>

#include "utils.h"
#include "backend_manager.h"
#include "nnbackend.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
constexpr int32_t NULL_PTR_LENGTH = 0;
constexpr int32_t NUMBER_CACHE_INFO_MEMBERS = 3;
constexpr int32_t NUMBER_CACHE_INFO_EXTENSION_MEMBERS = 2;
constexpr int32_t HEX_UNIT = 16;
constexpr size_t MAX_CACHE_SIZE = 2 * 1024 * 1024; // 限制最大校验内存为2MB
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

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::CheckCache(const std::string& cacheDir,
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
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::Restore(const std::string& cacheDir,
                                          uint32_t version,
                                          std::vector<OHOS::NeuralNetworkRuntime::Buffer>& caches)
{
    OH_NN_ReturnCode ret = CheckCache(cacheDir, version, caches);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    std::string cacheInfoPath = cacheDir + "/" + m_modelName + "cache_info.nncache";
    char path[PATH_MAX];
    if (realpath(cacheInfoPath.c_str(), path) == nullptr) {
        LOGE("[NNCompiledCache] Restore failed, fail to get the real path of cacheInfoPath.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (access(path, F_OK) != 0) {
        LOGE("[NNCompiledCache] Restore failed, cacheInfoPath is not exist.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNCompiledCacheInfo cacheInfo;
    ret = CheckCacheInfo(cacheInfo, path);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] Restore failed, error happened when calling CheckCacheInfo.");
        return ret;
    }

    if (static_cast<int64_t>(version) > cacheInfo.version) {
        LOGE("[NNCompiledCache] Restore failed, version is not match.");
        return OH_NN_INVALID_FILE;
    }

    if (static_cast<int64_t>(version) < cacheInfo.version) {
        LOGE("[NNCompiledCache] Restore failed, the current version is lower than the cache files, "
             "please set a higher version.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    for (uint32_t i = 0; i < cacheInfo.fileNumber; ++i) {
        std::string cacheModelPath = cacheDir + "/" + m_modelName + std::to_string(i) + ".nncache";
        OHOS::NeuralNetworkRuntime::Buffer modelBuffer;
        ret = ReadCacheModelFile(cacheModelPath, modelBuffer);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiledCache] Restore failed, error happened when calling ReadCacheModelFile.");
            return OH_NN_INVALID_FILE;
        }

        if (GetCrc16(static_cast<char*>(modelBuffer.data), modelBuffer.length) !=
            cacheInfo.modelCheckSum[i]) {
            LOGE("[NNCompiledCache] Restore failed, the cache model file %{public}s has been changed.",
                 cacheModelPath.c_str());
            close(modelBuffer.fd);
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

void NNCompiledCache::SetIsExceedRamLimit(const bool isExceedRamLimit)
{
    m_isExceedRamLimit = isExceedRamLimit;
}

OH_NN_ReturnCode NNCompiledCache::GenerateCacheFiles(const std::vector<OHOS::NeuralNetworkRuntime::Buffer>& caches,
                                                     const std::string& cacheDir,
                                                     uint32_t version) const
{
    const size_t cacheNumber = caches.size();
    uint32_t cacheSize = NUMBER_CACHE_INFO_MEMBERS + cacheNumber + NUMBER_CACHE_INFO_EXTENSION_MEMBERS;
    nlohmann::json cacheInfo;

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
                                                     nlohmann::json& cacheInfo,
                                                     const std::string& cacheDir,
                                                     uint32_t version) const
{
    size_t cacheNumber = caches.size();
    if (cacheNumber == 0 || cacheNumber > NN_CACHE_FILE_NUMBER_MAX) {
        LOGE("[NNCompiledCache] Caches size is equal 0 or greater than 100.");
        return OH_NN_FAILED;
    }

    cacheInfo["data"]["fileNumber"] = static_cast<int64_t>(cacheNumber);
    cacheInfo["data"]["version"] = static_cast<int64_t>(version);
    cacheInfo["data"]["deviceId"] = static_cast<int64_t>(m_backendID); // Should call SetBackend first.

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

    std::string cachePath = path;
    for (size_t i = 0; i < cacheNumber; ++i) {
        std::string cacheModelFile = cachePath + "/" + m_modelName + std::to_string(i) + ".nncache";
        std::ofstream cacheModelStream(cacheModelFile, std::ios::binary | std::ios::out | std::ios::trunc);
        if (cacheModelStream.fail()) {
            LOGE("[NNCompiledCache] GenerateCacheModel failed, model cache file is invalid.");
            return OH_NN_INVALID_PARAMETER;
        }

        uint64_t checkSum =
            static_cast<int64_t>(GetCrc16(static_cast<char*>(caches[i].data), caches[i].length));
        cacheInfo["data"]["modelCheckSum"][i] = checkSum;
        if (!cacheModelStream.write(static_cast<const char*>(caches[i].data), caches[i].length)) {
            LOGE("[NNCompiledCache] GenerateCacheModel failed, fail to write cache model.");
            cacheModelStream.close();
            return OH_NN_SAVE_CACHE_EXCEPTION;
        };

        cacheModelStream.close();
    }

    int currentOpVersion = 0;
    ret = m_device->ReadOpVersion(currentOpVersion);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] GenerateCacheModel failed, fail to read op version.");
        return ret;
    }
    cacheInfo["data"]["opVersion"] = currentOpVersion;

    cacheInfo["data"]["isExceedRamLimit"] = m_isExceedRamLimit ? 1 : 0;

    const size_t dataLength = cacheInfo["data"].dump().length();
    char cacheInfoData[dataLength + 1];
    if (strncpy_s(cacheInfoData, dataLength+1, cacheInfo["data"].dump().c_str(), dataLength) != 0) {
        LOGE("ParseStr failed due to strncpy_s error");
        return OH_NN_INVALID_PARAMETER;
    }

    cacheInfo["CheckSum"] = static_cast<int64_t>(CacheInfoGetCrc16(cacheInfoData, dataLength));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::WriteCacheInfo(uint32_t cacheSize,
                                                 nlohmann::json& cacheInfo,
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

    std::string cachePath = path;
    std::string cacheInfoPath = cachePath + "/" + m_modelName + "cache_info.nncache";
    std::ofstream cacheInfoStream(cacheInfoPath, std::ios::binary | std::ios::out | std::ios::trunc);
    if (cacheInfoStream.fail()) {
        LOGE("[NNCompiledCache] WriteCacheInfo failed, model cache info file is invalid.");
        return OH_NN_INVALID_FILE;
    }

    cacheInfoStream << cacheInfo << std::endl;

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

    std::string content((std::istreambuf_iterator<char>(infoCacheFile)), std::istreambuf_iterator<char>());
    infoCacheFile.close();
    if (!nlohmann::json::accept(content)) {
        LOGE("[NNCompiledCache] CheckCacheInfo JSON parse error");
        return OH_NN_INVALID_FILE;
    }

    // Parse the JSON string
    nlohmann::json j = nlohmann::json::parse(content);
    // modelCacheInfo.deviceId type is int64_t,
    // it is transformed from size_t value, so the transform here will not truncate value.
    if (j.find("data") == j.end()) {
        LOGE("[NNCompiledCache] CheckCacheInfo read cache info file failed.");
        return OH_NN_INVALID_FILE;
    }

    if (j["data"].find("deviceId") == j["data"].end()) {
        LOGE("[NNCompiledCache] CheckCacheInfo read deviceId from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }
    modelCacheInfo.deviceId = j["data"]["deviceId"].get<int64_t>();

    if (j["data"].find("version") == j["data"].end()) {
        LOGE("[NNCompiledCache] CheckCacheInfo read version from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }
    modelCacheInfo.version = j["data"]["version"].get<int64_t>();

    size_t deviceId = static_cast<size_t>(modelCacheInfo.deviceId);
    if (deviceId != m_backendID) {
        LOGE("[NNCompiledCache] CheckCacheInfo failed. The deviceId in the cache files "
             "is different from current deviceId,"
             "please change the cache directory or current deviceId.");
        return OH_NN_INVALID_FILE;
    }

    if (j["data"].find("fileNumber") == j["data"].end()) {
        LOGE("[NNCompiledCache] CheckCacheInfo read fileNumber from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }
    modelCacheInfo.fileNumber = j["data"]["fileNumber"].get<int64_t>();

    return CheckCacheInfoExtension(modelCacheInfo, j);
}

OH_NN_ReturnCode NNCompiledCache::CheckCacheInfoExtension(NNCompiledCacheInfo& modelCacheInfo,
                                                          nlohmann::json& j) const
{
    const size_t dataLength = j["data"].dump().length();
    char jData[dataLength + 1];
    if (strncpy_s(jData, dataLength+1, j["data"].dump().c_str(), dataLength) != 0) {
        LOGE("[NNCompiledCache] ParseStr failed due to strncpy_s error.");
        return OH_NN_INVALID_FILE;
    }

    if (j.find("CheckSum") == j.end()) {
        LOGE("[NNCompiledCache] read CheckSum from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }

    if (static_cast<int64_t>(CacheInfoGetCrc16(jData, dataLength)) != j["CheckSum"].get<int64_t>()) {
        LOGE("[NNCompiledCache] cache_info CheckSum is not correct.");
        return OH_NN_INVALID_FILE;
    }
    std::vector<int64_t> modelCheckSum;
    modelCheckSum.resize(modelCacheInfo.fileNumber);
    modelCacheInfo.modelCheckSum.resize(modelCacheInfo.fileNumber);
    if (j["data"].find("modelCheckSum") == j["data"].end()) {
        LOGE("[NNCompiledCache] CheckCacheInfo read modelCheckSum from cache file failed.");
        return OH_NN_INVALID_FILE;
    }
    for (uint32_t i = 0; i < modelCacheInfo.fileNumber; ++i) {
        modelCheckSum[i] = static_cast<int64_t>(j["data"]["modelCheckSum"][i]);
    }

    for (uint32_t i = 0; i < modelCacheInfo.fileNumber; ++i) {
        modelCacheInfo.modelCheckSum[i] = static_cast<unsigned short>(modelCheckSum[i]);
    }

    if (j["data"].find("opVersion") == j["data"].end()) {
        LOGW("[NNCompiledCache] CheckCacheInfo read opVersion from cache info file failed.");
    } else {
        modelCacheInfo.opVersion = j["data"]["opVersion"].get<int64_t>();
    }

    if (j["data"].find("isExceedRamLimit") == j["data"].end()) {
        LOGE("[NNCompiledCache] CheckCacheInfo read isExceedRamLimit from cache info file failed.");
        return OH_NN_INVALID_FILE;
    } else {
        modelCacheInfo.isExceedRamLimit = j["data"]["isExceedRamLimit"].get<int64_t>();
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiledCache::ReadCacheModelFile(const std::string& filePath,
                                                     OHOS::NeuralNetworkRuntime::Buffer& cache)
{
    char path[PATH_MAX];
    if (realpath(filePath.c_str(), path) == nullptr) {
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, fail to get the real path of filePath.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (access(path, 0) != 0) {
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, %{public}s is not exist.", path);
        return OH_NN_INVALID_PARAMETER;
    }

    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, file fopen failed.");
        return OH_NN_INVALID_FILE;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        close(fd);
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, get file %{public}s state failed.", filePath.c_str());
        return OH_NN_MEMORY_ERROR;
    }

    off_t fsize = sb.st_size;

    void *ptr = mmap(nullptr, fsize, PROT_READ, MAP_SHARED, fd, 0);
    if (ptr == MAP_FAILED) {
        LOGE("[NNCompiledCache] ReadCacheModelFile failed, failed to mmap file.");
        close(fd);
        return OH_NN_INVALID_FILE;
    }

    cache.data = ptr;
    cache.length = static_cast<size_t>(fsize); // fsize should be non-negative, safe to cast.
    cache.fd = fd;
    return OH_NN_SUCCESS;
}

unsigned short NNCompiledCache::GetCrc16(char* buffer, size_t length) const
{
    unsigned int sum = 0;

    if (buffer == nullptr) {
        return static_cast<unsigned short>(~sum);
    }

    if (length < MAX_CACHE_SIZE) {
        while (length > 1) {
            sum += *(reinterpret_cast<unsigned short*>(buffer));
            length -= sizeof(unsigned short);
            buffer += sizeof(unsigned short);
        }
    } else {
        size_t step = length / MAX_CACHE_SIZE;
        while (length > sizeof(unsigned short) * step + 1) {
            sum += *(reinterpret_cast<unsigned short*>(buffer));
            length -= step * sizeof(unsigned short);
            buffer += step * sizeof(unsigned short);
        }
    }

    if (length > 0) {
        buffer += length - 1;
        sum += *(reinterpret_cast<unsigned char*>(buffer));
    }

    while (sum >> HEX_UNIT) {
        sum = (sum >> HEX_UNIT) + (sum & 0xffff);
    }

    return static_cast<unsigned short>(~sum);
}

OH_NN_ReturnCode NNCompiledCache::GetCacheFileLength(FILE* pFile, long& fileSize) const
{
    int ret = fseek(pFile, 0L, SEEK_END);
    if (ret != 0) {
        LOGE("[NNCompiledCache] GetCacheFileLength failed, fail to set the position of the next character "
             "to be extracted from the input stream.");
        return OH_NN_FAILED;
    }

    long handleValue = ftell(pFile);
    if (handleValue == -1) {
        LOGE("[NNCompiledCache] GetCacheFileLength failed, fail to get position of the input stream.");
        return OH_NN_INVALID_FILE;
    }

    if (handleValue == NULL_PTR_LENGTH) {
        LOGE("[NNCompiledCache] GetCacheFileLength failed, unable to read huge or empty input stream, "
             "get cache file size=%{public}ld",
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

void NNCompiledCache::ReleaseCacheBuffer(std::vector<Buffer>& buffers)
{
    for (auto buffer : buffers) {
        munmap(buffer.data, buffer.length);
        close(buffer.fd);
    }
    buffers.clear();
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS
