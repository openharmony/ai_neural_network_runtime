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

#include "common/log.h"
#include "backend_manager.h"
#include "nnbackend.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
constexpr int MAX_MODEL_SIZE = 200 * 1024 * 1024; // 200MB
constexpr int OCT_UNIT = 8;
constexpr int NULL_PTR_LENGTH = 0;
constexpr int NUMBER_CACHE_INFO_MEMBERS = 3;

// CRC16 Table is created based on the Polynomial of G(x) = x^16 + x^12 + x^15 + 1 and
// CRC register initialization value of "0" (0x0000)
static const unsigned short CRC16_TAB[256] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7, 0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad,
    0xe1ce, 0xf1ef, 0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6, 0x9339, 0x8318, 0xb37b, 0xa35a,
    0xd3bd, 0xc39c, 0xf3ff, 0xe3de, 0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485, 0xa56a, 0xb54b,
    0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d, 0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc, 0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861,
    0x2802, 0x3823, 0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b, 0x5af5, 0x4ad4, 0x7ab7, 0x6a96,
    0x1a71, 0x0a50, 0x3a33, 0x2a12, 0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a, 0x6ca6, 0x7c87,
    0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41, 0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70, 0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a,
    0x9f59, 0x8f78, 0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f, 0x1080, 0x00a1, 0x30c2, 0x20e3,
    0x5004, 0x4025, 0x7046, 0x6067, 0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e, 0x02b1, 0x1290,
    0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256, 0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405, 0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e,
    0xc71d, 0xd73c, 0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634, 0xd94c, 0xc96d, 0xf90e, 0xe92f,
    0x99c8, 0x89e9, 0xb98a, 0xa9ab, 0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3, 0xcb7d, 0xdb5c,
    0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a, 0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9, 0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83,
    0x1ce0, 0x0cc1, 0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8, 0x6e17, 0x7e36, 0x4e55, 0x5e74,
    0x2e93, 0x3eb2, 0x0ed1, 0x1ef0};

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
    if (access(cacheInfoPath.c_str(), 0) != 0) {
        LOGE("[NNCompiledCache] Restore failed, cacheInfoPath is not exist.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNCompiledCacheInfo cacheInfo;
    OH_NN_ReturnCode ret = CheckCacheInfo(cacheInfo, cacheInfoPath);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiledCache] Restore failed, error happened when calling CheckCacheInfo.");
        return ret;
    }

    if ((uint64_t)version > cacheInfo.version) {
        LOGE("[NNCompiledCache] Restore failed, version is not match. The current version is %{public}u, "
             "but the cache files version is %{public}zu.",
             version,
             (size_t)cacheInfo.version);
        return OH_NN_INVALID_PARAMETER;
    }

    if ((uint64_t)version < cacheInfo.version) {
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

        if (GetCrc16(static_cast<const unsigned char*>(modelBuffer.data), modelBuffer.length) !=
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
    std::unique_ptr<uint64_t[]> cacheInfo = std::make_unique<uint64_t[]>(cacheSize);
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

    for (size_t i = 0; i < cacheNumber; ++i) {
        std::string cacheModelFile = cacheDir + "/" + m_modelName + std::to_string(i) + ".nncache";
        std::ofstream cacheModelStream(cacheModelFile, std::ios::binary | std::ios::out | std::ios::trunc);
        if (cacheModelStream.fail()) {
            LOGE("[NNCompiledCache] GenerateCacheModel failed, model cache file is invalid.");
            return OH_NN_INVALID_PARAMETER;
        }

        uint64_t checkSum =
            static_cast<uint64_t>(GetCrc16(static_cast<const unsigned char*>(caches[i].data), caches[i].length));
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
    if (!infoCacheFile.read((char*)&(modelCacheInfo), charNumber)) {
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
    if (!infoCacheFile.read((char*)&modelCheckSum[0], modelCacheInfo.fileNumber * sizeof(uint64_t))) {
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

unsigned short NNCompiledCache::GetCrc16(const unsigned char* buffer, size_t length) const
{
    unsigned short crc16 = 0;
    for (size_t i = 0; i < length; ++i) {
        uint8_t tableIndex = ((crc16 >> OCT_UNIT) ^ *buffer++) & 0x00ff;
        crc16 = (crc16 << OCT_UNIT) ^ CRC16_TAB[tableIndex];
    }
    return crc16;
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
} // namespace NeuralNetworkRuntime
} // namespace OHOS
