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

#include "compilation.h"

#include <sys/stat.h>
#include <unistd.h>
#include <cstdio>
#include <sys/types.h>
#include <fstream>
#include <climits>

#include "common/utils.h"
#include "common/scoped_trace.h"
#include "validation.h"
#include "device_manager.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
constexpr int MAX_MODEL_SIZE = 200 * 1024 * 1024; // 200MB
constexpr int OCT_UNIT = 8;
constexpr int NULL_PTR_LENGTH = 0;
constexpr int NUMBER_CACHE_INFO_MEMBERS = 3;

// CRC16 Table is created based on the Polynomial of G(x) = x^16 + x^12 + x^15 + 1 and
// CRC register initialization value of "0" (0x0000)
static const unsigned short CRC16_TAB[256] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7,
    0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad, 0xe1ce, 0xf1ef,
    0x1231, 0x0210, 0x3273, 0x2252, 0x52b5, 0x4294, 0x72f7, 0x62d6,
    0x9339, 0x8318, 0xb37b, 0xa35a, 0xd3bd, 0xc39c, 0xf3ff, 0xe3de,
    0x2462, 0x3443, 0x0420, 0x1401, 0x64e6, 0x74c7, 0x44a4, 0x5485,
    0xa56a, 0xb54b, 0x8528, 0x9509, 0xe5ee, 0xf5cf, 0xc5ac, 0xd58d,
    0x3653, 0x2672, 0x1611, 0x0630, 0x76d7, 0x66f6, 0x5695, 0x46b4,
    0xb75b, 0xa77a, 0x9719, 0x8738, 0xf7df, 0xe7fe, 0xd79d, 0xc7bc,
    0x48c4, 0x58e5, 0x6886, 0x78a7, 0x0840, 0x1861, 0x2802, 0x3823,
    0xc9cc, 0xd9ed, 0xe98e, 0xf9af, 0x8948, 0x9969, 0xa90a, 0xb92b,
    0x5af5, 0x4ad4, 0x7ab7, 0x6a96, 0x1a71, 0x0a50, 0x3a33, 0x2a12,
    0xdbfd, 0xcbdc, 0xfbbf, 0xeb9e, 0x9b79, 0x8b58, 0xbb3b, 0xab1a,
    0x6ca6, 0x7c87, 0x4ce4, 0x5cc5, 0x2c22, 0x3c03, 0x0c60, 0x1c41,
    0xedae, 0xfd8f, 0xcdec, 0xddcd, 0xad2a, 0xbd0b, 0x8d68, 0x9d49,
    0x7e97, 0x6eb6, 0x5ed5, 0x4ef4, 0x3e13, 0x2e32, 0x1e51, 0x0e70,
    0xff9f, 0xefbe, 0xdfdd, 0xcffc, 0xbf1b, 0xaf3a, 0x9f59, 0x8f78,
    0x9188, 0x81a9, 0xb1ca, 0xa1eb, 0xd10c, 0xc12d, 0xf14e, 0xe16f,
    0x1080, 0x00a1, 0x30c2, 0x20e3, 0x5004, 0x4025, 0x7046, 0x6067,
    0x83b9, 0x9398, 0xa3fb, 0xb3da, 0xc33d, 0xd31c, 0xe37f, 0xf35e,
    0x02b1, 0x1290, 0x22f3, 0x32d2, 0x4235, 0x5214, 0x6277, 0x7256,
    0xb5ea, 0xa5cb, 0x95a8, 0x8589, 0xf56e, 0xe54f, 0xd52c, 0xc50d,
    0x34e2, 0x24c3, 0x14a0, 0x0481, 0x7466, 0x6447, 0x5424, 0x4405,
    0xa7db, 0xb7fa, 0x8799, 0x97b8, 0xe75f, 0xf77e, 0xc71d, 0xd73c,
    0x26d3, 0x36f2, 0x0691, 0x16b0, 0x6657, 0x7676, 0x4615, 0x5634,
    0xd94c, 0xc96d, 0xf90e, 0xe92f, 0x99c8, 0x89e9, 0xb98a, 0xa9ab,
    0x5844, 0x4865, 0x7806, 0x6827, 0x18c0, 0x08e1, 0x3882, 0x28a3,
    0xcb7d, 0xdb5c, 0xeb3f, 0xfb1e, 0x8bf9, 0x9bd8, 0xabbb, 0xbb9a,
    0x4a75, 0x5a54, 0x6a37, 0x7a16, 0x0af1, 0x1ad0, 0x2ab3, 0x3a92,
    0xfd2e, 0xed0f, 0xdd6c, 0xcd4d, 0xbdaa, 0xad8b, 0x9de8, 0x8dc9,
    0x7c26, 0x6c07, 0x5c64, 0x4c45, 0x3ca2, 0x2c83, 0x1ce0, 0x0cc1,
    0xef1f, 0xff3e, 0xcf5d, 0xdf7c, 0xaf9b, 0xbfba, 0x8fd9, 0x9ff8,
    0x6e17, 0x7e36, 0x4e55, 0x5e74, 0x2e93, 0x3eb2, 0x0ed1, 0x1ef0
};

Compilation::Compilation(const InnerModel* innerModel)
    : m_liteGraph(innerModel->GetLiteGraphs()),
    m_inputTensors(innerModel->GetInputTensors()),
    m_outputTensors(innerModel->GetOutputTensors()) {}

OH_NN_ReturnCode Compilation::SetDevice(size_t deviceId)
{
    if (m_isBuild) {
        LOGE("Cannot set deviceId after compilation finish.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    auto& deviceManager = DeviceManager::GetInstance();
    std::shared_ptr<Device> availableDevice = deviceManager.GetDevice(deviceId);
    if (availableDevice == nullptr) {
        LOGE("[Compilation] DeviceId does not exist, deviceId=%zu", deviceId);
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<bool> supportedList;
    OH_NN_ReturnCode ret = availableDevice->GetSupportedOperation(m_liteGraph, supportedList);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] SetDevice failed, error happened when getting supported operation.");
        return ret;
    }

    for (bool isSupport : supportedList) {
        if (!isSupport) {
            LOGE("[Compilation] SetDevice failed, current device not support the model, device id: %zu.", deviceId);
            return OH_NN_FAILED;
        }
    }

    bool supportDynamic;
    ret = availableDevice->IsDynamicInputSupported(supportDynamic);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] SetDevice failed, error happened when checking whether device supports dynamic input.");
        return ret;
    }

    if (IsDynamicShape() && (!supportDynamic)) {
        LOGE("[Compilation] SetDevice failed."
             "The device does not support dynamic shape inputs, but the model has dynamic inputs.");
        return OH_NN_FAILED;
    }

    m_device = availableDevice;
    m_deviceId = deviceId;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::SetCacheDir(const std::string& cacheModelPath, uint32_t version)
{
    if (m_isBuild) {
        LOGE("Cannot set cache after compilation finish.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_device == nullptr) {
        LOGE("The parameter of m_device is nullptr, please call SetDevice function before calling SetCacheDir.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    bool isSupportedCache {false};
    OH_NN_ReturnCode ret = m_device->IsModelCacheSupported(isSupportedCache);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Fail to query whether the device is available to save cache model.");
        return ret;
    }

    if (!isSupportedCache) {
        LOGE("[Compilation] The device is unavailable to save cache model.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    char realPathRes[PATH_MAX];
    const char* filePath = realpath(cacheModelPath.c_str(), realPathRes);
    if (filePath == nullptr) {
        LOGE("[Compilation] The cache model path is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }

    struct stat fileInfo;
    if (stat(filePath, &fileInfo) != 0) {
        LOGE("[Compilation] The cache directory does not exist or cannot be accessed.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!(fileInfo.st_mode & S_IFDIR)) {
        LOGE("[Compilation] The cache model path is not a directory.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_cachePath = (std::string)filePath + "/";
    m_version = version;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::SetPerformance(OH_NN_PerformanceMode performance)
{
    if (m_isBuild) {
        LOGE("[Compilation] Cannot set performance after compilation finish.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_device == nullptr) {
        LOGE("Cannot set performance before set device, please set device first");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    bool isSupportedPerformance {false};
    OH_NN_ReturnCode ret = m_device->IsPerformanceModeSupported(isSupportedPerformance);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Call device %zu failed.", m_deviceId);
        return ret;
    }

    if (!isSupportedPerformance) {
        LOGE("[Compilation] This device %zu is not support performance setting.", m_deviceId);
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!Validation::ValidatePerformanceMode(performance)) {
        LOGE("[Compilation] SetPerformance passed invalid performance=%d", performance);
        return OH_NN_INVALID_PARAMETER;
    }

    m_performance = performance;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::SetPriority(OH_NN_Priority priority)
{
    if (m_isBuild) {
        LOGE("[Compilation] Cannot set priority after compilation finish.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_device == nullptr) {
        LOGE("Cannot set priority before set device, please set device first");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    bool isSupportedPriority {false};
    OH_NN_ReturnCode ret = m_device->IsPrioritySupported(isSupportedPriority);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Call device %zu failed.", m_deviceId);
        return ret;
    }

    if (!isSupportedPriority) {
        LOGE("[Compilation] This device %zu is not support priority setting.", m_deviceId);
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!Validation::ValidatePriority(priority)) {
        LOGE("[Compilation] SetPriority passed invalid priority=%d", priority);
        return OH_NN_INVALID_PARAMETER;
    }

    m_priority = priority;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::SetEnableFp16(bool isFp16)
{
    if (m_isBuild) {
        LOGE("[Compilation] Cannot enable float16 after compilation finish.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_device == nullptr) {
        LOGE("Cannot set enable fp16 before set device, please set device first");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    bool isSupportedFp16 {false};
    OH_NN_ReturnCode ret = m_device->IsFloat16PrecisionSupported(isSupportedFp16);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Call device %zu failed.", m_deviceId);
        return ret;
    }

    if (!isSupportedFp16) {
        LOGE("[Compilation] This device %zu is not support float16 precision setting.", m_deviceId);
        return OH_NN_OPERATION_FORBIDDEN;
    }

    m_enableFp16 = isFp16;
    return OH_NN_SUCCESS;
}

unsigned short Compilation::GetCrc16(const unsigned char* buffer, size_t length) const
{
    unsigned short crc16 = 0;
    for (size_t i = 0; i < length; ++i) {
        uint8_t tableIndex = ((crc16 >> OCT_UNIT) ^ *buffer++) & 0x00ff;
        crc16 = (crc16 << OCT_UNIT) ^ CRC16_TAB[tableIndex];
    }
    return crc16;
}

OH_NN_ReturnCode Compilation::GenerateCacheInfo(uint32_t cacheSize, std::unique_ptr<uint64_t[]>& cacheInfo) const
{
    std::string cacheInfoPath = m_cachePath + "cache_info.nncache";
    std::ofstream cacheInfoStream(cacheInfoPath, std::ios::binary | std::ios::out | std::ios::trunc);
    if (cacheInfoStream.fail()) {
        LOGE("[Compilation] Model cache info file is invalid.");
        return OH_NN_INVALID_FILE;
    }

    if (!cacheInfoStream.write(reinterpret_cast<const char*>(cacheInfo.get()), cacheSize)) {
        LOGE("[Compilation] Fail to write cache info.");
        cacheInfoStream.close();
        return OH_NN_FAILED;
    }

    cacheInfoStream.close();
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::GenerateCacheModel(size_t cacheNumber, std::unique_ptr<uint64_t[]>& cacheInfo,
    std::vector<ModelBuffer> modelBuffer) const
{
    auto cacheInfoPtr = cacheInfo.get();
    *cacheInfoPtr++ = static_cast<uint64_t>(cacheNumber);
    *cacheInfoPtr++ = static_cast<uint64_t>(m_version);
    *cacheInfoPtr++ = static_cast<uint64_t>(m_deviceId);
    for (uint32_t i = 0; i < cacheNumber; ++i) {
        std::string cacheModelFile = m_cachePath + std::to_string(i) + ".nncache";
        std::ofstream cacheModelStream(cacheModelFile, std::ios::binary | std::ios::out | std::ios::trunc);
        if (cacheModelStream.fail()) {
            LOGE("[Compilation] Model cache file is invalid.");
            return OH_NN_INVALID_FILE;
        }

        uint64_t checkSum = static_cast<uint64_t>(GetCrc16(static_cast<const unsigned char*>(modelBuffer[i].buffer),
            modelBuffer[i].length));
        *cacheInfoPtr++ = checkSum;
        if (!cacheModelStream.write(static_cast<const char*>(modelBuffer[i].buffer), modelBuffer[i].length)) {
            LOGE("[Compilation] Fail to write cache model.");
            cacheModelStream.close();
            return OH_NN_FAILED;
        };

        cacheModelStream.close();
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::GenerateCacheFiles(const std::vector<ModelBuffer>& modelBuffer) const
{
    const size_t cacheNumber = modelBuffer.size();
    uint32_t cacheSize = NUMBER_CACHE_INFO_MEMBERS + cacheNumber;
    std::unique_ptr<uint64_t[]> cacheInfo = std::make_unique<uint64_t[]>(cacheSize);
    if (cacheInfo == nullptr) {
        LOGE("Fail to create cacheInfo instance.");
        return OH_NN_MEMORY_ERROR;
    }

    OH_NN_ReturnCode ret = GenerateCacheModel(cacheNumber, cacheInfo, modelBuffer);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    uint32_t infoCharNumber = cacheSize * sizeof(uint64_t);
    ret = GenerateCacheInfo(infoCharNumber, cacheInfo);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::GetCacheFileLength(std::ifstream& ifs, int& fsize) const
{
    ifs.seekg(0, std::ios::end);
    if (!ifs.good()) {
        LOGE("[Compilation] Fail to set the position of the next character to be extracted from the input stream.");
        return OH_NN_INVALID_FILE;
    }

    int handleValue = ifs.tellg();
    if (handleValue == -1) {
        LOGE("[Compilation] Unable to get position of the input stream.");
        return OH_NN_INVALID_FILE;
    }

    if ((handleValue > MAX_MODEL_SIZE) || (handleValue == NULL_PTR_LENGTH)) {
        LOGE("[Compilation] Unable to read huge or empty input stream, get cache file size=%d", handleValue);
        return OH_NN_INVALID_FILE;
    }

    fsize = handleValue;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::ReadCacheModelFile(const std::string& file, ModelBuffer& modelBuffer) const
{
    std::ifstream ifs(file.c_str(), std::ios::in | std::ios::binary);
    if (!ifs) {
        LOGE("[Compilation] Fail to open cache file.");
        return OH_NN_INVALID_FILE;
    }

    int fsize {-1};
    OH_NN_ReturnCode ret = GetCacheFileLength(ifs, fsize);
    if (ret != OH_NN_SUCCESS) {
        ifs.close();
        return ret;
    }

    ifs.seekg(0, std::ios::beg);
    if (!ifs.good()) {
        LOGE("[Compilation] Fail to set the position of the next character to be extracted"
            "from the cache model stream.");
        ifs.close();
        return OH_NN_FAILED;
    }

    char* ptr = static_cast<char*>(m_device->AllocateBuffer(fsize));
    if (ptr == nullptr) {
        LOGE("[Compilation] Fail to create file buffer.");
        ifs.close();
        return OH_NN_NULL_PTR;
    }

    ifs.read(ptr, fsize);
    if (!ifs.good()) {
        LOGE("[Compilation] Fail to read the characters from the cache model stream.");
        ifs.close();
        m_device->ReleaseBuffer(ptr);
        ptr = nullptr;
        return OH_NN_FAILED;
    }

    ifs.close();
    modelBuffer.buffer = ptr;
    modelBuffer.length = fsize;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::CheckCacheInfo(ModelCacheInfo& modelCacheInfo, const std::string& cacheInfoPath) const
{
    std::ifstream infoCacheFile(cacheInfoPath.c_str(), std::ios::in | std::ios::binary);
    if (!infoCacheFile) {
        LOGE("[Compilation] Openning cache info file failed.");
        return OH_NN_INVALID_FILE;
    }

    int charNumber = NUMBER_CACHE_INFO_MEMBERS * sizeof(uint64_t);
    if (!infoCacheFile.read((char*)&(modelCacheInfo), charNumber)) {
        LOGE("[Compilation] Fail to get the content of info cache file.");
        infoCacheFile.close();
        return OH_NN_INVALID_FILE;
    }

    // modelCacheInfo.deviceId type is int64_t,
    // it is transformed from size_t value, so the transform here will not truncate value.
    size_t deviceId = static_cast<size_t>(modelCacheInfo.deviceId);
    if (deviceId != m_deviceId) {
        LOGE("[Compilation] The deviceId=%zu in the cache files is different from current deviceId=%zu,"
            "please change the cache directory or current deviceId.", deviceId, m_deviceId);
        infoCacheFile.close();
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<uint64_t> modelCheckSum;
    modelCheckSum.resize(modelCacheInfo.fileNumber);
    modelCacheInfo.modelCheckSum.resize(modelCacheInfo.fileNumber);
    if (!infoCacheFile.read((char*)&modelCheckSum[0], modelCacheInfo.fileNumber * sizeof(uint64_t))) {
        LOGE("[Compilation] The info cache file has been changed.");
        infoCacheFile.close();
        return OH_NN_INVALID_FILE;
    }

    for (uint32_t i = 0; i < modelCacheInfo.fileNumber; ++i) {
        modelCacheInfo.modelCheckSum[i] = static_cast<unsigned short>(modelCheckSum[i]);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::RemoveCacheFiles(uint32_t fileNumber) const
{
    std::string cacheInfoPath = m_cachePath + "cache_info.nncache";
    if (remove(cacheInfoPath.c_str()) == -1) {
        LOGE("[Compilation] Fail to remove the file %s, please delete the file manually.", cacheInfoPath.c_str());
        return OH_NN_FAILED;
    }
    LOGI("[Compilation] Succeed to remove the file cache_info.nncach.");

    for (uint32_t i = 0; i < fileNumber; ++i) {
        std::string fileName = std::to_string(i) + ".nncache";
        std::string cacheModelPath = m_cachePath + fileName;
        if (access(cacheModelPath.c_str(), 0) != 0) {
            LOGW("[Compilation] The file %s does not exist, no need to delete the file.", cacheModelPath.c_str());
            continue;
        }

        if (remove(cacheModelPath.c_str()) == -1) {
            LOGE("[Compilation] Fail to remove the file %s, please delete the file manually.", cacheModelPath.c_str());
            return OH_NN_FAILED;
        }
        LOGI("[Compilation] Succeed to remove the file %s", cacheModelPath.c_str());
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::CheckCacheModel(const ModelCacheInfo& modelCacheInfo,
    std::vector<ModelBuffer>& modelBuffers) const
{
    for (uint32_t i = 0; i < modelCacheInfo.fileNumber; ++i) {
        std::string cacheModelPath = m_cachePath + std::to_string(i) + ".nncache";
        if (access(cacheModelPath.c_str(), 0) != 0) {
            LOGE("[Compilation] The cache model file %s does not exist.", cacheModelPath.c_str());
            return OH_NN_INVALID_FILE;
        }

        ModelBuffer modelBuffer;
        OH_NN_ReturnCode ret = ReadCacheModelFile(cacheModelPath, modelBuffer);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[Compilation] Read cache model file failed.");
            return ret;
        }

        if (GetCrc16(static_cast<const unsigned char*>(modelBuffer.buffer),
            modelBuffer.length) != modelCacheInfo.modelCheckSum[i]) {
            LOGE("[Compilation] The cache model file %s has been changed.", cacheModelPath.c_str());
            return OH_NN_INVALID_FILE;
        }

        modelBuffers.emplace_back(std::move(modelBuffer));
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::NormalBuild(std::shared_ptr<PreparedModel>& preparedModel)
{
    ModelConfig config {m_enableFp16, m_performance, m_priority};
    OH_NN_ReturnCode ret = m_device->PrepareModel(m_liteGraph, config, preparedModel);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Preparing model failed when normally building.");
        return ret;
    }

    m_executionPlan = CreateSharedPtr<ExecutionPlan>(preparedModel, m_device);
    if (m_executionPlan == nullptr) {
        LOGE("Fail to create ExecutionPlan instance.");
        return OH_NN_MEMORY_ERROR;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::GenCacheBuild(std::shared_ptr<PreparedModel>& preparedModel)
{
    OH_NN_ReturnCode ret = NormalBuild(preparedModel);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Preparing model failed when generating cache.");
        return ret;
    }

    std::vector<ModelBuffer> modelBuffers;
    ret = preparedModel->ExportModelCache(modelBuffers);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Export model cache failed.");
        return ret;
    }

    ret = GenerateCacheFiles(modelBuffers);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Generate cache files failed.");
        return ret;
    }

    LOGI("[Compilation] Export model cache successfully.");
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::ReGenCacheBuild(uint32_t fileNumber, std::shared_ptr<PreparedModel>& preparedModel)
{
    OH_NN_ReturnCode ret = RemoveCacheFiles(fileNumber);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    ret = GenCacheBuild(preparedModel);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Generating cache building failed.");
        return ret;
    }

    LOGI("[Compilation] Update model cache successfully.");
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::LoadCacheBuild(std::shared_ptr<PreparedModel>& preparedModel,
    const ModelCacheInfo& cacheInfo)
{
    std::vector<ModelBuffer> modelBuffers;
    OH_NN_ReturnCode ret = CheckCacheModel(cacheInfo, modelBuffers);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Checking cache model failed.");
        for (size_t i = 0; i < modelBuffers.size(); ++i) {
            m_device->ReleaseBuffer(modelBuffers[i].buffer);
            modelBuffers[i].buffer = nullptr;
            modelBuffers[i].length = 0;
        }
        return ret;
    }

    ModelConfig config {m_enableFp16, m_performance, m_priority};
    ret = m_device->PrepareModelFromModelCache(modelBuffers, config, preparedModel);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[Compilation] Preparing model from cache failed.");
        return ret;
    }

    LOGI("[Compilation] Load cache successfully.");

    m_executionPlan = CreateSharedPtr<ExecutionPlan>(preparedModel, m_device);
    if (m_executionPlan == nullptr) {
        LOGE("Fail to create ExecutionPlan instance.");
        return OH_NN_MEMORY_ERROR;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::InnerBuild()
{
    OH_NN_ReturnCode ret;
    std::shared_ptr<PreparedModel> preparedModel;
    if (m_cachePath.empty()) {
        ret = NormalBuild(preparedModel);
        if (ret != OH_NN_SUCCESS) {
            LOGE("Fail to normally build.");
            return ret;
        }

        m_isBuild = true;
        return OH_NN_SUCCESS;
    }

    std::string cacheInfoPath = m_cachePath + "cache_info.nncache";
    if (access(cacheInfoPath.c_str(), 0) != 0) {
        ret = GenCacheBuild(preparedModel);
        if (ret != OH_NN_SUCCESS) {
            LOGE("Fail to build in generating cache mode.");
            return ret;
        }

        m_isBuild = true;
        return OH_NN_SUCCESS;
    }

    ModelCacheInfo cacheInfo;
    ret = CheckCacheInfo(cacheInfo, cacheInfoPath);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    if (m_version > cacheInfo.version) {
        ret = ReGenCacheBuild(cacheInfo.fileNumber, preparedModel);
        if (ret != OH_NN_SUCCESS) {
            return ret;
        }

        m_isBuild = true;
        return OH_NN_SUCCESS;
    }

    if (m_version < cacheInfo.version) {
        LOGE("[Compilation] The current version is lower than the cache files, please set a higher version.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    ret = LoadCacheBuild(preparedModel, cacheInfo);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    m_isBuild = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Compilation::Build()
{
    NNRT_TRACE_NAME("Compilation");
    if (m_isBuild) {
        LOGE("[Compilation] Cannot enable float16 after compilation finish.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_device == nullptr) {
        LOGE("The parameter of m_device is nullptr, please call SetDevice function before build model.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode ret = InnerBuild();
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    return OH_NN_SUCCESS;
}

std::shared_ptr<ExecutionPlan> Compilation::GetExecutionPlan() const
{
    return m_executionPlan;
}

std::vector<std::shared_ptr<NNTensor>> Compilation::GetInputTensors() const
{
    return m_inputTensors;
}

std::vector<std::shared_ptr<NNTensor>> Compilation::GetOutputTensors() const
{
    return m_outputTensors;
}

bool Compilation::IsBuild() const
{
    return m_isBuild;
}

bool Compilation::IsDynamicShape() const
{
    for (size_t i = 0; i < m_inputTensors.size(); ++i) {
        if (m_inputTensors[i]->IsDynamicShape()) {
            return true;
        }
    }
    return false;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS