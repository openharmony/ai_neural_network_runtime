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

#include "neural_network_runtime/neural_network_core.h"

#include <string>
#include <securec.h>
#include <sys/stat.h>
#include <utility>
#include <unordered_map>
#include <future>
#include <thread>
#include <openssl/sha.h>

#include "log.h"
#include "executor.h"
#include "tensor.h"
#include "compilation.h"
#include "backend_manager.h"
#include "nnrt_client.h"

using namespace OHOS::NeuralNetworkRuntime;
#define NNRT_API __attribute__((visibility("default")))
constexpr size_t INPUT_OUTPUT_MAX_INDICES = 200;
constexpr size_t MODEL_MAX_LIMIT = 200 * 1024 * 1024; // 200MB
constexpr size_t CACHE_ID_CALCULATE_SIZE = 512 * 1024; // 0.5MB
constexpr size_t WIDTH = 4;
constexpr size_t TWO = 2;
constexpr size_t END = 1;
constexpr unsigned char MASK = 0x0F;

namespace {
std::string Sha256(const std::vector<void*>& dataList, const std::vector<size_t>& sizeList, bool isUpper)
{
    unsigned char hash[SHA256_DIGEST_LENGTH * TWO + END] = "";

    SHA256_CTX ctx;
    SHA256_Init(&ctx);
    for (size_t index = 0; index < dataList.size(); ++index) {
        SHA256_Update(&ctx, dataList[index], sizeList[index]);
    }
    SHA256_Final(&hash[SHA256_DIGEST_LENGTH], &ctx);

    // here we translate sha256 hash to hexadecimal.
    // each 8-bit char will be presented by two characters([0-9a-f])
    const char* hexCode = isUpper ? "0123456789ABCDEF" : "0123456789abcdef";
    for (int32_t i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        unsigned char value = hash[SHA256_DIGEST_LENGTH + i];
        // uint8_t is 2 digits in hexadecimal.
        hash[i * TWO] = hexCode[(value >> WIDTH) & MASK];
        hash[i * TWO + END] = hexCode[value & MASK];
    }
    hash[SHA256_DIGEST_LENGTH * TWO] = 0;
    return reinterpret_cast<char*>(hash);
}

std::string GetBufferId(const void* buffer, size_t size)
{
    std::vector<void*> dataList;
    std::vector<size_t> sizeList;

    // 模型buffer小于等于1MB，整个模型cache用于计算cache的ID
    if (size <= (CACHE_ID_CALCULATE_SIZE + CACHE_ID_CALCULATE_SIZE)) {
        dataList.emplace_back(const_cast<void*>(buffer));
        sizeList.emplace_back(size);
    } else { // 模型buffer大于1M，分别取模型buffer的前0.5MB 和 后0.5MB计算buffer的ID
        dataList.emplace_back(const_cast<void*>(buffer));
        sizeList.emplace_back(CACHE_ID_CALCULATE_SIZE);

        size_t offset = size - CACHE_ID_CALCULATE_SIZE;
        const char* convertBuffer = static_cast<const char*>(buffer);
        char* convertConstBuffer = const_cast<char*>(convertBuffer);
        dataList.emplace_back(reinterpret_cast<void*>(convertConstBuffer + offset));
        sizeList.emplace_back(CACHE_ID_CALCULATE_SIZE);
    }

    return Sha256(dataList, sizeList, false);
}

OH_NN_ReturnCode GetNnrtModelId(Compilation* compilationImpl)
{
    // 模型在线构图场景获取modelID
    if (compilationImpl->nnModel != nullptr) {
        compilationImpl->nnrtModelID = compilationImpl->compiler->GetOnlineModelID();
        return OH_NN_SUCCESS;
    }

    // omc路径加载场景获取modelID
    if (compilationImpl->offlineModelPath != nullptr) {
        std::string pathStr{compilationImpl->offlineModelPath};
        compilationImpl->nnrtModelID = std::hash<std::string>{}(pathStr);
        return OH_NN_SUCCESS;
    }

    // 模型缓存路径加载场景获取modelID
    if (compilationImpl->cachePath != nullptr) {
        struct stat buffer;
        if (stat(compilationImpl->cachePath, &buffer) != 0) {
            LOGE("GetModelId failed, cachePath is not exit or permission.");
            return OH_NN_INVALID_PARAMETER;
        }

        compilationImpl->nnrtModelID = compilationImpl->compiler->GetOnlineModelID();
        return OH_NN_SUCCESS;
    }

    // omc buffer加载场景获取modelID
    if ((compilationImpl->offlineModelBuffer.first != nullptr) &&
        (compilationImpl->offlineModelBuffer.second != size_t(0))) {
        std::string bufferSha = GetBufferId(compilationImpl->offlineModelBuffer.first,
            compilationImpl->offlineModelBuffer.second);
        compilationImpl->nnrtModelID = std::hash<std::string>{}(bufferSha);
        return OH_NN_SUCCESS;
    }

    // 模型缓存buffer场景获取modelID
    if ((compilationImpl->cacheBuffer.first != nullptr) &&
        (compilationImpl->cacheBuffer.second != size_t(0))) {
        std::string bufferSha = GetBufferId(compilationImpl->cacheBuffer.first,
            compilationImpl->cacheBuffer.second);
        compilationImpl->nnrtModelID = std::hash<std::string>{}(bufferSha);
        return OH_NN_SUCCESS;
    }

    LOGE("GetModelId failed, no available model to set modelId, please check.");
    return OH_NN_INVALID_PARAMETER;
}

OH_NN_ReturnCode IsCompilationAvaliable(Compilation* compilationImpl)
{
    if (compilationImpl == nullptr) {
        LOGE("IsCompilationAvaliable failed, compilation implementation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (((compilationImpl->nnModel != nullptr) && (compilationImpl->offlineModelPath != nullptr)) ||
        ((compilationImpl->nnModel != nullptr) &&
         ((compilationImpl->offlineModelBuffer.first != nullptr) ||
          (compilationImpl->offlineModelBuffer.second != static_cast<size_t>(0)))) ||
        ((compilationImpl->offlineModelPath != nullptr) &&
         ((compilationImpl->offlineModelBuffer.first != nullptr) ||
          (compilationImpl->offlineModelBuffer.second != static_cast<size_t>(0))))) {
        LOGE("IsCompilationAvaliable failed, find multi model to build compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (compilationImpl->compiler != nullptr) {
        LOGE("IsCompilationAvaliable failed, the compiler in compilation is not nullptr, "
             "please input a new compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode GetModelSize(const Compilation* compilation, size_t& modelSize)
{
    // 模型在线构图场景获取modelSize
    if (compilation->nnModel != nullptr) {
        modelSize = compilation->compiler->GetModelSize();
        return OH_NN_SUCCESS;
    }

    // omc路径加载场景获取modelSize
    if (compilation->offlineModelPath != nullptr) {
        modelSize = compilation->compiler->GetModelSize();
        return OH_NN_SUCCESS;
    }

    // 模型缓存路径加载场景获取modelSize
    if (compilation->cachePath != nullptr) {
        struct stat buffer;
        if (stat(compilation->cachePath, &buffer) != 0) {
            LOGE("CheckExceedRamLimit failed, cachePath is not exit or permission.");
            return OH_NN_INVALID_PARAMETER;
        }

        modelSize = compilation->compiler->GetModelSize();
        return OH_NN_SUCCESS;
    }

    // omc buffer加载场景获取modelSize
    if ((compilation->offlineModelBuffer.first != nullptr) &&
               (compilation->offlineModelBuffer.second != size_t(0))) {
        modelSize = compilation->offlineModelBuffer.second;
        return OH_NN_SUCCESS;
    }

    // 模型缓存buffer场景获取modelSize
    if ((compilation->cacheBuffer.first != nullptr) &&
        (compilation->cacheBuffer.second != size_t(0))) {
        modelSize = compilation->cacheBuffer.second;
        return OH_NN_SUCCESS;
    }

    LOGE("CheckExceedRamLimit failed, no available model to check.");
    return OH_NN_INVALID_PARAMETER;
}
}

NNRT_API OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t **allDevicesID, uint32_t *deviceCount)
{
    if (allDevicesID == nullptr) {
        LOGE("OH_NNDevice_GetAllDevicesID failed, passed nullptr to allDevicesID.");
        return OH_NN_INVALID_PARAMETER;
    }

    if ((*allDevicesID) != nullptr) {
        LOGE("OH_NNDevice_GetAllDevicesID failed, *allDevicesID should be nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (deviceCount == nullptr) {
        LOGE("OH_NNDevice_GetAllDevicesID failed, passed nullptr to deviceCount.");
        return OH_NN_INVALID_PARAMETER;
    }

    BackendManager& backendManager = BackendManager::GetInstance();
    const std::vector<size_t>& allDevices = backendManager.GetAllBackendsID();

    if (allDevices.empty()) {
        LOGW("OH_NNDevice_GetAllDevicesID got no device.");
        *allDevicesID = nullptr;
        *deviceCount = 0;
        return OH_NN_SUCCESS;
    }

    *allDevicesID = allDevices.data();
    // allDevices.size() will not exceed UINT32_MAX, it is safe to cast to uint32_t.
    *deviceCount = static_cast<uint32_t>(allDevices.size());

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNDevice_GetName(size_t deviceID, const char **name)
{
    if (name == nullptr) {
        LOGE("OH_NNDevice_GetName failed, passed nullptr to name.");
        return OH_NN_INVALID_PARAMETER;
    }

    if ((*name) != nullptr) {
        LOGE("OH_NNDevice_GetName failed, *name should be nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    BackendManager& backendManager = BackendManager::GetInstance();
    const std::string& backendName = backendManager.GetBackendName(deviceID);
    if (backendName.empty()) {
        LOGE("OH_NNDevice_GetName failed, error happened when getting name of deviceID.");
        *name = nullptr;
        return OH_NN_FAILED;
    }

    *name = backendName.data();
    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNDevice_GetType(size_t deviceID, OH_NN_DeviceType* deviceType)
{
    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(deviceID);
    if (backend == nullptr) {
        LOGE("OH_NNDevice_GetType failed, passed invalid deviceID.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (deviceType == nullptr) {
        LOGE("OH_NNDevice_GetType failed, passed nullptr to deviceType.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = backend->GetBackendType(*deviceType);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNDevice_GetType failed.");
        return ret;
    }
    return OH_NN_SUCCESS;
}

NNRT_API OH_NNCompilation *OH_NNCompilation_Construct(const OH_NNModel *model)
{
    if (model == nullptr) {
        LOGE("OH_NNCompilation_Construct failed, passed nullptr to model.");
        return nullptr;
    }

    Compilation *compilation = new (std::nothrow) Compilation();
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_Construct failed, please check whether it has enough memory.");
        return nullptr;
    }

    compilation->nnModel = const_cast<void*>(reinterpret_cast<const void*>(model));

    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    return nnCompilation;
}

NNRT_API OH_NNCompilation *OH_NNCompilation_ConstructWithOfflineModelFile(const char *modelPath)
{
    if (modelPath == nullptr) {
        LOGE("OH_NNCompilation_ConstructWithOfflineModelFile failed, passed nullptr to modelPath.");
        return nullptr;
    }

    Compilation *compilation = new (std::nothrow) Compilation();
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_ConstructWithOfflineModelFile failed, please check whether it has enough memory.");
        return nullptr;
    }

    compilation->offlineModelPath = const_cast<char*>(modelPath);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);

    return nnCompilation;
}

NNRT_API OH_NNCompilation *OH_NNCompilation_ConstructWithOfflineModelBuffer(const void *modelBuffer, size_t modelSize)
{
    if (modelBuffer == nullptr) {
        LOGE("OH_NNCompilation_ConstructWithOfflineModelBuffer failed, modelBuffer is nullptr.");
        return nullptr;
    }

    if (modelSize == static_cast<size_t>(0)) {
        LOGE("OH_NNCompilation_ConstructWithOfflineModelBuffer failed, modelSize is 0.");
        return nullptr;
    }

    Compilation *compilation = new (std::nothrow) Compilation();
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_ConstructWithOfflineModelBuffer failed, please check whether it has enough memory.");
        return nullptr;
    }

    compilation->offlineModelBuffer.first = const_cast<void*>(modelBuffer);
    compilation->offlineModelBuffer.second = modelSize;
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);

    return nnCompilation;
}

NNRT_API OH_NNCompilation *OH_NNCompilation_ConstructForCache()
{
    Compilation *compilation = new (std::nothrow) Compilation();
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_ConstructForCache failed, please check whether it has enough memory.");
        return nullptr;
    }

    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    return nnCompilation;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_ExportCacheToBuffer(OH_NNCompilation *compilation,
                                                               const void *buffer,
                                                               size_t length,
                                                               size_t *modelSize)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_ExportCacheToBuffer failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (buffer == nullptr) {
        LOGE("OH_NNCompilation_ExportCacheToBuffer failed, buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (length == static_cast<size_t>(0)) {
        LOGE("OH_NNCompilation_ExportCacheToBuffer failed, pass length equals to 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (modelSize == nullptr) {
        LOGE("OH_NNCompilation_ExportCacheToBuffer failed, modelSize is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);
    if (compilationImpl->compiler == nullptr) {
        LOGE("OH_NNCompilation_ExportCacheToBuffer failed, should call OH_NNCompilation_Build before export cache.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = compilationImpl->compiler->SaveToCacheBuffer(buffer, length, modelSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_ExportCacheToBuffer failed, fail to save cache to buffer.");
    }

    return ret;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_ImportCacheFromBuffer(OH_NNCompilation *compilation,
                                                                 const void *buffer,
                                                                 size_t modelSize)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_ImportCacheFromBuffer failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (buffer == nullptr) {
        LOGE("OH_NNCompilation_ImportCacheFromBuffer failed, buffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (modelSize == static_cast<size_t>(0)) {
        LOGE("OH_NNCompilation_ImportCacheFromBuffer failed, modelSize is 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);
    compilationImpl->offlineModelBuffer.first = const_cast<void*>(buffer);
    compilationImpl->offlineModelBuffer.second = modelSize;

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_AddExtensionConfig(OH_NNCompilation *compilation,
                                                              const char *configName,
                                                              const void *configValue,
                                                              const size_t configValueSize)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_AddExtensionConfig failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (configName == nullptr) {
        LOGE("OH_NNCompilation_AddExtensionConfig failed, configName is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (configValue == nullptr) {
        LOGE("OH_NNCompilation_AddExtensionConfig failed, configValue is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (configValueSize == static_cast<size_t>(0)) {
        LOGE("OH_NNCompilation_AddExtensionConfig failed, configValueSize is 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);

    std::string configNameStr = configName;
    if (configNameStr.empty()) {
        LOGE("OH_NNCompilation_AddExtensionConfig failed, configName is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<char> configValueVec(configValueSize, '0');
    void* configValueAddr = reinterpret_cast<void*>(configValueVec.data());
    errno_t ret = memcpy_s(configValueAddr, configValueVec.size(), configValue, configValueSize);
    if (ret != EOK) {
        LOGE("OH_NNCompilation_AddExtensionConfig failed, copy config value failed.");
        return OH_NN_FAILED;
    }

    auto iter = compilationImpl->configs.find(configNameStr);
    if (iter == compilationImpl->configs.end()) {
        compilationImpl->configs.emplace(configNameStr, configValueVec);
    } else {
        iter->second.emplace_back('|');
        iter->second.insert(iter->second.end(), configValueVec.begin(), configValueVec.end());
    }

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation *compilation, size_t deviceID)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetDevice failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);
    compilationImpl->backendID = deviceID;

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetCache(OH_NNCompilation *compilation,
                                                    const char *cachePath,
                                                    uint32_t version)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetCache failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (cachePath == nullptr) {
        LOGE("OH_NNCompilation_SetCache failed, cachePath is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);
    compilationImpl->cachePath = const_cast<char*>(cachePath);
    compilationImpl->cacheVersion = version;

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetPerformanceMode(OH_NNCompilation *compilation,
                                                              OH_NN_PerformanceMode performanceMode)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetPerformanceMode failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);
    compilationImpl->performance = performanceMode;

    if (compilationImpl->compiler != nullptr) {
        OH_NN_ReturnCode ret = compilationImpl->compiler->SetPerformance(performanceMode);
        if (ret != OH_NN_SUCCESS) {
            LOGE("OH_NNCompilation_SetPerformanceMode failed.");
            return ret;
        }
    }

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetPriority(OH_NNCompilation *compilation, OH_NN_Priority priority)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetPriority failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);
    compilationImpl->priority = priority;

    if (compilationImpl->compiler != nullptr) {
        OH_NN_ReturnCode ret = compilationImpl->compiler->SetPriority(priority);
        if (ret != OH_NN_SUCCESS) {
            LOGE("OH_NNCompilation_SetPriority failed.");
            return ret;
        }
    }

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_EnableFloat16(OH_NNCompilation *compilation, bool enableFloat16)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_EnableFloat16 failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);
    compilationImpl->enableFp16 = enableFloat16;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode CreateCompiler(Compilation* compilation, Compiler** compiler)
{
    if (compilation == nullptr) {
        LOGE("CreateCompiler failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (compiler == nullptr) {
        LOGE("CreateCompiler failed, compiler is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    BackendManager& manager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = manager.GetBackend(compilation->backendID);
    if (backend == nullptr) {
        LOGE("CreateCompiler failed, fail to get backend %{public}zu.", compilation->backendID);
        return OH_NN_FAILED;
    }

    *compiler = backend->CreateCompiler(compilation);
    if (*compiler == nullptr) {
        LOGE("CreateCompiler failed, fail to create compiler.");
        return OH_NN_FAILED;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode SetCompilationOptions(Compilation* compilation)
{
    if (compilation == nullptr) {
        LOGE("SetCompilationOptions failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (compilation->compiler == nullptr) {
        LOGE("SetCompilationOptions failed, compiler is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = OH_NN_SUCCESS;
    if (compilation->cachePath != nullptr) {
        ret = compilation->compiler->SetCacheDir(compilation->cachePath, compilation->cacheVersion);
        if (ret != OH_NN_SUCCESS) {
            LOGE("SetCompilationOptions failed, fail to set cache dir.");
            return ret;
        }
    }

    ret = compilation->compiler->SetEnableFp16(compilation->enableFp16);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetCompilationOptions failed, fail to set enable fp16.");
        return ret;
    }

    ret = compilation->compiler->SetPerformance(compilation->performance);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetCompilationOptions failed, fail to set performance.");
        return ret;
    }

    ret = compilation->compiler->SetPriority(compilation->priority);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetCompilationOptions failed, fail to set priority.");
        return ret;
    }

    ret = compilation->compiler->SetExtensionConfig(compilation->configs);
    if ((ret != OH_NN_SUCCESS) && (ret != OH_NN_UNSUPPORTED)) {
        LOGE("SetCompilationOptions failed, fail to set extenstion configs.");
        return ret;
    }

    ret = compilation->compiler->SetOptions(compilation->options);
    if ((ret != OH_NN_SUCCESS) && (ret != OH_NN_UNSUPPORTED)) {
        LOGE("SetCompilationOptions failed, fail to set extenstion options.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode CheckExceedRamLimit(const Compilation* compilation, bool& isExceedRamLimit)
{
    if (compilation == nullptr) {
        LOGE("CheckExceedRamLimit failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    size_t modelSize = 0;
    OH_NN_ReturnCode ret = GetModelSize(compilation, modelSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("CheckExceedRamLimit failed, fail to get model size.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (modelSize == 0) {
        LOGE("CheckExceedRamLimit failed, modelSize is 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    isExceedRamLimit = modelSize > MODEL_MAX_LIMIT ? true : false;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode AuthenticateModel(const Compilation* compilation, bool &isExceedRamLimit)
{
    OH_NN_ReturnCode retCode = CheckExceedRamLimit(compilation, isExceedRamLimit);
    if (retCode != OH_NN_SUCCESS) {
        LOGE("AuthenticateModel failed, fail to check if model exceed ram limit.");
        return retCode;
    }

    if (!isExceedRamLimit) {
        LOGD("Model accupy memory less then limit, no need authenticating");
        return OH_NN_SUCCESS; // If model ram is less than max limit, no need authenticating.
    }

    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGW("AuthenticateModel failed, fail to get nnrt service, skip authenticating.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.IsSupportAuthentication == nullptr) {
        LOGE("Authentication failed, nnrtService IsSupportAuthentication func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    bool supportStat = false;
    int ret = nnrtService.IsSupportAuthentication(&supportStat);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("Authentication failed, some error happened when judge if support authenticating.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }

    if (!supportStat) {
        LOGW("device not support authenticating, jumper over authenticating model.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.Authentication == nullptr) {
        LOGE("Authentication failed, nnrtService Authentication func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    ret = nnrtService.Authentication();
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("Authentication failed, input model cannot run by npu.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Authentication(Compilation** compilation, bool &isExceedRamLimit)
{
    if (compilation == nullptr) {
        LOGE("Authentication failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = *compilation;
    if (compilationImpl == nullptr) {
        LOGE("Authentication failed, compilation implementation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGW("Authentication failed, fail to get nnrt service, skip Authentication.");
        return OH_NN_SUCCESS;
    }

    OH_NN_ReturnCode ret = AuthenticateModel(compilationImpl, isExceedRamLimit);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Authentication failed, fail to authenticate model.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode GetModelId(Compilation** compilation)
{
    if (compilation == nullptr) {
        LOGE("GetModelId failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = *compilation;
    if (compilationImpl == nullptr) {
        LOGE("GetModelId failed, compilation implementation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto ret = GetNnrtModelId(compilationImpl);
    if (ret != OH_NN_SUCCESS) {
        LOGE("GetNnrtModelId is failed.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation *compilation)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_Build failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = OH_NN_SUCCESS;
    Compilation* compilationImpl = reinterpret_cast<Compilation*>(compilation);
    ret = IsCompilationAvaliable(compilationImpl);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, fail to compiler parameter.");
        return ret;
    }

    Compiler* compiler = nullptr;
    ret = CreateCompiler(compilationImpl, &compiler);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, fail to create compiler.");
        return ret;
    }
    compilationImpl->compiler = compiler;

    ret = SetCompilationOptions(compilationImpl);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, fail to create compiler.");
        return ret;
    }

    bool isExceedRamLimit = false;
    ret = Authentication(&compilationImpl, isExceedRamLimit);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, fail to create compiler.");
        return ret;
    }

    std::unordered_map<std::string, std::vector<char>> configs;

    std::vector<char> configContents;
    if (isExceedRamLimit) {
        configContents.push_back('1');
    } else {
        configContents.push_back('0');
    }

    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (nnrtService.IsServiceAvaliable()) {
        bool retCode = nnrtService.PullUpDlliteService();
        if (!retCode) {
            LOGW("OH_NNCompilation_Build failed, PullUpDlliteService failed.");
        }
    }

    configs["isExceedRamLimit"] = configContents;
    compilationImpl->compiler->SetExtensionConfig(configs);

    bool isBuild = compilationImpl->compiler->IsBuild();
    if (isBuild) {
        LOGE("OH_NNCompilation_Build failed, compilation has been built, don't build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    ret = compilationImpl->compiler->Build();
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, fail to build compilation.");
        return ret;
    }

    ret = GetModelId(&compilationImpl);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, fail to get modelId.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

NNRT_API void OH_NNCompilation_Destroy(OH_NNCompilation **compilation)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_Destroy failed, compilation is nullptr.");
        return;
    }

    if (*compilation == nullptr) {
        LOGE("OH_NNCompilation_Destroy failed, compilation is nullptr.");
        return;
    }

    Compilation* compilationImpl = reinterpret_cast<Compilation*>(*compilation);
    if (compilationImpl->compiler != nullptr) {
        BackendManager& manager = BackendManager::GetInstance();
        std::shared_ptr<Backend> backend = manager.GetBackend(compilationImpl->backendID);
        if (backend == nullptr) {
            LOGE("OH_NNCompilation_Destroy failed, fail to get backend %{public}zu.", compilationImpl->backendID);
            return;
        }

        OH_NN_ReturnCode ret = backend->DestroyCompiler(compilationImpl->compiler);
        if (ret != OH_NN_SUCCESS) {
            LOGE("OH_NNCompilation_Destroy failed, fail to destroy compiler.");
            return;
        }
    }

    delete compilationImpl;
    *compilation = nullptr;
}

NNRT_API NN_TensorDesc *OH_NNTensorDesc_Create()
{
    TensorDesc *tensorDescImpl = new (std::nothrow) TensorDesc();
    if (tensorDescImpl == nullptr) {
        LOGE("OH_NNTensorDesc_Create failed, failed to create tensor desc.");
        return nullptr;
    }

    NN_TensorDesc *tensorDesc = reinterpret_cast<NN_TensorDesc *>(tensorDescImpl);
    return tensorDesc;
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_Destroy(NN_TensorDesc **tensorDesc)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_Destroy failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (*tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_Destroy failed, *tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    TensorDesc *tensorDescImpl = reinterpret_cast<TensorDesc *>(*tensorDesc);
    delete tensorDescImpl;
    *tensorDesc = nullptr;
    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_SetName(NN_TensorDesc *tensorDesc, const char *name)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_SetName failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (name == nullptr) {
        LOGE("OH_NNTensorDesc_SetName failed, name is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    TensorDesc *tensorDescImpl = reinterpret_cast<TensorDesc *>(tensorDesc);
    return tensorDescImpl->SetName(name);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_GetName(const NN_TensorDesc *tensorDesc, const char **name)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_GetName failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (name == nullptr) {
        LOGE("OH_NNTensorDesc_GetName failed, name is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (*name != nullptr) {
        LOGE("OH_NNTensorDesc_GetName failed, *name is not nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const TensorDesc *tensorDescImpl = reinterpret_cast<const TensorDesc *>(tensorDesc);
    return tensorDescImpl->GetName(name);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_SetDataType(NN_TensorDesc *tensorDesc, OH_NN_DataType dataType)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_SetDataType failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    TensorDesc *tensorDescImpl = reinterpret_cast<TensorDesc *>(tensorDesc);
    return tensorDescImpl->SetDataType(dataType);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_GetDataType(const NN_TensorDesc *tensorDesc, OH_NN_DataType *dataType)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_GetDataType failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (dataType == nullptr) {
        LOGE("OH_NNTensorDesc_GetDataType failed, dataType is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const TensorDesc *tensorDescImpl = reinterpret_cast<const TensorDesc *>(tensorDesc);
    return tensorDescImpl->GetDataType(dataType);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_SetShape(NN_TensorDesc *tensorDesc, const int32_t *shape, size_t shapeLength)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_SetShape failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shape == nullptr) {
        LOGE("OH_NNTensorDesc_SetShape failed, shape is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shapeLength == 0) {
        LOGE("OH_NNTensorDesc_SetShape failed, shapeLength is 0.");
        return OH_NN_INVALID_PARAMETER;
    }
    TensorDesc *tensorDescImpl = reinterpret_cast<TensorDesc *>(tensorDesc);
    return tensorDescImpl->SetShape(shape, shapeLength);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_GetShape(const NN_TensorDesc *tensorDesc,
                                                   int32_t **shape,
                                                   size_t *shapeLength)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_GetShape failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shape == nullptr) {
        LOGE("OH_NNTensorDesc_GetShape failed, shape is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (*shape != nullptr) {
        LOGE("OH_NNTensorDesc_GetShape failed, *shape is not nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shapeLength == nullptr) {
        LOGE("OH_NNTensorDesc_GetShape failed, shapeLength is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const TensorDesc *tensorDescImpl = reinterpret_cast<const TensorDesc *>(tensorDesc);
    return tensorDescImpl->GetShape(shape, shapeLength);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_SetFormat(NN_TensorDesc *tensorDesc, OH_NN_Format format)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_SetFormat failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    TensorDesc *tensorDescImpl = reinterpret_cast<TensorDesc *>(tensorDesc);
    return tensorDescImpl->SetFormat(format);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_GetFormat(const NN_TensorDesc *tensorDesc, OH_NN_Format *format)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_GetFormat failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (format == nullptr) {
        LOGE("OH_NNTensorDesc_GetFormat failed, format is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const TensorDesc *tensorDescImpl = reinterpret_cast<const TensorDesc *>(tensorDesc);
    return tensorDescImpl->GetFormat(format);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_GetElementCount(const NN_TensorDesc *tensorDesc, size_t *elementCount)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_GetElementCount failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (elementCount == nullptr) {
        LOGE("OH_NNTensorDesc_GetElementCount failed, elementCount is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const TensorDesc *tensorDescImpl = reinterpret_cast<const TensorDesc *>(tensorDesc);
    return tensorDescImpl->GetElementNum(elementCount);
}

NNRT_API OH_NN_ReturnCode OH_NNTensorDesc_GetByteSize(const NN_TensorDesc *tensorDesc, size_t *byteSize)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensorDesc_GetByteSize failed, tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (byteSize == nullptr) {
        LOGE("OH_NNTensorDesc_GetByteSize failed, byteSize is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const TensorDesc *tensorDescImpl = reinterpret_cast<const TensorDesc *>(tensorDesc);
    return tensorDescImpl->GetByteSize(byteSize);
}

NNRT_API NN_Tensor* OH_NNTensor_Create(size_t deviceID, NN_TensorDesc *tensorDesc)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensor_Create failed, tensorDesc is nullptr.");
        return nullptr;
    }

    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(deviceID);
    if (backend == nullptr) {
        LOGE("OH_NNTensor_Create failed, passed invalid backend name.");
        return nullptr;
    }

    TensorDesc* descImpl = reinterpret_cast<TensorDesc*>(tensorDesc);
    Tensor* tensorImpl = backend->CreateTensor(descImpl);
    if (tensorImpl == nullptr) {
        LOGE("OH_NNTensor_Create failed, failed to create tensor.");
        return nullptr;
    }

    OH_NN_ReturnCode ret = tensorImpl->CreateData();
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNTensor_Create failed, failed to create tensor.");
        backend->DestroyTensor(tensorImpl);
        return nullptr;
    }

    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(tensorImpl);
    return tensor;
}

NNRT_API NN_Tensor* OH_NNTensor_CreateWithSize(size_t deviceID, NN_TensorDesc *tensorDesc, size_t size)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensor_CreateWithSize failed, tensorDesc is nullptr.");
        return nullptr;
    }

    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(deviceID);
    if (backend == nullptr) {
        LOGE("OH_NNTensor_CreateWithSize failed, passed invalid backend name.");
        return nullptr;
    }

    TensorDesc* descImpl = reinterpret_cast<TensorDesc*>(tensorDesc);
    Tensor* tensorImpl = backend->CreateTensor(descImpl);
    if (tensorImpl == nullptr) {
        LOGE("OH_NNTensor_CreateWithSize failed, failed to create tensor.");
        return nullptr;
    }

    OH_NN_ReturnCode ret = tensorImpl->CreateData(size);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNTensor_CreateWithSize failed, failed to create tensor.");
        backend->DestroyTensor(tensorImpl);
        return nullptr;
    }

    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(tensorImpl);
    return tensor;
}

NNRT_API NN_Tensor* OH_NNTensor_CreateWithFd(size_t deviceID,
                                             NN_TensorDesc *tensorDesc,
                                             int fd, size_t size,
                                             size_t offset)
{
    if (tensorDesc == nullptr) {
        LOGE("OH_NNTensor_CreateWithFd failed, tensorDesc is nullptr.");
        return nullptr;
    }
    if (fd < 0) {
        LOGE("OH_NNTensor_CreateWithFd failed, fd is less than zero.");
        return nullptr;
    }
    if (size == 0) {
        LOGE("OH_NNTensor_CreateWithFd failed, size is zero.");
        return nullptr;
    }
    if (size < offset) {
        LOGE("OH_NNTensor_CreateWithFd failed, size is smaller than offset.");
        return nullptr;
    }
    TensorDesc* descImpl = reinterpret_cast<TensorDesc*>(tensorDesc);
    size_t byteSize = 0;
    auto ret = descImpl->GetByteSize(&byteSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::CreateData failed, failed to get byte size from tensorDesc.");
        return nullptr;
    }
    if ((size - offset) < byteSize) {
        LOGE("OH_NNTensor_CreateWithFd failed, size of fd is insufficient.");
        return nullptr;
    }

    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(deviceID);
    if (backend == nullptr) {
        LOGE("OH_NNTensor_CreateWithFd failed, passed invalid backend name.");
        return nullptr;
    }

    Tensor* tensorImpl = backend->CreateTensor(descImpl);
    if (tensorImpl == nullptr) {
        LOGE("OH_NNTensor_CreateWithFd failed, failed to create tensor.");
        return nullptr;
    }

    ret = tensorImpl->CreateData(fd, size, offset);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNTensor_CreateWithFd failed, failed to create tensor.");
        backend->DestroyTensor(tensorImpl);
        return nullptr;
    }

    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(tensorImpl);
    return tensor;
}

NNRT_API OH_NN_ReturnCode OH_NNTensor_Destroy(NN_Tensor **tensor)
{
    if (tensor == nullptr) {
        LOGE("OH_NNTensor_Destroy failed, tensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (*tensor == nullptr) {
        LOGE("OH_NNTensor_Destroy failed, *tensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Tensor* tensorImpl = reinterpret_cast<Tensor*>(*tensor);
    if (tensorImpl == nullptr) {
        LOGE("OH_NNTensor_Destroy failed, tensorImpl is nullptr.");
        return OH_NN_NULL_PTR;
    }

    size_t backendID = tensorImpl->GetBackendID();
    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(backendID);
    if (backend == nullptr) {
        LOGE("OH_NNTensor_Destroy failed, passed invalid backend name %{public}zu.", backendID);
        return OH_NN_NULL_PTR;
    }

    auto ret = backend->DestroyTensor(tensorImpl);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNTensor_Destroy failed, failed to destroy tensor.");
        return ret;
    }
    *tensor = nullptr;
    return OH_NN_SUCCESS;
}

NNRT_API NN_TensorDesc* OH_NNTensor_GetTensorDesc(const NN_Tensor *tensor)
{
    if (tensor == nullptr) {
        LOGE("OH_NNTensor_GetTensorDesc failed, tensor is nullptr.");
        return nullptr;
    }

    const Tensor *tensorImpl = reinterpret_cast<const Tensor *>(tensor);
    auto tensorDescImpl = tensorImpl->GetTensorDesc();
    if (tensorDescImpl == nullptr) {
        LOGE("OH_NNTensor_GetTensorDesc failed, tensor desc is nullptr.");
        return nullptr;
    }

    NN_TensorDesc *tensorDesc = reinterpret_cast<NN_TensorDesc *>(tensorDescImpl);
    return tensorDesc;
}

NNRT_API void* OH_NNTensor_GetDataBuffer(const NN_Tensor *tensor)
{
    if (tensor == nullptr) {
        LOGE("OH_NNTensor_GetDataBuffer failed, tensor is nullptr.");
        return nullptr;
    }

    const Tensor *tensorImpl = reinterpret_cast<const Tensor *>(tensor);
    auto data = tensorImpl->GetData();
    if (data == nullptr) {
        LOGE("OH_NNTensor_GetDataBuffer failed, data is nullptr.");
        return nullptr;
    }

    return data;
}

NNRT_API OH_NN_ReturnCode OH_NNTensor_GetSize(const NN_Tensor *tensor, size_t *size)
{
    if (tensor == nullptr) {
        LOGE("OH_NNTensor_GetSize failed, tensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (size == nullptr) {
        LOGE("OH_NNTensor_GetSize failed, size is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const Tensor *tensorImpl = reinterpret_cast<const Tensor *>(tensor);
    *size = tensorImpl->GetSize();
    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNTensor_GetFd(const NN_Tensor *tensor, int *fd)
{
    if (tensor == nullptr) {
        LOGE("OH_NNTensor_GetFd failed, tensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (fd == nullptr) {
        LOGE("OH_NNTensor_GetFd failed, fd is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const Tensor *tensorImpl = reinterpret_cast<const Tensor *>(tensor);
    *fd = tensorImpl->GetFd();
    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNTensor_GetOffset(const NN_Tensor *tensor, size_t *offset)
{
    if (tensor == nullptr) {
        LOGE("OH_NNTensor_GetOffset failed, tensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (offset == nullptr) {
        LOGE("OH_NNTensor_GetOffset failed, offset is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const Tensor *tensorImpl = reinterpret_cast<const Tensor *>(tensor);
    *offset = tensorImpl->GetOffset();
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode Scheduling(Compilation** compilation)
{
    if (compilation == nullptr) {
        LOGE("Scheduling failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = *compilation;
    if (compilationImpl == nullptr) {
        LOGE("Scheduling failed, compilation implementation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGW("Scheduling failed, fail to get nnrt service, skip schedule.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.IsSupportScheduling == nullptr) {
        LOGE("Scheduling failed, nnrtService IsSupportScheduling func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::string cachePath = "";
    if (compilationImpl->cachePath != nullptr) {
        cachePath = compilationImpl->cachePath;
    }

    bool supportStat = false;
    int ret = nnrtService.IsSupportScheduling(&supportStat);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("Scheduling failed, some error happened when judge if support scheduling.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }
    if (!supportStat) {
        LOGW("device not support scheduling, jumper over scheduling.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.Scheduling == nullptr) {
        LOGE("Scheduling failed, nnrtService IsSupportScheduling func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    bool needModelLatency = false;
    ret = nnrtService.Scheduling(compilationImpl->hiaiModelId, &needModelLatency, cachePath.c_str());
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("Scheduling failed, some error happened when scheduling.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }

    compilationImpl->isNeedModelLatency = needModelLatency;

    LOGI("Scheduling success.");
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode SetModelId(const Compilation* compilation)
{
    if (compilation == nullptr) {
        LOGE("SetModelId failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGW("SetModelId failed, fail to get nnrt service, skip set modelId.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.SetModelID == nullptr) {
        LOGE("SetModelId failed, nnrtService SetModelID func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    int ret = nnrtService.SetModelID(
        compilation->hiaiModelId, compilation->nnrtModelID);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("SetModelId failed, fail to set modelId.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ExecutorPrepare(Executor** executor, Compilation** compilation)
{
    if (executor == nullptr) {
        LOGE("ExecutorPrepare failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (compilation == nullptr) {
        LOGE("ExecutorPrepare failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor* executorImpl = *executor;
    if (executorImpl == nullptr) {
        LOGE("ExecutorPrepare failed, executor implementation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpl = *compilation;
    if (compilationImpl == nullptr) {
        LOGE("ExecutorPrepare failed, compilation implementation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = SetModelId(compilationImpl);
    if (ret != OH_NN_SUCCESS) {
        LOGE("ExecutorPrepare failed, fail to set modelId.");
        return ret;
    }

    LOGD("ExecutorPrepare parameter, hiaiModelId: %{public}u, nnrtModelId: %{public}zu.",
        compilationImpl->hiaiModelId, compilationImpl->nnrtModelID);

    ret = Scheduling(&compilationImpl);
    if (ret != OH_NN_SUCCESS) {
        LOGE("ExecutorPrepare failed, failed to create executor.");
        return ret;
    }

    std::unordered_map<std::string, std::vector<char>> configMap;
    std::string hiaiModelIdStr = std::to_string(compilationImpl->hiaiModelId);
    std::vector<char> vechiaiModelId(hiaiModelIdStr.begin(), hiaiModelIdStr.end());
    vechiaiModelId.emplace_back('\0');
    configMap["hiaiModelId"] = vechiaiModelId;

    std::vector<char> vecNeedLatency = { static_cast<char>(compilationImpl->isNeedModelLatency) };
    configMap["isNeedModelLatency"] = vecNeedLatency;

    executorImpl->SetExtensionConfig(configMap);
    if (ret != OH_NN_SUCCESS) {
        LOGE("ExecutorPrepare failed, failed to set config to executor.");
        return ret;
    }

    ret = executorImpl->SetDeinitModelCallBack();
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetDeinitModelCallBack failed, failed to set DeinitModelCallBack to client.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

NNRT_API OH_NNExecutor *OH_NNExecutor_Construct(OH_NNCompilation *compilation)
{
    if (compilation == nullptr) {
        LOGE("OH_NNExecutor_Construct failed, compilation is nullptr.");
        return nullptr;
    }

    Compilation *compilationImpl = reinterpret_cast<Compilation *>(compilation);
    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(compilationImpl->backendID);
    if (backend == nullptr) {
        LOGE("OH_NNExecutor_Construct failed, failed to get backend of %{public}zu.", compilationImpl->backendID);
        return nullptr;
    }

    Executor* executorImpl = backend->CreateExecutor(compilationImpl);
    if (executorImpl == nullptr) {
        LOGE("OH_NNExecutor_Construct failed, failed to create executor.");
        return nullptr;
    }

    OH_NN_ReturnCode ret = executorImpl->GetModelID(compilationImpl->hiaiModelId);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_Construct failed, failed to get hiai modelId.");
        OH_NNExecutor_Destroy(reinterpret_cast<OH_NNExecutor **>(&executorImpl));
        return nullptr;
    }

    ret = ExecutorPrepare(&executorImpl, &compilationImpl);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_Construct failed, failed to prepare executor.");
        OH_NNExecutor_Destroy(reinterpret_cast<OH_NNExecutor **>(&executorImpl));
        return nullptr;
    }

    OH_NNExecutor *executor = reinterpret_cast<OH_NNExecutor *>(executorImpl);
    return executor;
}

OH_NN_ReturnCode Unload(const ExecutorConfig* config)
{
    if (config == nullptr) {
        LOGE("Unload failed, config is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGW("Unload failed, fail to get nnrt service, skip unload.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.Unload == nullptr) {
        LOGE("Unload failed, nnrtService Unload func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    int ret = nnrtService.Unload(config->hiaiModelId);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("Unload failed, some error happen when unload hiaiModelId.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }

    LOGI("Unload success.");
    return OH_NN_SUCCESS;
}

NNRT_API void OH_NNExecutor_Destroy(OH_NNExecutor **executor)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_Destroy failed, executor is nullptr.");
        return;
    }
    if (*executor == nullptr) {
        LOGE("OH_NNExecutor_Destroy failed, *executor is nullptr.");
        return;
    }

    Executor *executorImpl = reinterpret_cast<Executor *>(*executor);
    size_t backendID = executorImpl->GetBackendID();
    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(backendID);
    if (backend == nullptr) {
        LOGE("OH_NNExecutor_Destroy failed, failed to get backend of %{public}zu.", backendID);
        return;
    }

    OH_NN_ReturnCode ret = Unload(executorImpl->GetExecutorConfig());
    if (ret != OH_NN_SUCCESS) {
        LOGE("Unload failed, some error happened when unload nnrt service.");
    }

    auto returnCode = backend->DestroyExecutor(executorImpl);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_Destroy failed, failed to destroy executor.");
        return;
    }

    *executor = nullptr;
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_GetOutputShape(OH_NNExecutor *executor,
                                                       uint32_t outputIndex,
                                                       int32_t **shape,
                                                       uint32_t *shapeLength)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_GetOutputShape failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shape == nullptr) {
        LOGE("OH_NNExecutor_GetOutputShape failed, shape is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (*shape != nullptr) {
        LOGE("OH_NNExecutor_GetOutputShape failed, *shape is not nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shapeLength == nullptr) {
        LOGE("OH_NNExecutor_GetOutputShape failed, shapeLength is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    
    Executor *executorImpl = reinterpret_cast<Executor *>(executor);
    return executorImpl->GetOutputShape(outputIndex, shape, shapeLength);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_GetInputCount(const OH_NNExecutor *executor, size_t *inputCount)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_GetInputCount failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (inputCount == nullptr) {
        LOGE("OH_NNExecutor_GetInputCount failed, inputCount is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const Executor *executorImpl = reinterpret_cast<const Executor *>(executor);
    *inputCount = executorImpl->GetInputNum();
    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_GetOutputCount(const OH_NNExecutor *executor, size_t *outputCount)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_GetOutputCount failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputCount == nullptr) {
        LOGE("OH_NNExecutor_GetOutputCount failed, outputCount is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const Executor *executorImpl = reinterpret_cast<const Executor *>(executor);
    *outputCount = executorImpl->GetOutputNum();
    return OH_NN_SUCCESS;
}

NNRT_API NN_TensorDesc* OH_NNExecutor_CreateInputTensorDesc(const OH_NNExecutor *executor, size_t index)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_CreateInputTensorDesc failed, executor is nullptr.");
        return nullptr;
    }

    const Executor *executorImpl = reinterpret_cast<const Executor *>(executor);
    return executorImpl->CreateInputTensorDesc(index);
}

NNRT_API NN_TensorDesc* OH_NNExecutor_CreateOutputTensorDesc(const OH_NNExecutor *executor, size_t index)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_CreateOutputTensorDesc failed, executor is nullptr.");
        return nullptr;
    }

    const Executor *executorImpl = reinterpret_cast<const Executor *>(executor);
    return executorImpl->CreateOutputTensorDesc(index);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_GetInputDimRange(const OH_NNExecutor *executor,
                                                         size_t index,
                                                         size_t **minInputDims,
                                                         size_t **maxInputDims,
                                                         size_t *shapeLength)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_GetInputDimRange failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (minInputDims == nullptr) {
        LOGE("OH_NNExecutor_GetInputDimRange failed, minInputDims is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (maxInputDims == nullptr) {
        LOGE("OH_NNExecutor_GetInputDimRange failed, maxInputDims is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shapeLength == nullptr) {
        LOGE("OH_NNExecutor_GetInputDimRange failed, shapeLength is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    const Executor *executorImpl = reinterpret_cast<const Executor *>(executor);
    return executorImpl->GetInputDimRange(index, minInputDims, maxInputDims, shapeLength);
}
                                              
NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetOnRunDone(OH_NNExecutor *executor, NN_OnRunDone onRunDone)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_SetOnRunDone failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (onRunDone == nullptr) {
        LOGE("OH_NNExecutor_SetOnRunDone failed, onRunDone is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *executorImpl = reinterpret_cast<Executor *>(executor);
    return executorImpl->SetOnRunDone(onRunDone);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetOnServiceDied(OH_NNExecutor *executor, NN_OnServiceDied onServiceDied)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_SetOnServiceDied failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (onServiceDied == nullptr) {
        LOGE("OH_NNExecutor_SetOnServiceDied failed, onServiceDied is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *executorImpl = reinterpret_cast<Executor *>(executor);
    return executorImpl->SetOnServiceDied(onServiceDied);
}

OH_NN_ReturnCode UpdateModelLatency(const ExecutorConfig* config, int32_t modelLatency)
{
    if (config == nullptr) {
        LOGE("UpdateModelLatency failed, config is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGW("UpdateModelLatency failed, fail to get nnrt service, skip update model latency.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.UpdateModelLatency == nullptr) {
        LOGE("UpdateModelLatency failed, nnrtService UpdateModelLatency func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    LOGD("UpdateModelLatency, hiaiModelId: %{public}u, modelLatency: %{public}d.", config->hiaiModelId, modelLatency);

    int ret = nnrtService.UpdateModelLatency(config->hiaiModelId, modelLatency);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("UpdateModelLatency failed, nnrtService is not exist, jump over UpdateModelLatency.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode RunSync(Executor *executor,
                         NN_Tensor *inputTensor[],
                         size_t inputCount,
                         NN_Tensor *outputTensor[],
                         size_t outputCount)
{
    ExecutorConfig* configPtr = executor->GetExecutorConfig();
    if (configPtr == nullptr) {
        LOGE("RunSync failed, executor config is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    long timeStart = 0;
    if (configPtr->isNeedModelLatency) {
        timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    OH_NN_ReturnCode ret = executor->RunSync(inputTensor, inputCount, outputTensor, outputCount);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_RunSync failed, fail to run executor.");
        return ret;
    }

    if (configPtr->isNeedModelLatency) {
        long timeEnd = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        int32_t modelLatency = static_cast<int32_t>((timeEnd - timeStart));
        std::thread t(UpdateModelLatency, configPtr, modelLatency);
        t.detach();

        configPtr->isNeedModelLatency = false;
        std::unordered_map<std::string, std::vector<char>> configMap;
        std::vector<char> vecNeedLatency = { static_cast<char>(configPtr->isNeedModelLatency) };
        configMap["isNeedModelLatency"] = vecNeedLatency;

        ret = executor->SetExtensionConfig(configMap);
        if (ret != OH_NN_SUCCESS) {
            LOGE("OH_NNExecutor_RunSync failed, fail update executor config.");
            return ret;
        }
    }

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_RunSync(OH_NNExecutor *executor,
                                                NN_Tensor *inputTensor[],
                                                size_t inputCount,
                                                NN_Tensor *outputTensor[],
                                                size_t outputCount)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_RunSync failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (inputTensor == nullptr) {
        LOGE("OH_NNExecutor_RunSync failed, inputTensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if ((inputCount == 0) || (inputCount > INPUT_OUTPUT_MAX_INDICES)) {
        LOGE("OH_NNExecutor_RunSync failed, inputCount is 0 or more than 200.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputTensor == nullptr) {
        LOGE("OH_NNExecutor_RunSync failed, outputTensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if ((outputCount == 0) || (outputCount > INPUT_OUTPUT_MAX_INDICES)) {
        LOGE("OH_NNExecutor_RunSync failed, outputCount is 0 or more than 200.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *executorImpl = reinterpret_cast<Executor *>(executor);
    return RunSync(executorImpl, inputTensor, inputCount, outputTensor, outputCount);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_RunAsync(OH_NNExecutor *executor,
                                                 NN_Tensor* inputTensor[],
                                                 size_t inputCount,
                                                 NN_Tensor* outputTensor[],
                                                 size_t outputCount,
                                                 int32_t timeout,
                                                 void* userData)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_RunAsync failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (inputTensor == nullptr) {
        LOGE("OH_NNExecutor_RunAsync failed, inputTensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if ((inputCount == 0) || (inputCount > INPUT_OUTPUT_MAX_INDICES)) {
        LOGE("OH_NNExecutor_RunAsync failed, inputCount is 0 or more than 200.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputTensor == nullptr) {
        LOGE("OH_NNExecutor_RunAsync failed, outputTensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if ((outputCount == 0) || (outputCount > INPUT_OUTPUT_MAX_INDICES)) {
        LOGE("OH_NNExecutor_RunAsync failed, outputCount is 0 or more than 200.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (userData == nullptr) {
        LOGE("OH_NNExecutor_RunAsync failed, userData is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *executorImpl = reinterpret_cast<Executor *>(executor);
    return executorImpl->RunAsync(inputTensor, inputCount, outputTensor, outputCount, timeout, userData);
}
