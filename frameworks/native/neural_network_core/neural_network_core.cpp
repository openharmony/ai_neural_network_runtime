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

#include "interfaces/kits/c/neural_network_runtime/neural_network_core.h"

#include <string>
#include <securec.h>

#include "common/log.h"
#include "executor.h"
#include "tensor.h"
#include "compilation.h"
#include "backend_manager.h"

using namespace OHOS::NeuralNetworkRuntime;
#define NNRT_API __attribute__((visibility("default")))

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
        LOGE("OH_NNDevice_GetName failed, error happened when getting name of deviceID %{public}zu.", deviceID);
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
        LOGE("OH_NNDevice_GetType failed, device id: %{public}zu.", deviceID);
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

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);
    if (compilationImpr->compiler == nullptr) {
        LOGE("OH_NNCompilation_ExportCacheToBuffer failed, should call OH_NNCompilation_Build before export cache.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = compilationImpr->compiler->SaveToCacheBuffer(buffer, length, modelSize);
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

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);
    compilationImpr->offlineModelBuffer.first = const_cast<void*>(buffer);
    compilationImpr->offlineModelBuffer.second = modelSize;

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

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);

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

    auto emplaceResult = compilationImpr->configs.emplace(configNameStr, configValueVec);
    if (!emplaceResult.second) {
        LOGE("OH_NNCompilation_AddExtensionConfig failed, configName %{public}s already exists,"
             "don't set again.", configName);
        return OH_NN_FAILED;
    }

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation *compilation, size_t deviceID)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetDevice failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);
    compilationImpr->backendID = deviceID;

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

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);
    compilationImpr->cachePath = const_cast<char*>(cachePath);
    compilationImpr->cacheVersion = version;

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetPerformanceMode(OH_NNCompilation *compilation,
                                                              OH_NN_PerformanceMode performanceMode)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetPerformanceMode failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);
    compilationImpr->performance = performanceMode;

    if (compilationImpr->compiler != nullptr) {
        OH_NN_ReturnCode ret = compilationImpr->compiler->SetPerformance(performanceMode);
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

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);
    compilationImpr->priority = priority;

    if (compilationImpr->compiler != nullptr) {
        OH_NN_ReturnCode ret = compilationImpr->compiler->SetPriority(priority);
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

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);
    compilationImpr->enableFp16 = enableFloat16;

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

NNRT_API OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation *compilation)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_Build failed, compilation is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(compilation);

    if (((compilationImpr->nnModel != nullptr) && (compilationImpr->offlineModelPath != nullptr)) ||
        ((compilationImpr->nnModel != nullptr) &&
         ((compilationImpr->offlineModelBuffer.first != nullptr) ||
          (compilationImpr->offlineModelBuffer.second != static_cast<size_t>(0)))) ||
        ((compilationImpr->offlineModelPath != nullptr) &&
         ((compilationImpr->offlineModelBuffer.first != nullptr) ||
          (compilationImpr->offlineModelBuffer.second != static_cast<size_t>(0))))) {
        LOGE("OH_NNCompilation_Build failed, find multi model to build compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = OH_NN_SUCCESS;
    if (compilationImpr->compiler != nullptr) {
        LOGE("OH_NNCompilation_Build failed, the compiler in compilation is not nullptr, "
             "please input a new compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compiler* compiler = nullptr;
    ret = CreateCompiler(compilationImpr, &compiler);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, faile to create compiler.");
        return ret;
    }
    compilationImpr->compiler = compiler;

    ret = SetCompilationOptions(compilationImpr);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, faile to create compiler.");
        return ret;
    }

    bool isBuild = compilationImpr->compiler->IsBuild();
    if (isBuild) {
        LOGE("OH_NNCompilation_Build failed, compilation has been built, don't build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    ret = compilationImpr->compiler->Build();
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNCompilation_Build failed, faile to build compilation.");
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

    Compilation* compilationImpr = reinterpret_cast<Compilation*>(*compilation);
    if (compilationImpr->compiler != nullptr) {
        BackendManager& manager = BackendManager::GetInstance();
        std::shared_ptr<Backend> backend = manager.GetBackend(compilationImpr->backendID);
        if (backend == nullptr) {
            LOGE("OH_NNCompilation_Destroy failed, fail to get backend %{public}zu.", compilationImpr->backendID);
            return;
        }

        OH_NN_ReturnCode ret = backend->DestroyCompiler(compilationImpr->compiler);
        if (ret != OH_NN_SUCCESS) {
            LOGE("OH_NNCompilation_Destroy failed, fail to destroy compiler.");
            return;
        }
    }

    delete compilationImpr;
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
        LOGE("OH_NNTensor_Create failed, passed invalid backend name %{public}zu.", deviceID);
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
        LOGE("OH_NNTensor_CreateWithSize failed, passed invalid backend name %{public}zu.", deviceID);
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
        LOGE("OH_NNTensor_CreateWithFd failed, passed invalid backend name %{public}zu.", deviceID);
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

    OH_NNExecutor *executor = reinterpret_cast<OH_NNExecutor *>(executorImpl);
    return executor;
}

NNRT_API void OH_NNExecutor_Destroy(OH_NNExecutor **executor)
{
    if (executor == nullptr) {
        LOGW("OH_NNExecutor_Destroy failed, executor is nullptr.");
        return;
    }
    if (*executor == nullptr) {
        LOGW("OH_NNExecutor_Destroy failed, *executor is nullptr.");
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
    if (inputCount == 0) {
        LOGE("OH_NNExecutor_RunSync failed, inputCount is 0.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputTensor == nullptr) {
        LOGE("OH_NNExecutor_RunSync failed, outputTensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputCount == 0) {
        LOGE("OH_NNExecutor_RunSync failed, outputCount is 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *executorImpl = reinterpret_cast<Executor *>(executor);
    return executorImpl->RunSync(inputTensor, inputCount, outputTensor, outputCount);
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
    if (inputCount == 0) {
        LOGE("OH_NNExecutor_RunAsync failed, inputCount is 0.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputTensor == nullptr) {
        LOGE("OH_NNExecutor_RunAsync failed, outputTensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputCount == 0) {
        LOGE("OH_NNExecutor_RunAsync failed, outputCount is 0.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (userData == nullptr) {
        LOGE("OH_NNExecutor_RunAsync failed, userData is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *executorImpl = reinterpret_cast<Executor *>(executor);
    return executorImpl->RunAsync(inputTensor, inputCount, outputTensor, outputCount, timeout, userData);
}
