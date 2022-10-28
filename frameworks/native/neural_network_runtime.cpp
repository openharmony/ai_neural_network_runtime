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

#include "interfaces/innerkits/c/neural_network_runtime_inner.h"
#include "interfaces/kits/c/neural_network_runtime.h"

#include "compilation.h"
#include "device_manager.h"
#include "executor.h"
#include "inner_model.h"
#include "common/log.h"


using namespace OHOS::NeuralNetworkRuntime;

#define NNRT_API __attribute__((visibility("default")))

NNRT_API OH_NNModel *OH_NNModel_Construct(void)
{
    InnerModel *innerModel = new(std::nothrow) InnerModel();
    if (innerModel == nullptr) {
        LOGE("OH_NNModel_Construct failed, please check whether it has enough memory.");
        return nullptr;
    }

    OH_NNModel *nnModel = reinterpret_cast<OH_NNModel*>(innerModel);
    return nnModel;
}

NNRT_API OH_NN_ReturnCode OH_NNModel_AddTensor(OH_NNModel *model, const OH_NN_Tensor *tensor)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_AddTensor failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor == nullptr) {
        LOGE("OH_NNModel_AddTensor failed, passed nullptr to tensor.");
        return OH_NN_INVALID_PARAMETER;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->AddTensor(*tensor);
}

NNRT_API OH_NN_ReturnCode OH_NNModel_AddOperation(OH_NNModel *model,
                                                  OH_NN_OperationType op,
                                                  const OH_NN_UInt32Array *paramIndices,
                                                  const OH_NN_UInt32Array *inputIndices,
                                                  const OH_NN_UInt32Array *outputIndices)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_AddOperation failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (paramIndices == nullptr) {
        LOGE("OH_NNModel_AddOperation failed, passed nullptr to paramIndices.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (inputIndices == nullptr) {
        LOGE("OH_NNModel_AddOperation failed, passed nullptr to inputIndices.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (outputIndices == nullptr) {
        LOGE("OH_NNModel_AddOperation failed, passed nullptr to outputIndices.");
        return OH_NN_INVALID_PARAMETER;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->AddOperation(op, *paramIndices, *inputIndices, *outputIndices);
}

NNRT_API OH_NN_ReturnCode OH_NNModel_SetTensorData(OH_NNModel *model,
                                                   uint32_t index,
                                                   const void *dataBuffer,
                                                   size_t length)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_SetTensorData failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (dataBuffer == nullptr) {
        LOGE("OH_NNModel_SetTensorData failed, passed nullptr to dataBuffer, which has no effect.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (length == 0) {
        LOGE("OH_NNModel_SetTensorData failed, passed dataBuffer with length 0, which has no effect.");
        return OH_NN_INVALID_PARAMETER;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->SetTensorValue(index, dataBuffer, length);
}

NNRT_API OH_NN_ReturnCode OH_NNModel_SpecifyInputsAndOutputs(OH_NNModel *model,
                                                             const OH_NN_UInt32Array *inputIndices,
                                                             const OH_NN_UInt32Array *outputIndices)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_SpecifyInputsAndOutputs failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (inputIndices == nullptr) {
        LOGE("OH_NNModel_SpecifyInputsAndOutputs failed, passed nullptr to inputIndices.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (outputIndices == nullptr) {
        LOGE("OH_NNModel_SpecifyInputsAndOutputs failed, passed nullptr to outputIndices.");
        return OH_NN_INVALID_PARAMETER;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->SpecifyInputsAndOutputs(*inputIndices, *outputIndices);
}

NNRT_API OH_NN_ReturnCode OH_NNModel_Finish(OH_NNModel *model)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_Finish failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->Build();
}

NNRT_API OH_NN_ReturnCode OH_NNModel_BuildFromLiteGraph(OH_NNModel *model, const void *liteGraph)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_BuildFromLiteGraph failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (liteGraph == nullptr) {
        LOGE("OH_NNModel_BuildFromLiteGraph failed, passed nullptr to liteGraph.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto *pLiteGraph = static_cast<const mindspore::lite::LiteGraph*>(liteGraph);
    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);

    // Once the innerModel built from the liteGraph successfully, the innerModel
    // owns the liteGraph, in which case, the invoker should not delete
    // the liteGraph actively. Otherwise, the invoker still has the ownership.
    return innerModel->BuildFromLiteGraph(pLiteGraph);
}

NNRT_API void OH_NNModel_Destroy(OH_NNModel **model)
{
    if (model == nullptr) {
        LOGW("OH_NNModel_Destroy has no effect, passed nullptr to model.");
        return;
    }

    if (*model == nullptr) {
        LOGW("OH_NNModel_Destroy has no effect, passed nullptr to *model.");
        return;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(*model);
    delete innerModel;
    *model = nullptr;
}

NNRT_API OH_NN_ReturnCode OH_NNModel_GetAvailableOperations(OH_NNModel *model,
                                                            size_t deviceID,
                                                            const bool **isAvailable,
                                                            uint32_t *opCount)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_GetAvailableOperations failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (isAvailable == nullptr) {
        LOGE("OH_NNModel_GetAvailableOperations failed, passed nullptr to isAvailable.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (*isAvailable != nullptr) {
        LOGE("OH_NNModel_GetAvailableOperations failed, *isAvailable is not nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (opCount == nullptr) {
        LOGE("OH_NNModel_GetAvailableOperations failed, passed nullptr to opCount.");
        return OH_NN_INVALID_PARAMETER;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->GetSupportedOperations(deviceID, isAvailable, *opCount);
}

NNRT_API OH_NNCompilation *OH_NNCompilation_Construct(const OH_NNModel *model)
{
    if (model == nullptr) {
        LOGE("OH_NNCompilation_Construct failed, passed nullptr to model.");
        return nullptr;
    }
    const InnerModel *innerModel = reinterpret_cast<const InnerModel*>(model);

    if (!innerModel->IsBuild()) {
        LOGE("OH_NNCompilation_Construct failed, should call OH_NNModel_Finish before creating compilation.");
        return nullptr;
    }

    Compilation *compilation = new(std::nothrow) Compilation(innerModel);
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_Construct failed, please check whether it has enough memory.");
        return nullptr;
    }

    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    return nnCompilation;
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation *compilation, size_t deviceID)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetDevice failed, passed nullptr to compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* innerCompilation = reinterpret_cast<Compilation*>(compilation);
    return innerCompilation->SetDevice(deviceID);
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetCache(OH_NNCompilation *compilation,
                                                    const char *cachePath,
                                                    uint32_t version)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetCache failed, passed nullptr to compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (cachePath == nullptr) {
        LOGE("OH_NNCompilation_SetCache failed, passed nullptr to cachePath.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* innerCompilation = reinterpret_cast<Compilation*>(compilation);
    return innerCompilation->SetCacheDir(cachePath, version);
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetPerformanceMode(OH_NNCompilation *compilation,
                                                              OH_NN_PerformanceMode performanceMode)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetPerformanceMode failed, passed nullptr to compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* innerCompilation = reinterpret_cast<Compilation*>(compilation);
    return innerCompilation->SetPerformance(performanceMode);
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_SetPriority(OH_NNCompilation *compilation,
                                                       OH_NN_Priority priority)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_SetPriority failed, passed nullptr to compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* innerCompilation = reinterpret_cast<Compilation*>(compilation);
    return innerCompilation->SetPriority(priority);
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_EnableFloat16(OH_NNCompilation *compilation, bool enableFloat16)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_EnableFloat16 failed, passed nullptr to compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* innerCompilation = reinterpret_cast<Compilation*>(compilation);
    return innerCompilation->SetEnableFp16(enableFloat16);
}

NNRT_API OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation *compilation)
{
    if (compilation == nullptr) {
        LOGE("OH_NNCompilation_Build failed, passed nullptr to compilation.");
        return OH_NN_INVALID_PARAMETER;
    }

    Compilation* innerCompilation = reinterpret_cast<Compilation*>(compilation);
    return innerCompilation->Build();
}

NNRT_API void OH_NNCompilation_Destroy(OH_NNCompilation **compilation)
{
    if (compilation == nullptr) {
        LOGW("OH_NNCompilation_Destroy has no effect, passed nullptr to compilation.");
        return;
    }

    if (*compilation == nullptr) {
        LOGW("OH_NNCompilation_Destroy has no effect, passed nullptr to *compilation.");
        return;
    }

    Compilation *innerCompilation = reinterpret_cast<Compilation*>(*compilation);
    delete innerCompilation;
    *compilation = nullptr;
}

NNRT_API OH_NNExecutor *OH_NNExecutor_Construct(OH_NNCompilation *compilation)
{
    if (compilation == nullptr) {
        LOGE("OH_NNExecutor_Construct failed, passed nullptr to compilation.");
        return nullptr;
    }
    Compilation *innerCompilation = reinterpret_cast<Compilation*>(compilation);

    if (!innerCompilation->IsBuild()) {
        LOGE("OH_NNExecutor_Construct failed, should call OH_NNCompilation_Build before creating executor.");
        return nullptr;
    }

    Executor* executor = new(std::nothrow) Executor(innerCompilation);
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_Construct failed, please check whether it has enough memory.");
        return nullptr;
    }

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);
    return nnExecutor;
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetInput(OH_NNExecutor *executor,
                                                 uint32_t inputIndex,
                                                 const OH_NN_Tensor *tensor,
                                                 const void *dataBuffer,
                                                 size_t length)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_SetInput failed, passed nullptr to executor.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor == nullptr) {
        LOGE("OH_NNExecutor_SetInput failed, passed nullptr to tensor.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (dataBuffer == nullptr) {
        LOGE("OH_NNExecutor_SetInput failed, passed nullptr to dataBuffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (length == 0) {
        LOGE("OH_NNExecutor_SetInput failed, dataBuffer length is 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor* innerExecutor = reinterpret_cast<Executor*>(executor);
    return innerExecutor->SetInput(inputIndex, *tensor, dataBuffer, length);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetOutput(OH_NNExecutor *executor,
                                                  uint32_t outputIndex,
                                                  void *dataBuffer,
                                                  size_t length)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_SetOutput failed, passed nullptr to executor.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (dataBuffer == nullptr) {
        LOGE("OH_NNExecutor_SetOutput failed, passed nullptr to dataBuffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (length == 0) {
        LOGE("OH_NNExecutor_SetOutput failed, dataBuffer length is 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor* innerExecutor = reinterpret_cast<Executor*>(executor);
    return innerExecutor->SetOutput(outputIndex, dataBuffer, length);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_GetOutputShape(OH_NNExecutor *executor,
                                                       uint32_t outputIndex,
                                                       int32_t **shape,
                                                       uint32_t *shapeLength)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_GetOutputShape failed, passed nullptr to executor.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (shape == nullptr) {
        LOGE("OH_NNExecutor_GetOutputShape failed, passed nullptr to shape.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (*shape != nullptr) {
        LOGE("OH_NNExecutor_GetOutputShape failed, *shape is not nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (shapeLength == nullptr) {
        LOGE("OH_NNExecutor_GetOutputShape failed, passed nullptr to shapeLength.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor* innerExecutor = reinterpret_cast<Executor*>(executor);
    return innerExecutor->GetOutputShape(outputIndex, shape, *shapeLength);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_Run(OH_NNExecutor *executor)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_Run failed, passed nullptr to executor.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *innerExecutor = reinterpret_cast<Executor*>(executor);
    return innerExecutor->Run();
}

NNRT_API OH_NN_Memory *OH_NNExecutor_AllocateInputMemory(OH_NNExecutor *executor, uint32_t inputIndex, size_t length)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_AllocateInputMemory failed, passed nullptr to executor.");
        return nullptr;
    }

    if (length == 0) {
        LOGW("OH_NNExecutor_AllocateInputMemory has no effect, passed length equals 0.");
        return nullptr;
    }

    OH_NN_Memory *nnMemory = nullptr;
    Executor *innerExecutor = reinterpret_cast<Executor*>(executor);
    OH_NN_ReturnCode ret = innerExecutor->CreateInputMemory(inputIndex, length, &nnMemory);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_AllocateInputMemory failed, error happened when creating input memory in executor.");
        return nullptr;
    }

    return nnMemory;
}

NNRT_API OH_NN_Memory *OH_NNExecutor_AllocateOutputMemory(OH_NNExecutor *executor, uint32_t outputIndex, size_t length)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_AllocateOutputMemory failed, passed nullptr to executor.");
        return nullptr;
    }

    if (length == 0) {
        LOGW("OH_NNExecutor_AllocateOutputMemory has no effect, passed length equals 0.");
        return nullptr;
    }

    OH_NN_Memory *nnMemory = nullptr;
    Executor *innerExecutor = reinterpret_cast<Executor*>(executor);
    OH_NN_ReturnCode ret = innerExecutor->CreateOutputMemory(outputIndex, length, &nnMemory);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_AllocateOutputMemory failed, error happened when creating output memory in executor.");
        return nullptr;
    }

    return nnMemory;
}

NNRT_API void OH_NNExecutor_DestroyInputMemory(OH_NNExecutor *executor, uint32_t inputIndex, OH_NN_Memory **memory)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_DestroyInputMemory failed, passed nullptr to executor.");
        return;
    }

    if (memory == nullptr) {
        LOGW("OH_NNExecutor_DestroyInputMemory has no effect, passed nullptr to memory.");
        return;
    }

    if (*memory == nullptr) {
        LOGW("OH_NNExecutor_DestroyInputMemory has no effect, passed nullptr to *memory.");
        return;
    }

    Executor *innerExecutor = reinterpret_cast<Executor*>(executor);
    OH_NN_ReturnCode ret = innerExecutor->DestroyInputMemory(inputIndex, memory);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_DestroyInputMemory failed, error happened when destroying input memory.");
        return;
    }

    *memory = nullptr;
}

NNRT_API void OH_NNExecutor_DestroyOutputMemory(OH_NNExecutor *executor, uint32_t outputIndex, OH_NN_Memory **memory)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_DestroyOutputMemory failed, passed nullptr to executor.");
        return;
    }

    if (memory == nullptr) {
        LOGW("OH_NNExecutor_DestroyOutputMemory has no effect, passed nullptr to memory.");
        return;
    }

    if (*memory == nullptr) {
        LOGW("OH_NNExecutor_DestroyOutputMemory has no effect, passed nullptr to *memory.");
        return;
    }

    Executor *innerExecutor = reinterpret_cast<Executor*>(executor);
    OH_NN_ReturnCode ret = innerExecutor->DestroyOutputMemory(outputIndex, memory);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_DestroyOutputMemory failed, error happened when destroying output memory.");
        return;
    }

    *memory = nullptr;
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetInputWithMemory(OH_NNExecutor *executor,
                                                           uint32_t inputIndex,
                                                           const OH_NN_Tensor *tensor,
                                                           const OH_NN_Memory *memory)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_SetInputWithMemory failed, passed nullptr to executor.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensor == nullptr) {
        LOGE("OH_NNExecutor_SetInputWithMemory failed, passed nullptr to tensor.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (memory == nullptr) {
        LOGE("OH_NNExecutor_SetInputWithMemory failed, passed nullptr to memory.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *innerExecutor = reinterpret_cast<Executor*>(executor);
    return innerExecutor->SetInputFromMemory(inputIndex, *tensor, *memory);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetOutputWithMemory(OH_NNExecutor *executor,
                                                            uint32_t outputIndex,
                                                            const OH_NN_Memory *memory)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_SetOutputWithMemory failed, passed nullptr to executor.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (memory == nullptr) {
        LOGE("OH_NNExecutor_SetOutputWithMemory failed, passed nullptr to memory.");
        return OH_NN_INVALID_PARAMETER;
    }

    Executor *innerExecutor = reinterpret_cast<Executor*>(executor);
    return innerExecutor->SetOutputFromMemory(outputIndex, *memory);
}

NNRT_API void OH_NNExecutor_Destroy(OH_NNExecutor **executor)
{
    if (executor == nullptr) {
        LOGW("OH_NNExecutor_Destroy has no effect, since executor is nullptr.");
        return;
    }

    if ((*executor) == nullptr) {
        LOGW("OH_NNExecutor_Destroy has no effect, since *executor is nullptr");
        return;
    }

    Executor *innerExecutor = reinterpret_cast<Executor*>(*executor);
    delete innerExecutor;
    *executor = nullptr;
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

    DeviceManager& deviceManager = DeviceManager::GetInstance();
    const std::vector<size_t>& allDevices = deviceManager.GetAllDeviceId();

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

    DeviceManager& deviceManager = DeviceManager::GetInstance();
    const std::string& deviceName = deviceManager.GetDeviceName(deviceID);
    if (deviceName.empty()) {
        LOGE("OH_NNDevice_GetName failed, error happened when getting name of deviceID %zu.", deviceID);
        *name = nullptr;
        return OH_NN_FAILED;
    }

    *name = deviceName.data();
    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNDevice_GetType(size_t deviceID, OH_NN_DeviceType* deviceType)
{
    DeviceManager& deviceManager = DeviceManager::GetInstance();
    std::shared_ptr<Device> device = deviceManager.GetDevice(deviceID);
    if (device == nullptr) {
        LOGE("OH_NNDevice_GetName failed, passed invalid deviceID.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (deviceType == nullptr) {
        LOGE("OH_NNDevice_GetType failed, passed nullptr to deviceType.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = device->GetDeviceType(*deviceType);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNDevice_GetType failed, device id: %zu.", deviceID);
        return ret;
    }
    return OH_NN_SUCCESS;
}