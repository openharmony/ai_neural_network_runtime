/*
 * Copyright (c) 2022-2023 Huawei Device Co., Ltd.
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
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime.h"

#include "compilation.h"
#include "nnexecutor.h"
#include "inner_model.h"
#include "common/log.h"


using namespace OHOS::NeuralNetworkRuntime;

#define NNRT_API __attribute__((visibility("default")))

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

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetInput(OH_NNExecutor *executor,
    uint32_t inputIndex, const OH_NN_Tensor *tensor, const void *dataBuffer, size_t length)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_SetInput failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (tensor == nullptr) {
        LOGE("OH_NNExecutor_SetInput failed, tensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (dataBuffer == nullptr) {
        LOGE("OH_NNExecutor_SetInput failed, dataBuffer is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (length == 0) {
        LOGE("OH_NNExecutor_SetInput failed, dataBuffer length is 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNExecutor *executorImpl = reinterpret_cast<NNExecutor *>(executor);
    return executorImpl->SetInput(inputIndex, *tensor, dataBuffer, length);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetOutput(OH_NNExecutor *executor,
    uint32_t outputIndex, void *dataBuffer, size_t length)
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

    NNExecutor *executorImpl = reinterpret_cast<NNExecutor *>(executor);
    return executorImpl->SetOutput(outputIndex, dataBuffer, length);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_Run(OH_NNExecutor *executor)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_Run failed, passed nullptr to executor.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNExecutor *executorImpl = reinterpret_cast<NNExecutor *>(executor);
    return executorImpl->Run();
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
    NNExecutor *executorImpl = reinterpret_cast<NNExecutor *>(executor);
    OH_NN_ReturnCode ret = executorImpl->CreateInputMemory(inputIndex, length, &nnMemory);
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
    NNExecutor *executorImpl = reinterpret_cast<NNExecutor *>(executor);
    OH_NN_ReturnCode ret = executorImpl->CreateOutputMemory(outputIndex, length, &nnMemory);
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

    NNExecutor *executorImpl = reinterpret_cast<NNExecutor *>(executor);
    OH_NN_ReturnCode ret = executorImpl->DestroyInputMemory(inputIndex, memory);
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

    NNExecutor *executorImpl = reinterpret_cast<NNExecutor*>(executor);
    OH_NN_ReturnCode ret = executorImpl->DestroyOutputMemory(outputIndex, memory);
    if (ret != OH_NN_SUCCESS) {
        LOGE("OH_NNExecutor_DestroyOutputMemory failed, error happened when destroying output memory.");
        return;
    }

    *memory = nullptr;
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetInputWithMemory(OH_NNExecutor *executor,
    uint32_t inputIndex, const OH_NN_Tensor *tensor, const OH_NN_Memory *memory)
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

    NNExecutor *executorImpl = reinterpret_cast<NNExecutor *>(executor);
    return executorImpl->SetInputFromMemory(inputIndex, *tensor, *memory);
}

NNRT_API OH_NN_ReturnCode OH_NNExecutor_SetOutputWithMemory(OH_NNExecutor *executor,
    uint32_t outputIndex, const OH_NN_Memory *memory)
{
    if (executor == nullptr) {
        LOGE("OH_NNExecutor_SetOutputWithMemory failed, passed nullptr to executor.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (memory == nullptr) {
        LOGE("OH_NNExecutor_SetOutputWithMemory failed, passed nullptr to memory.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNExecutor *executorImpl = reinterpret_cast<NNExecutor *>(executor);
    return executorImpl->SetOutputFromMemory(outputIndex, *memory);
}