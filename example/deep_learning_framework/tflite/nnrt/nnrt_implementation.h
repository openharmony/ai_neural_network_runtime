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

#ifndef TENSORFLOW_LITE_NNRT_IMPLEMENTATION_H
#define TENSORFLOW_LITE_NNRT_IMPLEMENTATION_H

#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <cstdio>
#include <cstdlib>
#include <memory>

#include "neural_network_runtime_type.h"

namespace tflite {
#define NNRT_LOG(format, ...) fprintf(stderr, format "\n", __VA_ARGS__)

struct NnrtApi {
    // This indicates the availability of nnrt library. If it is false, it means that loading
    // the nnrt library failed and tflite will not use nnrt to run the model, vice versa.
    bool nnrtExists;

    // Create model interface
    OH_NNModel* (*OH_NNModel_Construct)(void);
    OH_NN_ReturnCode (*OH_NNModel_AddTensor)(OH_NNModel* model, const OH_NN_Tensor* nnTensor);
    OH_NN_ReturnCode (*OH_NNModel_SetTensorData)(OH_NNModel* model, uint32_t index, const void* buffer,
        size_t length);
    OH_NN_ReturnCode (*OH_NNModel_AddOperation)(OH_NNModel* model, OH_NN_OperationType op,
        const OH_NN_UInt32Array* paramIndices, const OH_NN_UInt32Array* inputIndices,
        const OH_NN_UInt32Array* outputIndices);
    OH_NN_ReturnCode (*OH_NNModel_SpecifyInputsAndOutputs)(OH_NNModel* model, const OH_NN_UInt32Array* inputIndices,
        const OH_NN_UInt32Array* outputIndices);
    OH_NN_ReturnCode (*OH_NNModel_Finish)(OH_NNModel* model);
    void (*OH_NNModel_Destroy)(OH_NNModel** model);
    OH_NN_ReturnCode (*OH_NNModel_GetAvailableOperations)(OH_NNModel* model, size_t deviceID, const bool** isSupported,
        uint32_t* opCount);
    // Compilation interface
    OH_NNCompilation* (*OH_NNCompilation_Construct)(const OH_NNModel* model);
    OH_NN_ReturnCode (*OH_NNCompilation_SetCache)(OH_NNCompilation* compilation, const char* cacheDir,
        uint32_t version);
    OH_NN_ReturnCode (*OH_NNCompilation_SetPerformanceMode)(OH_NNCompilation* compilation,
        OH_NN_PerformanceMode performanceMode);
    OH_NN_ReturnCode (*OH_NNCompilation_SetPriority)(OH_NNCompilation* compilation, OH_NN_Priority priority);
    OH_NN_ReturnCode (*OH_NNCompilation_EnableFloat16)(OH_NNCompilation* compilation, bool enablefloat16);
    OH_NN_ReturnCode (*OH_NNCompilation_SetDevice)(OH_NNCompilation* compilation, size_t deviceID);
    OH_NN_ReturnCode (*OH_NNCompilation_Build)(OH_NNCompilation* compilation);
    void (*OH_NNCompilation_Destroy)(OH_NNCompilation** compilation);
    // Executor interface
    OH_NNExecutor* (*OH_NNExecutor_Construct)(OH_NNCompilation* compilation);
    OH_NN_ReturnCode (*OH_NNExecutor_SetInput)(OH_NNExecutor* executor, uint32_t inputIndex,
        const OH_NN_Tensor* nnTensor, const void* buffer, size_t length);
    OH_NN_ReturnCode (*OH_NNExecutor_SetOutput)(const OH_NNExecutor* executor, uint32_t outputIndex, void* buffer,
        size_t length);
    OH_NN_ReturnCode (*OH_NNExecutor_GetOutputShape)(const OH_NNExecutor* executor, uint32_t outputIndex,
        const uint32_t** dimensions, uint32_t* dimensionCount);
    OH_NN_ReturnCode (*OH_NNExecutor_Run)(OH_NNExecutor* executor);
    OH_NN_Memory* (*OH_NNExecutor_AllocateInputMemory)(OH_NNExecutor* executor, uint32_t inputIndex, size_t length);
    OH_NN_Memory* (*OH_NNExecutor_AllocateOutputMemory)(OH_NNExecutor* executor, uint32_t outputIndex, size_t length);
    void (*OH_NNExecutor_DestroyOutputMemory)(OH_NNExecutor* executor, uint32_t outputIndex, OH_NN_Memory** memory);
    void (*OH_NNExecutor_DestroyInputMemory)(OH_NNExecutor* executor, uint32_t inputIndex, OH_NN_Memory** memory);
    OH_NN_ReturnCode (*OH_NNExecutor_SetInputWithMemory)(OH_NNExecutor* executor, uint32_t inputIndex,
        const OH_NN_Tensor* nnTensor, const OH_NN_Memory* memory);
    OH_NN_ReturnCode (*OH_NNExecutor_SetOutputWithMemory)(OH_NNExecutor* executor, uint32_t outputIndex,
        const OH_NN_Memory* memory);
    void (*OH_NNExecutor_Destroy)(OH_NNExecutor** executor);
    // Device interface
    OH_NN_ReturnCode (*OH_NNDevice_GetAllDevicesID)(const size_t** allDevicesID, uint32_t* deviceCount);
    OH_NN_ReturnCode (*OH_NNDevice_GetName)(size_t deviceID, const char** name);
    OH_NN_ReturnCode (*OH_NNDevice_GetType)(size_t deviceID, OH_NN_DeviceType* deviceType);
};

const NnrtApi* NnrtImplementation();
} // namespace tflite

#endif // TENSORFLOW_LITE_NNRT_IMPLEMENTATION_H