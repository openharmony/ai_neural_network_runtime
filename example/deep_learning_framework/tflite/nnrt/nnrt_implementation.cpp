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

#include "nnrt_implementation.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

#include <algorithm>
#include <string>

namespace tflite {
// These function parameters are guaranteed to be nullptr by the caller
template<class T>
void LoadFunction(void* handle, const char* name, T* nnrtFunction)
{
    if (name == nullptr) {
        NNRT_LOG("nnrt error: the function %s does not exist.", name);
        return;
    }

    void* fn = dlsym(handle, name);
    if (fn == nullptr) {
        NNRT_LOG("nnrt error: unable to open function %s", name);
        return;
    }

    *nnrtFunction = reinterpret_cast<T>(fn);
    return;
}

const NnrtApi LoadNnrt()
{
    NnrtApi nnrt;
    nnrt.nnrtExists = false;
    void* libNeuralNetworks = nullptr;

    // Assumes there can be multiple instances of NN API
    static std::string nnrtLibraryName = "libneural_network_runtime.z.so";
    libNeuralNetworks = dlopen(nnrtLibraryName.c_str(), RTLD_LAZY | RTLD_NODELETE);
    if (libNeuralNetworks == nullptr) {
        NNRT_LOG("nnrt error: unable to open library %s", nnrtLibraryName.c_str());
        return nnrt;
    } else {
        nnrt.nnrtExists = true;
    }

    // NNModel
    LoadFunction(libNeuralNetworks, "OH_NNModel_Construct", &nnrt.OH_NNModel_Construct);
    LoadFunction(libNeuralNetworks, "OH_NNModel_AddTensor", &nnrt.OH_NNModel_AddTensor);
    LoadFunction(libNeuralNetworks, "OH_NNModel_SetTensorData", &nnrt.OH_NNModel_SetTensorData);
    LoadFunction(libNeuralNetworks, "OH_NNModel_AddOperation", &nnrt.OH_NNModel_AddOperation);
    LoadFunction(libNeuralNetworks, "OH_NNModel_SpecifyInputsAndOutputs", &nnrt.OH_NNModel_SpecifyInputsAndOutputs);
    LoadFunction(libNeuralNetworks, "OH_NNModel_Finish", &nnrt.OH_NNModel_Finish);
    LoadFunction(libNeuralNetworks, "OH_NNModel_Destroy", &nnrt.OH_NNModel_Destroy);
    LoadFunction(libNeuralNetworks, "OH_NNModel_GetAvailableOperations", &nnrt.OH_NNModel_GetAvailableOperations);

    // NNCompilation
    LoadFunction(libNeuralNetworks, "OH_NNCompilation_Construct", &nnrt.OH_NNCompilation_Construct);
    LoadFunction(libNeuralNetworks, "OH_NNCompilation_SetDevice", &nnrt.OH_NNCompilation_SetDevice);
    LoadFunction(libNeuralNetworks, "OH_NNCompilation_SetCache", &nnrt.OH_NNCompilation_SetCache);
    LoadFunction(libNeuralNetworks, "OH_NNCompilation_SetPerformanceMode", &nnrt.OH_NNCompilation_SetPerformanceMode);
    LoadFunction(libNeuralNetworks, "OH_NNCompilation_SetPriority", &nnrt.OH_NNCompilation_SetPriority);
    LoadFunction(libNeuralNetworks, "OH_NNCompilation_EnableFloat16", &nnrt.OH_NNCompilation_EnableFloat16);
    LoadFunction(libNeuralNetworks, "OH_NNCompilation_Build", &nnrt.OH_NNCompilation_Build);
    LoadFunction(libNeuralNetworks, "OH_NNCompilation_Destroy", &nnrt.OH_NNCompilation_Destroy);

    // NNExecutor
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_Construct", &nnrt.OH_NNExecutor_Construct);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_SetInput", &nnrt.OH_NNExecutor_SetInput);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_SetOutput", &nnrt.OH_NNExecutor_SetOutput);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_GetOutputShape", &nnrt.OH_NNExecutor_GetOutputShape);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_Run", &nnrt.OH_NNExecutor_Run);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_AllocateInputMemory", &nnrt.OH_NNExecutor_AllocateInputMemory);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_AllocateOutputMemory", &nnrt.OH_NNExecutor_AllocateOutputMemory);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_DestroyInputMemory", &nnrt.OH_NNExecutor_DestroyInputMemory);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_DestroyOutputMemory", &nnrt.OH_NNExecutor_DestroyOutputMemory);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_SetInputWithMemory", &nnrt.OH_NNExecutor_SetInputWithMemory);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_SetOutputWithMemory", &nnrt.OH_NNExecutor_SetOutputWithMemory);
    LoadFunction(libNeuralNetworks, "OH_NNExecutor_Destroy", &nnrt.OH_NNExecutor_Destroy);

    // NNDevice
    LoadFunction(libNeuralNetworks, "OH_NNDevice_GetAllDevicesID", &nnrt.OH_NNDevice_GetAllDevicesID);
    LoadFunction(libNeuralNetworks, "OH_NNDevice_GetName", &nnrt.OH_NNDevice_GetName);
    LoadFunction(libNeuralNetworks, "OH_NNDevice_GetType", &nnrt.OH_NNDevice_GetType);

    return nnrt;
}

const NnrtApi* NnrtImplementation()
{
    static const NnrtApi nnrt = LoadNnrt();
    if (!nnrt.nnrtExists) {
        return nullptr;
    }
    return &nnrt;
}

} // namespace tflite
