/*
 * Copyright (C) 2024 Huawei Device Co., Ltd.
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
#include "nncore_fuzzer.h"
#include "../data.h"
#include "../../../common/log.h"
#include "neural_network_core.h"
#include <string>

namespace OHOS {
namespace NeuralNetworkRuntime {
bool NNCoreDeviceFuzzTest()
{
    auto ret = OH_NNDevice_GetAllDevicesID(nullptr, nullptr);
    if (ret != OH_NN_INVALID_PARAMETER) {
        LOGE("[NNCoreFuzzTest]OH_NNDevice_GetAllDevicesID should return OH_NN_INVALID_PARAMETER.");
        return false;
    }

    const size_t* allDevicesID = new size_t[1];
    ret = OH_NNDevice_GetAllDevicesID(&allDevicesID, nullptr);
    if (ret != OH_NN_INVALID_PARAMETER) {
        LOGE("[NNCoreFuzzTest]OH_NNDevice_GetAllDevicesID with allDevicesID should return OH_NN_INVALID_PARAMETER.");
        delete[] allDevicesID;
        return false;
    }
    delete[] allDevicesID;

    const size_t *allDevicesIDNull = nullptr;
    ret = OH_NNDevice_GetAllDevicesID(&allDevicesIDNull, nullptr);
    if (ret != OH_NN_INVALID_PARAMETER) {
        LOGE("[NNCoreFuzzTest]OH_NNDevice_GetAllDevicesID with null deviceCount should return OH_NN_INVALID_PARAMETER.");
        return false;
    }

    uint32_t deviceCount = 0;
    ret = OH_NNDevice_GetAllDevicesID(&allDevicesIDNull, &deviceCount);

    ret = OH_NNDevice_GetName(0, nullptr);
    if (ret != OH_NN_INVALID_PARAMETER) {
        LOGE("[NNCoreFuzzTest]OH_NNDevice_GetName with null name should return OH_NN_INVALID_PARAMETER.");
        return false;
    }

    std::string name = "test";
    const char* nameC = name.c_str();
    ret = OH_NNDevice_GetName(0, &nameC);
    if (ret != OH_NN_INVALID_PARAMETER) {
        LOGE("[NNCoreFuzzTest]OH_NNDevice_GetName with invalid name should return OH_NN_INVALID_PARAMETER.");
        return false;
    }

    const char* nnameNullC = nullptr;
    ret = OH_NNDevice_GetName(0, &nnameNullC);
    if (ret != OH_NN_FAILED) {
        LOGE("[NNCoreFuzzTest]OH_NNDevice_GetName with invalid deviceid should return OH_NN_FAILED.");
        return false;
    }

    ret = OH_NNDevice_GetType(0, nullptr);
    if (ret != OH_NN_INVALID_PARAMETER) {
        LOGE("[NNCoreFuzzTest]OH_NNDevice_GetType with invalid device id should return OH_NN_INVALID_PARAMETER.");
        return false;
    }
    return true;
}

bool NNCoreCompilationConstructTest()
{
    auto ret = OH_NNCompilation_Construct(nullptr);
    if (ret != nullptr) {
        LOGE("[NNCoreFuzzTest]OH_NNCompilation_Construct with nullptr should return nullptr.");
        return false;
    }
    return true;
}

bool NNCoreFuzzTest(const uint8_t* data, size_t size)
{
    if (!NNCoreDeviceFuzzTest()) {
        return false;
    }

    return true;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS

/* Fuzzer entry point */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    OHOS::NeuralNetworkRuntime::NNCoreFuzzTest(data, size);
    return 0;
}