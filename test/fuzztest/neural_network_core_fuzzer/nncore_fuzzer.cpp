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
#include "compilation.h"
#include "inner_model.h"
#include "neural_network_core.h"
#include <string>

namespace OHOS {
namespace NeuralNetworkRuntime {
const size_t SIZE_ONE = 1;
const size_t CACHE_VERSION = 1;
const size_t BUFFER_SIZE = 32;

// 返回值检查宏
#define CHECKNEQ(realRet, expectRet, retValue, ...) \
    do { \
        if ((realRet) != (expectRet)) { \
            printf(__VA_ARGS__); \
            return (retValue); \
        } \
    } while (0)

#define CHECKEQ(realRet, expectRet, retValue, ...) \
    do { \
        if ((realRet) == (expectRet)) { \
            printf(__VA_ARGS__); \
            return (retValue); \
        } \
    } while (0)

OH_NN_ReturnCode BuildModel(OH_NNModel** pmodel)
{
    // 创建模型实例model，进行模型构造
    OH_NNModel* model = OH_NNModel_Construct();
    CHECKEQ(model, nullptr, OH_NN_NULL_PTR, "Create model failed.");

    // 添加Add算子的第一个输入张量，类型为float32，张量形状为[1, 2, 2, 3]
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    CHECKEQ(tensorDesc, nullptr, OH_NN_NULL_PTR, "Create TensorDesc failed.");

    int32_t inputDims[4] = {1, 2, 2, 3};
    auto returnCode = OH_NNTensorDesc_SetShape(tensorDesc, inputDims, 4);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc shape failed.");

    returnCode = OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_FLOAT32);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc data type failed.");

    returnCode = OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NONE);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc format failed.");

    returnCode = OH_NNModel_AddTensorToModel(model, tensorDesc);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Add first TensorDesc to model failed.");

    returnCode = OH_NNModel_SetTensorType(model, 0, OH_NN_TENSOR);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set model tensor type failed.");

    // 添加Add算子的第二个输入张量，类型为float32，张量形状为[1, 2, 2, 3]
    tensorDesc = OH_NNTensorDesc_Create();
    CHECKEQ(tensorDesc, nullptr, OH_NN_NULL_PTR, "Create TensorDesc failed.");

    returnCode = OH_NNTensorDesc_SetShape(tensorDesc, inputDims, 4);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc shape failed.");

    returnCode = OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_FLOAT32);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc data type failed.");

    returnCode = OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NONE);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc format failed.");

    returnCode = OH_NNModel_AddTensorToModel(model, tensorDesc);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Add second TensorDesc to model failed.");

    returnCode = OH_NNModel_SetTensorType(model, 1, OH_NN_TENSOR);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set model tensor type failed.");

    // 添加Add算子的参数张量，该参数张量用于指定激活函数的类型，张量的数据类型为int8。
    tensorDesc = OH_NNTensorDesc_Create();
    CHECKEQ(tensorDesc, nullptr, OH_NN_NULL_PTR, "Create TensorDesc failed.");

    int32_t activationDims = 1;
    returnCode = OH_NNTensorDesc_SetShape(tensorDesc, &activationDims, 1);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc shape failed.");

    returnCode = OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_INT8);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc data type failed.");

    returnCode = OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NONE);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc format failed.");

    returnCode = OH_NNModel_AddTensorToModel(model, tensorDesc);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Add second TensorDesc to model failed.");

    returnCode = OH_NNModel_SetTensorType(model, 2, OH_NN_ADD_ACTIVATIONTYPE);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set model tensor type failed.");

    // 将激活函数类型设置为OH_NNBACKEND_FUSED_NONE，表示该算子不添加激活函数。
    int8_t activationValue = OH_NN_FUSED_NONE;
    returnCode = OH_NNModel_SetTensorData(model, 2, &activationValue, sizeof(int8_t));
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set model tensor data failed.");

    // 设置Add算子的输出张量，类型为float32，张量形状为[1, 2, 2, 3]
    tensorDesc = OH_NNTensorDesc_Create();
    CHECKEQ(tensorDesc, nullptr, OH_NN_NULL_PTR, "Create TensorDesc failed.");

    returnCode = OH_NNTensorDesc_SetShape(tensorDesc, inputDims, 4);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc shape failed.");

    returnCode = OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_FLOAT32);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc data type failed.");

    returnCode = OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NONE);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set TensorDesc format failed.");

    returnCode = OH_NNModel_AddTensorToModel(model, tensorDesc);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Add forth TensorDesc to model failed.");

    returnCode = OH_NNModel_SetTensorType(model, 3, OH_NN_TENSOR);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Set model tensor type failed.");

    // 指定Add算子的输入张量、参数张量和输出张量的索引
    uint32_t inputIndicesValues[2] = {0, 1};
    uint32_t paramIndicesValues = 2;
    uint32_t outputIndicesValues = 3;
    OH_NN_UInt32Array paramIndices = {&paramIndicesValues, 1 * 4};
    OH_NN_UInt32Array inputIndices = {inputIndicesValues, 2 * 4};
    OH_NN_UInt32Array outputIndices = {&outputIndicesValues, 1 * 4};

    // 向模型实例添加Add算子
    returnCode = OH_NNModel_AddOperation(model, OH_NN_OPS_ADD, &paramIndices, &inputIndices, &outputIndices);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Add operation to model failed.");

    // 设置模型实例的输入张量、输出张量的索引
    returnCode = OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Specify model inputs and outputs failed.");

    // 完成模型实例的构建
    returnCode = OH_NNModel_Finish(model);
    CHECKNEQ(returnCode, OH_NN_SUCCESS, returnCode, "Build model failed.");

    // 返回模型实例
    *pmodel = model;
    return OH_NN_SUCCESS;
}

void NNCoreDeviceFuzzTest(const uint8_t* data, size_t size)
{
    OH_NNDevice_GetAllDevicesID(nullptr, nullptr);

    const size_t* allDevicesID = new size_t[SIZE_ONE];
    OH_NNDevice_GetAllDevicesID(&allDevicesID, nullptr);
    delete[] allDevicesID;

    const size_t *allDevicesIDNull = nullptr;
    OH_NNDevice_GetAllDevicesID(&allDevicesIDNull, nullptr);

    uint32_t deviceCount = 0;
    OH_NNDevice_GetAllDevicesID(&allDevicesIDNull, &deviceCount);

    Data dataFuzz(data, size);
    size_t deviceid = dataFuzz.GetData<size_t>();
    const char* name = nullptr;
    OH_NNDevice_GetName(deviceid, &name);
    OH_NN_DeviceType deviceType;
    OH_NNDevice_GetType(deviceid, &deviceType);
}

bool NNCoreCompilationConstructTest(const uint8_t* data, size_t size)
{
    Data dataFuzz(data, size);
    InnerModel model = dataFuzz.GetData<InnerModel>();
    OH_NNCompilation_Construct(reinterpret_cast<OH_NNModel*>(&model));

    size_t bufferSize = BUFFER_SIZE;
    auto bufferAddr = dataFuzz.GetSpecificData(0, bufferSize);
    std::string path((char*)bufferAddr, (char*)bufferAddr + bufferSize);
    OH_NNCompilation_ConstructWithOfflineModelFile(path.c_str());

    OH_NNCompilation_ConstructWithOfflineModelBuffer(bufferAddr, bufferSize);

    Compilation compilation = dataFuzz.GetData<Compilation>();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    size_t modelSize = 0;
    char buffer[SIZE_ONE];
    OH_NNCompilation_ExportCacheToBuffer(nnCompilation, buffer, SIZE_ONE, &modelSize);

    OH_NNCompilation_ImportCacheFromBuffer(nnCompilation, buffer, SIZE_ONE);

    OH_NNCompilation_Build(nnCompilation);

    OH_NNCompilation_Destroy(&nnCompilation);

    OH_NNModel* validModel;
    if (BuildModel(&validModel) != OH_NN_SUCCESS) {
        LOGE("NNCoreCompilationConstructTest failed, build model failed.");
        return false;
    }
    OH_NNCompilation* validCompilation = OH_NNCompilation_Construct(validModel);
    OH_NNModel_Destroy(&validModel);
    if (validCompilation == nullptr) {
        LOGE("NNCoreCompilationConstructTest failed, construct valid compilation failed.");
        return false;
    }
    OH_NNCompilation_AddExtensionConfig(validCompilation, "test", bufferAddr, bufferSize);

    size_t deviceid = dataFuzz.GetData<size_t>();
    OH_NNCompilation_SetDevice(validCompilation, deviceid);

    OH_NNCompilation_SetCache(validCompilation, path.c_str(), CACHE_VERSION);

    OH_NN_PerformanceMode perf = dataFuzz.GetData<OH_NN_PerformanceMode>();
    OH_NNCompilation_SetPerformanceMode(validCompilation, perf);

    OH_NN_Priority priority = dataFuzz.GetData<OH_NN_Priority>();
    OH_NNCompilation_SetPriority(validCompilation, priority);

    bool enableFloat16 = dataFuzz.GetData<bool>();
    OH_NNCompilation_EnableFloat16(validCompilation, enableFloat16);
    OH_NNCompilation_Destroy(&validCompilation);

    return true;
}

bool NNCoreTensorDescFuzzTest(const uint8_t* data, size_t size)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    if (tensorDesc == nullptr) {
        LOGE("NNCoreTensorDescFuzzTest failed, create tensor desc failed.");
        return false;
    }

    Data dataFuzz(data, size);
    size_t bufferSize = BUFFER_SIZE;
    auto bufferAddr = dataFuzz.GetSpecificData(0, bufferSize);
    std::string path((char*)bufferAddr, (char*)bufferAddr + bufferSize);
    OH_NNTensorDesc_SetName(tensorDesc, path.c_str());
    const char* name = nullptr;
    OH_NNTensorDesc_GetName(tensorDesc, &name);

    OH_NN_DataType dataType = dataFuzz.GetData<OH_NN_DataType>();
    OH_NNTensorDesc_SetDataType(tensorDesc, dataType);
    OH_NN_DataType dataTypeOut;
    OH_NNTensorDesc_GetDataType(tensorDesc, &dataTypeOut);

    int32_t dim[SIZE_ONE] = {dataFuzz.GetData<int32_t>()};
    OH_NNTensorDesc_SetShape(tensorDesc, dim, SIZE_ONE);
    int32_t* shape = nullptr;
    size_t shapeLength = 0;
    OH_NNTensorDesc_GetShape(tensorDesc, &shape, &shapeLength);

    OH_NN_Format format = dataFuzz.GetData<OH_NN_Format>();
    OH_NNTensorDesc_SetFormat(tensorDesc, format);
    OH_NN_Format formatOut;
    OH_NNTensorDesc_GetFormat(tensorDesc, &formatOut);

    size_t elementSize = 0;
    OH_NNTensorDesc_GetElementCount(tensorDesc, &elementSize);
    size_t byteSize = 0;
    OH_NNTensorDesc_GetByteSize(tensorDesc, &byteSize);

    OH_NNTensorDesc_Destroy(&tensorDesc);

    return true;
}

bool NNCoreFuzzTest(const uint8_t* data, size_t size)
{
    bool ret = true;
    NNCoreDeviceFuzzTest(data, size);
    if (!NNCoreCompilationConstructTest(data, size)) {
        ret = false;
    }
    if (!NNCoreTensorDescFuzzTest(data, size)) {
        ret = false;
    }
    if (!NNCoreTensorFuzzTest(data, size)) {
        ret = false;
    }
    if (!NNCoreExecutorFuzzTest(data, size)) {
        ret = false;
    }

    return ret;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS

/* Fuzzer entry point */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    return OHOS::NeuralNetworkRuntime::NNCoreFuzzTest(data, size);
}