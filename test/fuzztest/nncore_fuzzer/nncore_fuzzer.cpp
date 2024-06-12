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
const size_t SHAPE_LENTH = 4;

struct OHNNOperandTest {
    OH_NN_DataType dataType;
    OH_NN_TensorType type;
    std::vector<int32_t> shape;
    void *data {nullptr};
    int32_t length {0};
    OH_NN_Format format = OH_NN_FORMAT_NONE;
    const OH_NN_QuantParam *quantParam = nullptr;
};

struct OHNNGraphArgs {
    OH_NN_OperationType operationType;
    std::vector<OHNNOperandTest> operands;
    std::vector<uint32_t> paramIndices;
    std::vector<uint32_t> inputIndices;
    std::vector<uint32_t> outputIndices;
    bool build = true;
    bool specifyIO = true;
    bool addOperation = true;
};

struct Model0 {
    float value = 1;
    OHNNOperandTest input = {OH_NN_FLOAT32, OH_NN_TENSOR, {1}, &value, sizeof(float)};
    OHNNOperandTest output = {OH_NN_FLOAT32, OH_NN_TENSOR, {1}, &value, sizeof(float)};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_ADD,
                               .operands = {input, output},
                               .paramIndices = {},
                               .inputIndices = {0},
                               .outputIndices = {1}};
};

OH_NN_UInt32Array TransformUInt32Array(const std::vector<uint32_t>& vector)
{
    uint32_t* data = (vector.empty()) ? nullptr : const_cast<uint32_t*>(vector.data());
    return {data, vector.size()};
}

NN_TensorDesc* createTensorDesc(const int32_t* shape, size_t shapeNum, OH_NN_DataType dataType, OH_NN_Format format)
{
    NN_TensorDesc* tensorDescTmp = OH_NNTensorDesc_Create();
    if (tensorDescTmp == nullptr) {
        LOGE("[NncoreFuzzTest]OH_NNTensorDesc_Create failed!");
        return nullptr;
    }

    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetDataType(tensorDescTmp, dataType);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NncoreFuzzTest]OH_NNTensorDesc_SetDataType failed!ret = %d\n", ret);
        return nullptr;
    }

    if (shape != nullptr) {
        ret = OH_NNTensorDesc_SetShape(tensorDescTmp, shape, shapeNum);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NncoreFuzzTest]OH_NNTensorDesc_SetShape failed!ret = %d\n", ret);
            return nullptr;
        }
    }

    ret = OH_NNTensorDesc_SetFormat(tensorDescTmp, format);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NncoreFuzzTest]OH_NNTensorDesc_SetShape failed!ret = %d\n", ret);
        return nullptr;
    }

    return tensorDescTmp;
}

int SingleModelBuildEndStep(OH_NNModel *model, const OHNNGraphArgs &graphArgs)
{
    int ret = 0;
    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);

    if (graphArgs.addOperation) {
        ret = OH_NNModel_AddOperation(model, graphArgs.operationType, &paramIndices, &inputIndices,
                                      &outputIndices);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NncoreFuzzTest] OH_NNModel_AddOperation failed! ret=%{public}d\n", ret);
            return ret;
        }
    }

    if (graphArgs.specifyIO) {
        ret = OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NncoreFuzzTest] OH_NNModel_SpecifyInputsAndOutputs failed! ret=%{public}d\n", ret);
            return ret;
        }
    }

    if (graphArgs.build) {
        ret = OH_NNModel_Finish(model);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NncoreFuzzTest] OH_NNModel_Finish failed! ret=%d\n", ret);
            return ret;
        }
    }
    return ret;
}

int BuildSingleOpGraph(OH_NNModel *model, const OHNNGraphArgs &graphArgs)
{
    int ret = 0;
    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        NN_TensorDesc* tensorDesc = createTensorDesc(operandTem.shape.data(),
                                                     (uint32_t) operandTem.shape.size(),
                                                     operandTem.dataType, operandTem.format);

        ret = OH_NNModel_AddTensorToModel(model, tensorDesc);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NncoreFuzzTest] OH_NNModel_AddTensor failed! ret=%d\n", ret);
            return ret;
        }

        ret = OH_NNModel_SetTensorType(model, i, operandTem.type);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NncoreFuzzTest] OH_NNBackend_SetModelTensorType failed! ret=%d\n", ret);
            return ret;
        }

        if (std::find(graphArgs.paramIndices.begin(), graphArgs.paramIndices.end(), i) !=
            graphArgs.paramIndices.end()) {
            ret = OH_NNModel_SetTensorData(model, i, operandTem.data, operandTem.length);
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NncoreFuzzTest] OH_NNModel_SetTensorData failed! ret=%{public}d\n", ret);
                return ret;
            }
        }
        OH_NNTensorDesc_Destroy(&tensorDesc);
    }
    ret = SingleModelBuildEndStep(model, graphArgs);
    return ret;
}

OH_NNModel* buildModel0(uint32_t opsType)
{
    OH_NNModel *model = OH_NNModel_Construct();

    Model0 model0;
    OHNNGraphArgs graphArgs = model0.graphArgs;
    graphArgs.operationType = static_cast<OH_NN_OperationType>(opsType);
    BuildSingleOpGraph(model, graphArgs);
    return model;
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
    uint32_t opsType = dataFuzz.GetData<uint32_t>()
        % (OH_NN_OPS_GATHER_ND - OH_NN_OPS_ADD + 1);
    OH_NNModel* model = buildModel0(opsType);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);

    size_t bufferSize = BUFFER_SIZE;
    auto bufferAddr = dataFuzz.GetSpecificData(0, bufferSize);
    std::string path((char*)bufferAddr, (char*)bufferAddr + bufferSize);
    OH_NNCompilation_ConstructWithOfflineModelFile(path.c_str());

    OH_NNCompilation_ConstructWithOfflineModelBuffer(bufferAddr, bufferSize);

    size_t modelSize = 0;
    char buffer[SIZE_ONE];
    OH_NNCompilation_ExportCacheToBuffer(nnCompilation, buffer, SIZE_ONE, &modelSize);

    OH_NNCompilation_ImportCacheFromBuffer(nnCompilation, buffer, SIZE_ONE);

    OH_NNCompilation_Build(nnCompilation);

    OH_NNCompilation_Destroy(&nnCompilation);

    OH_NNCompilation* validCompilation = OH_NNCompilation_Construct(model);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_AddExtensionConfig(validCompilation, "test", bufferAddr, bufferSize);

    size_t deviceid = dataFuzz.GetData<size_t>();
    OH_NNCompilation_SetDevice(validCompilation, deviceid);

    OH_NNCompilation_SetCache(validCompilation, path.c_str(), CACHE_VERSION);

    OH_NN_PerformanceMode perf = static_cast<OH_NN_PerformanceMode>(
        dataFuzz.GetData<uint32_t>() % (OH_NN_PERFORMANCE_EXTREME - OH_NN_PERFORMANCE_NONE + 1));
    OH_NNCompilation_SetPerformanceMode(validCompilation, perf);

    OH_NN_Priority priority = static_cast<OH_NN_Priority>(
        dataFuzz.GetData<uint32_t>() % (OH_NN_PRIORITY_HIGH - OH_NN_PRIORITY_NONE + 1));
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

    OH_NN_DataType dataType = static_cast<OH_NN_DataType>(
        dataFuzz.GetData<uint32_t>() % (OH_NN_FLOAT64 - OH_NN_UNKNOWN + 1));
    OH_NNTensorDesc_SetDataType(tensorDesc, dataType);
    OH_NN_DataType dataTypeOut;
    OH_NNTensorDesc_GetDataType(tensorDesc, &dataTypeOut);

    int32_t dim[SIZE_ONE] = {dataFuzz.GetData<int32_t>()};
    OH_NNTensorDesc_SetShape(tensorDesc, dim, SIZE_ONE);
    int32_t* shape = nullptr;
    size_t shapeLength = 0;
    OH_NNTensorDesc_GetShape(tensorDesc, &shape, &shapeLength);

    OH_NN_Format format = static_cast<OH_NN_Format>(
        dataFuzz.GetData<uint32_t>() % (OH_NN_FORMAT_ND - OH_NN_FORMAT_NONE + 1));
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

bool NNCoreTensorFuzzTest(const uint8_t* data, size_t size)
{
    Data dataFuzz(data, size);
    size_t deviceId = dataFuzz.GetData<size_t>();
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    int32_t inputDims[4] = {1, 2, 2, 3};
    OH_NNModel* model = OH_NNModel_Construct();
    OH_NNTensorDesc_SetShape(tensorDesc, inputDims, SHAPE_LENTH);
    OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_FLOAT32);
    OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NONE);
    OH_NNModel_AddTensorToModel(model, tensorDesc);
    OH_NNModel_SetTensorType(model, 0, OH_NN_TENSOR);
    NN_Tensor* nnTensor = OH_NNTensor_Create(deviceId, tensorDesc);
    
    size_t tensorSize = dataFuzz.GetData<size_t>();
    nnTensor = OH_NNTensor_CreateWithSize(deviceId, tensorDesc, tensorSize);

    int fd = dataFuzz.GetData<int>();
    size_t offset = dataFuzz.GetData<size_t>();
    nnTensor = OH_NNTensor_CreateWithFd(deviceId, tensorDesc, fd, tensorSize, offset);

    OH_NNTensor_GetTensorDesc(nnTensor);

    OH_NNTensor_GetDataBuffer(nnTensor);

    OH_NNTensor_GetFd(nnTensor, &fd);

    OH_NNTensor_GetSize(nnTensor, &tensorSize);

    OH_NNTensor_GetOffset(nnTensor, &offset);

    OH_NNTensor_Destroy(&nnTensor);
    return true;
}

bool NNCoreExecutorFuzzTest(const uint8_t* data, size_t size)
{
    Data dataFuzz(data, size);
    uint32_t opsType = dataFuzz.GetData<uint32_t>()
        % (OH_NN_OPS_GATHER_ND - OH_NN_OPS_ADD + 1);
    OH_NNModel* model = buildModel0(opsType);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    size_t deviceid = dataFuzz.GetData<size_t>();
    OH_NNCompilation_SetDevice(nnCompilation, deviceid);
    OH_NNCompilation_Build(nnCompilation);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

    uint32_t outputIndex = dataFuzz.GetData<uint32_t>();
    int32_t *shape = nullptr;
    uint32_t shapeLenth = 0;
    OH_NNExecutor_GetOutputShape(nnExecutor, outputIndex, &shape, &shapeLenth);

    size_t inputCount = 0;
    OH_NNExecutor_GetInputCount(nnExecutor, &inputCount);

    size_t outputCount = 0;
    OH_NNExecutor_GetOutputCount(nnExecutor, &outputCount);
    std::vector<NN_TensorDesc*> inputTensorDescs;
    std::vector<NN_TensorDesc*> outputTensorDescs;
    size_t index = 0;
    for (size_t i = 0; i < inputCount; i++) {
        index = dataFuzz.GetData<size_t>() % inputCount;
        NN_TensorDesc* nnTensorDesc = OH_NNExecutor_CreateInputTensorDesc(nnExecutor, index);
        inputTensorDescs.emplace_back(nnTensorDesc);
    }
    for (size_t i = 0; i < outputCount; i++) {
        index = dataFuzz.GetData<size_t>() % outputCount;
        NN_TensorDesc* nnTensorDesc = OH_NNExecutor_CreateOutputTensorDesc(nnExecutor, index);
        outputTensorDescs.emplace_back(nnTensorDesc);
    }

    size_t *minInputDims = nullptr;
    size_t *maxInputDIms = nullptr;
    size_t shapeLength = 0;
    OH_NNExecutor_GetInputDimRange(nnExecutor, index, &minInputDims, &maxInputDIms, &shapeLength);

    std::vector<NN_Tensor*> inputTensors;
    std::vector<NN_Tensor*> outputTensors;
    for (size_t i = 0; i < inputCount; ++i) {
        NN_Tensor* tensor = OH_NNTensor_Create(deviceid, inputTensorDescs[i]);
        inputTensors.emplace_back(tensor);
    }
    for (size_t i = 0; i < outputCount; ++i) {
        NN_Tensor* tensor = OH_NNTensor_Create(deviceid, outputTensorDescs[i]);
        outputTensors.emplace_back(tensor);
    }
    void* userData = dataFuzz.GetData<void*>();
    OH_NNExecutor_RunSync(nnExecutor, inputTensors.data(), inputCount, outputTensors.data(), outputCount);

    int32_t timeout = dataFuzz.GetData<int32_t>();
    OH_NNExecutor_RunAsync(nnExecutor, inputTensors.data(), inputCount, outputTensors.data(),
                           outputCount, timeout, userData);

    OH_NNExecutor_Destroy(&nnExecutor);
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