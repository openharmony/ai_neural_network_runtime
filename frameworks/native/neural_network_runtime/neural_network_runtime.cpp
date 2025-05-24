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

#include "neural_network_runtime_inner.h"
#include "neural_network_runtime/neural_network_runtime.h"

#include "compilation.h"
#include "executor.h"
#include "inner_model.h"
#include "log.h"
#include "quant_param.h"
#include "validation.h"
#include "syspara/parameter.h"

#include <cstring>
#include <fstream>
#include <filesystem>
#include <sys/stat.h>
#include <unistd.h>
#include "nlohmann/json.hpp"
#include "securec.h"

using namespace OHOS::NeuralNetworkRuntime;

#define NNRT_API __attribute__((visibility("default")))

namespace {
const std::string EXTENSION_KEY_QUANT_BUFFER = "QuantBuffer";
const std::string EXTENSION_KEY_MODEL_NAME = "ModelName";
const std::string EXTENSION_KEY_IS_PROFILING = "isProfiling";
const std::string EXTENSION_KEY_OP_LAYOUT = "opLayout";
const std::string EXTENSION_KEY_INPUT_DIMS = "InputDims";
const std::string EXTENSION_KEY_DYNAMIC_DIMS = "DynamicDims";
const std::string EXTENSION_KEY_FM_SHARED = "NPU_FM_SHARED";
const std::string EXTENSION_KEY_IS_EXCEED_RAMLIMIT = "isExceedRamLimit";

const std::string NULL_HARDWARE_NAME = "default";
const std::string NNRT_DEVICE_NAME = "const.ai.nnrt_deivce";
const std::string HARDWARE_NAME = "ohos.boot.hardware";
const std::string HARDWARE_VERSION = "v5_0";
constexpr size_t HARDWARE_NAME_MAX_LENGTH = 128;
constexpr size_t FILE_NUMBER_MAX = 100; // 限制cache文件数量最大为100
constexpr size_t EXTENSION_MAX_SIZE = 200; // 限制MS传过来的参数最多为200
constexpr size_t INPUT_MAX_COUNT = 200; // 限制模型最大输入个数为200
constexpr int32_t HEX_UNIT = 16;
}

unsigned short CacheInfoGetCrc16(char* buffer, size_t length)
{
    unsigned int sum = 0;
    while (length > 1) {
        sum += *(reinterpret_cast<unsigned short*>(buffer));
        length -= sizeof(unsigned short);
        buffer += sizeof(unsigned short);
    }

    if (length > 0) {
        sum += *(reinterpret_cast<unsigned char*>(buffer));
    }

    while (sum >> HEX_UNIT) {
        sum = (sum >> HEX_UNIT) + (sum & 0xffff);
    }

    return static_cast<unsigned short>(~sum);
}

NNRT_API NN_QuantParam *OH_NNQuantParam_Create()
{
    auto* quantParamImpl = new (std::nothrow) QuantParams();
    if (quantParamImpl == nullptr) {
        LOGE("OH_NNQuantParam_Create failed, please check whether it has enough memory.");
        return nullptr;
    }

    return (NN_QuantParam*)(quantParamImpl);
}

NNRT_API OH_NN_ReturnCode OH_NNQuantParam_SetScales(NN_QuantParam* quantParams, const double* scales, size_t quantNum)
{
    if (quantParams == nullptr) {
        LOGE("OH_NNQuantParam_SetScales failed, passed nullptr to quantParams.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (scales == nullptr) {
        LOGE("OH_NNQuantParam_SetScales failed, passed nullptr to scales.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (quantNum == 0) {
        LOGE("OH_NNQuantParam_SetScales failed, passed 0 to quantNum.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto* quantParamImpl = reinterpret_cast<QuantParams*>(quantParams);
    std::vector<double> scaleVector(scales, scales + quantNum);
    quantParamImpl->SetScales(scaleVector);

    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNQuantParam_SetZeroPoints(NN_QuantParam* quantParams,
                                                        const int32_t* zeroPoints,
                                                        size_t quantNum)
{
    if (quantParams == nullptr) {
        LOGE("OH_NNQuantParam_SetZeroPoints failed, passed nullptr to quantParams.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (zeroPoints == nullptr) {
        LOGE("OH_NNQuantParam_SetZeroPoints failed, passed nullptr to zeroPoints.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (quantNum == 0) {
        LOGE("OH_NNQuantParam_SetZeroPoints failed, passed 0 to quantNum.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto* quantParamImpl = reinterpret_cast<QuantParams*>(quantParams);
    std::vector<int32_t> zeroPointVector(zeroPoints, zeroPoints + quantNum);
    quantParamImpl->SetZeroPoints(zeroPointVector);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNQuantParam_SetNumBits(NN_QuantParam* quantParams, const uint32_t* numBits, size_t quantNum)
{
    if (quantParams == nullptr) {
        LOGE("OH_NNQuantParam_SetNumBits failed, passed nullptr to quantParams.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (numBits == nullptr) {
        LOGE("OH_NNQuantParam_SetNumBits failed, passed nullptr to numBits.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (quantNum == 0) {
        LOGE("OH_NNQuantParam_SetNumBits failed, passed 0 to quantNum.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto* quantParamImpl = reinterpret_cast<QuantParams*>(quantParams);
    std::vector<uint32_t> numBitVector(numBits, numBits + quantNum);
    quantParamImpl->SetNumBits(numBitVector);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNQuantParam_Destroy(NN_QuantParam** quantParams)
{
    if (quantParams == nullptr) {
        LOGE("OH_NNQuantParam_Destroy failed, passed nullptr to quantParams.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (*quantParams == nullptr) {
        LOGW("OH_NNQuantParam_Destroy failed, passed nullptr to *quantParams.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto* quantParamImpl = reinterpret_cast<QuantParams*>(*quantParams);
    delete quantParamImpl;
    *quantParams = nullptr;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OH_NNModel_AddTensorToModel(OH_NNModel* model, const NN_TensorDesc* tensorDesc)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_AddTensorToModel failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (tensorDesc == nullptr) {
        LOGE("OH_NNModel_AddTensorToModel failed, passed nullptr to tensorDesc.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto* innerModel = reinterpret_cast<OHOS::NeuralNetworkRuntime::InnerModel*>(model);
    OH_NN_ReturnCode returnCode = innerModel->AddTensorDesc(tensorDesc);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("OH_NNModel_AddTensorToModel failed, error happened when adding tensor to model.");
    }

    return returnCode;
}

OH_NN_ReturnCode OH_NNModel_SetTensorQuantParams(OH_NNModel* model, uint32_t index, NN_QuantParam* quantParam)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_SetTensorQuantParams failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (quantParam == nullptr) {
        LOGE("OH_NNModel_SetTensorQuantParams failed, passed nullptr to quantParam.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto* innerModel = reinterpret_cast<OHOS::NeuralNetworkRuntime::InnerModel*>(model);
    OH_NN_ReturnCode returnCode = innerModel->SetTensorQuantParam((uint32_t)(index), quantParam);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("OH_NNModel_SetTensorQuantParams failed, error happened when setting tensor quantParam.");
    }

    return returnCode;
}

OH_NN_ReturnCode OH_NNModel_SetTensorType(OH_NNModel* model, uint32_t index, OH_NN_TensorType tensorType)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_SetTensorType failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!Validation::ValidateTensorType(tensorType)) {
        LOGE("OH_NNModel_SetTensorType failed, invalid tensor type.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto* innerModel = reinterpret_cast<OHOS::NeuralNetworkRuntime::InnerModel*>(model);
    OH_NN_ReturnCode returnCode = innerModel->SetTensorType((uint32_t)(index), tensorType);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("OH_NNModel_SetTensorType failed, error happened when setting tensor type.");
    }

    return returnCode;
}

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

OH_NN_ReturnCode ParseInputDimsFromExtensions(char* data, size_t dataSize, const mindspore::lite::LiteGraph* liteGraph,
    ExtensionConfig& extensionConfig, size_t& dynamicCount)
{
    extensionConfig.inputDims.clear();
    int32_t* dimsValue = reinterpret_cast<int32_t*>(data);
    size_t allDimsSize = dataSize / sizeof(int32_t);

    size_t inputCount = liteGraph->input_indices_.size(); // LiteGraph输入个数
    size_t allTensorSize = liteGraph->all_tensors_.size(); // LiteGraph所有tensor个数
    if (inputCount > INPUT_MAX_COUNT) {
        LOGE("ParseInputDimsFromExtensions failed, inputCount more than 200.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<int32_t> inputDim;
    size_t dataIndex = 0;
    for (size_t i = 0; i < inputCount; ++i) {
        inputDim.clear();
        if (liteGraph->input_indices_[i] >= allTensorSize) {
            LOGE("ParseInputDimsFromExtensions failed, indice of input %u is out of range.",
                liteGraph->input_indices_[i]);
            extensionConfig.inputDims.clear();
            return OH_NN_INVALID_PARAMETER;
        }
        //获取当前输入的维度
        mindspore::lite::TensorPtr tensor = liteGraph->all_tensors_[liteGraph->input_indices_[i]];
        auto tensorDims = mindspore::lite::MindIR_Tensor_GetDims(tensor);
        size_t inputDimSize = tensorDims.size();
        if (allDimsSize < inputDimSize) {
            LOGE("ParseInputDimsFromExtensions failed, dataSize is invalid.");
            extensionConfig.inputDims.clear();
            return OH_NN_INVALID_PARAMETER;
        }
        // 读取extensor中当前输入的dim值
        for (size_t j = 0; j < inputDimSize; ++j) {
            inputDim.emplace_back(dimsValue[dataIndex]);
            if (dimsValue[dataIndex] == -1) {
                ++dynamicCount;
            }
            ++dataIndex;
        }
        extensionConfig.inputDims.emplace_back(inputDim);
        allDimsSize -= inputDimSize;
    }
    // allDimsSize应和模型一致，遍历完后，allDimsSize等于0
    if (allDimsSize != 0) {
        LOGE("ParseInputDimsFromExtensions failed, allDimsSize is not equal to liteGraph.");
        extensionConfig.inputDims.clear();
        return OH_NN_INVALID_PARAMETER;
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ParseDynamicDimsFromExtensions(
    const std::unordered_map<std::string, std::vector<std::pair<char*, size_t>>>& extensionMap,
    const mindspore::lite::LiteGraph* liteGraph, ExtensionConfig& extensionConfig)
{
    const std::vector<std::pair<char*, size_t>>& inputDims = extensionMap.at(EXTENSION_KEY_INPUT_DIMS);
    if (inputDims.empty()) {
        LOGE("ParseDynamicDimsFromExtensions failed, input dims is empty.");
        return OH_NN_INVALID_PARAMETER;
    }
    auto dynamicDims = extensionMap.at(EXTENSION_KEY_DYNAMIC_DIMS);
    if (dynamicDims.empty()) {
        LOGE("ParseDynamicDimsFromExtensions failed, dynamic dims is empty.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (inputDims[0].first == nullptr || inputDims[0].second == 0 ||
        dynamicDims[0].first == nullptr || dynamicDims[0].second == 0) {
        LOGE("ParseDynamicDimsFromExtensions failed, data or dataSize is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }

    size_t dynamicCount = 0;
    auto returnCode = ParseInputDimsFromExtensions(
        inputDims[0].first, inputDims[0].second, liteGraph, extensionConfig, dynamicCount);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("ParseDynamicDimsFromExtensions failed, failed to get input dims from extensions.");
        return returnCode;
    }
    if (dynamicCount == 0) {
        LOGE("ParseDynamicDimsFromExtensions failed, dynamic count is 0.");
        extensionConfig.inputDims.clear();
        return OH_NN_INVALID_PARAMETER;
    }

    extensionConfig.dynamicDims.clear();
    int32_t* dynamicDimsValue = reinterpret_cast<int32_t*>(dynamicDims[0].first);
    size_t dynamicDimsSize = dynamicDims[0].second / sizeof(int32_t);
    if ((dynamicDimsSize % dynamicCount) != 0) {
        LOGE("ParseDynamicDimsFromExtensions failed, dynamic dataSize is invalid.");
        extensionConfig.inputDims.clear();
        return OH_NN_INVALID_PARAMETER;
    }
    size_t dynamicSize = dynamicDimsSize / dynamicCount;
    std::vector<int32_t> dynamicDim;
    size_t dataIndex = 0;
    for (size_t i = 0; i < dynamicSize; ++i) {
        dynamicDim.clear();
        for (size_t j = 0; j < dynamicCount; ++j) {
            dynamicDim.emplace_back(dynamicDimsValue[dataIndex]);
            ++dataIndex;
        }
        extensionConfig.dynamicDims.emplace_back(dynamicDim);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode CheckExtensionConfigs(
    const std::unordered_map<std::string, std::vector<std::pair<char*, size_t>>>& extensionMap,
    ExtensionConfig& extensionConfig)
{
    if (extensionMap.find(EXTENSION_KEY_QUANT_BUFFER) != extensionMap.end()) {
        const std::vector<std::pair<char*, size_t>>& value = extensionMap.at(EXTENSION_KEY_QUANT_BUFFER);
        if (value.empty()) {
            LOGE("ParseExtensionConfigs failed, get empty quant buffer value.");
            return OH_NN_INVALID_PARAMETER;
        }
        extensionConfig.quantBuffer.data = value[0].first;
        extensionConfig.quantBuffer.length = value[0].second;
    }
    if (extensionMap.find(EXTENSION_KEY_MODEL_NAME) != extensionMap.end()) {
        const std::vector<std::pair<char*, size_t>>& value = extensionMap.at(EXTENSION_KEY_MODEL_NAME);
        if (value.empty()) {
            LOGE("ParseExtensionConfigs failed, get empty model name value.");
            return OH_NN_INVALID_PARAMETER;
        }
        extensionConfig.modelName.assign(value[0].first, value[0].first + value[0].second);
    }
    if (extensionMap.find(EXTENSION_KEY_IS_PROFILING) != extensionMap.end()) {
        const std::vector<std::pair<char*, size_t>>& value = extensionMap.at(EXTENSION_KEY_IS_PROFILING);
        if (value.empty()) {
            LOGE("ParseExtensionConfigs failed, get empty isProfiling value.");
            return OH_NN_INVALID_PARAMETER;
        }
        extensionConfig.isProfiling.assign(value[0].first, value[0].first + value[0].second);
    }
    if (extensionMap.find(EXTENSION_KEY_OP_LAYOUT) != extensionMap.end()) {
        const std::vector<std::pair<char*, size_t>>& value = extensionMap.at(EXTENSION_KEY_OP_LAYOUT);
        if (value.empty()) {
            LOGE("ParseExtensionConfigs failed, get empty op layout value.");
            return OH_NN_INVALID_PARAMETER;
        }
        std::string ops;
        for (auto singleValue : value) {
            ops.assign(singleValue.first, singleValue.first + singleValue.second);
            extensionConfig.opLayout.insert({ops, "hiai::ExecuteDevice::CPU"});
            LOGI("ParseExtensionConfigs opLayout:%{public}s.", ops.c_str());
        }
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode ParseExtensionConfigs(
    const std::unordered_map<std::string, std::vector<std::pair<char*, size_t>>>& extensionMap,
    const mindspore::lite::LiteGraph* pLiteGraph, ExtensionConfig& extensionConfig)
{
    extensionConfig.tuningStrategy = TuningStrategy::ON_DEVICE_PREPROCESS_TUNING;
    OH_NN_ReturnCode ret = CheckExtensionConfigs(extensionMap, extensionConfig);
    if (ret != OH_NN_SUCCESS) {
        LOGE("CheckExtensionConfigs failed.");
        return ret;
    }
    if (extensionMap.find(EXTENSION_KEY_INPUT_DIMS) != extensionMap.end() &&
        extensionMap.find(EXTENSION_KEY_DYNAMIC_DIMS) != extensionMap.end()) {
        auto returnCode = ParseDynamicDimsFromExtensions(extensionMap, pLiteGraph, extensionConfig);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("ParseExtensionConfigs failed, parse dynamic dims from extensions failed.");
            return returnCode;
        }
        extensionConfig.tuningStrategy = TuningStrategy::OFF; // 分档shape不支持fftl
    }
    if (extensionMap.find(EXTENSION_KEY_FM_SHARED) != extensionMap.end()) {
        extensionConfig.isNpuFmShared = true;
        LOGI("NNRT enable fm shared success.");
    }
    return OH_NN_SUCCESS;
}

NNRT_API OH_NN_ReturnCode OH_NNModel_BuildFromLiteGraph(OH_NNModel *model, const void *liteGraph,
    const OH_NN_Extension *extensions, size_t extensionSize)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_BuildFromLiteGraph failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (liteGraph == nullptr) {
        LOGE("OH_NNModel_BuildFromLiteGraph failed, passed nullptr to liteGraph.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (extensionSize > EXTENSION_MAX_SIZE) {
        LOGE("OH_NNModel_BuildFromLiteGraph failed, extensionSize more than 200.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto *pLiteGraph = reinterpret_cast<const mindspore::lite::LiteGraph*>(liteGraph);
    ExtensionConfig extensionConfig;
    std::unordered_map<std::string, std::vector<std::pair<char*, size_t>>> extensionMap;
    for (size_t i = 0; i < extensionSize; ++i) {
        std::string name = extensions[i].name;
        if (extensionMap.find(name) == extensionMap.end()) {
            extensionMap.insert({name, {{extensions[i].value, extensions[i].valueSize}}});
        } else {
            extensionMap[name].push_back({extensions[i].value, extensions[i].valueSize});
        }
    }
    auto returnCode = ParseExtensionConfigs(extensionMap, pLiteGraph, extensionConfig);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("OH_NNModel_BuildFromLiteGraph failed, parse extension configs failed.");
        return returnCode;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);

    // Once the innerModel built from the liteGraph successfully, the innerModel
    // owns the liteGraph, in which case, the invoker should not delete
    // the liteGraph actively. Otherwise, the invoker still has the ownership.
    return innerModel->BuildFromLiteGraph(pLiteGraph, extensionConfig);
}

namespace {
OH_NN_ReturnCode CheckCacheFileExtension(const std::string& content, int64_t& fileNumber,
                                         int64_t& cacheVersion, int64_t& deviceId)
{
    if (!nlohmann::json::accept(content)) {
        LOGE("OH_NNModel_HasCache CheckCacheFile JSON parse error");
        return OH_NN_INVALID_FILE;
    }

    nlohmann::json j = nlohmann::json::parse(content);
    if (j.find("data") == j.end()) {
        LOGE("OH_NNModel_HasCache read data from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }

    if (j["data"].find("deviceId") == j["data"].end()) {
        LOGE("OH_NNModel_HasCache read deviceId from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }
    deviceId = j["data"]["deviceId"].get<int64_t>();

    if (j["data"].find("fileNumber") == j["data"].end()) {
        LOGE("OH_NNModel_HasCache read fileNumber from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }
    fileNumber = j["data"]["fileNumber"].get<int>();

    if (j["data"].find("version") == j["data"].end()) {
        LOGE("OH_NNModel_HasCache read version from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }
    cacheVersion = j["data"]["version"].get<int>();

    if (j.find("CheckSum") == j.end()) {
        LOGE("OH_NNModel_HasCache read CheckSum from cache info file failed.");
        return OH_NN_INVALID_FILE;
    }
    const size_t dataLength = j["data"].dump().length();
    char jData[dataLength + 1];

    if (strncpy_s(jData, dataLength+1, j["data"].dump().c_str(), dataLength) != 0) {
        LOGE("OH_NNModel_HasCache ParseStr failed due to strncpy_s error.");
        return OH_NN_INVALID_FILE;
    }

    if (static_cast<int64_t>(CacheInfoGetCrc16(jData, dataLength)) != j["CheckSum"].get<int64_t>()) {
        LOGE("OH_NNModel_HasCache cache_info CheckSum is not correct.");
        return OH_NN_INVALID_FILE;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode CheckCacheFile(const std::string& cacheInfoPath, int64_t& fileNumber,
                                int64_t& cacheVersion, int64_t& deviceId)
{
    char path[PATH_MAX];
    if (realpath(cacheInfoPath.c_str(), path) == nullptr) {
        LOGE("OH_NNModel_HasCache get real path of cache info failed.");
        return OH_NN_INVALID_FILE;
    }

    if (access(path, F_OK) != 0) {
        LOGE("OH_NNModel_HasCache access cache info file failed.");
        return OH_NN_INVALID_FILE;
    }

    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        LOGE("OH_NNModel_HasCache open cache info file failed.");
        return OH_NN_INVALID_FILE;
    }

    // Read the entire file into a string
    std::string content((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
    ifs.close();

    return CheckCacheFileExtension(content, fileNumber, cacheVersion, deviceId);
}

OH_NN_ReturnCode CheckDeviceId(int64_t& deviceId)
{
    std::string deviceName;
    char cName[HARDWARE_NAME_MAX_LENGTH];
    int ret = GetParameter(HARDWARE_NAME.c_str(), NULL_HARDWARE_NAME.c_str(), cName, HARDWARE_NAME_MAX_LENGTH);
    if (ret <= 0) {
        LOGE("OH_NNModel_HasCache failed to get parameter.");
        return OH_NN_FAILED;
    }

    deviceName = HARDWARE_NAME + "." + cName;
    if (deviceId != static_cast<int64_t>(std::hash<std::string>{}(deviceName))) {
        LOGE("OH_NNModel_HasCache the deviceId in the cache files is different from current deviceId.");
        return OH_NN_FAILED;
    }

    return OH_NN_SUCCESS;
}
}

NNRT_API bool OH_NNModel_HasCache(const char *cacheDir, const char *modelName, uint32_t version)
{
    if (cacheDir == nullptr) {
        LOGI("OH_NNModel_HasCache get empty cache directory.");
        return false;
    }

    if (modelName == nullptr) {
        LOGI("OH_NNModel_HasCache get empty model name.");
    }

    std::string cacheInfoPath = std::string(cacheDir) + "/" + std::string(modelName) + "cache_info.nncache";

    // determine whether cache info file exists
    struct stat buffer;
    bool exist = (stat(cacheInfoPath.c_str(), &buffer) == 0);
    if (!exist) {
        return false;
    }

    int64_t deviceId{0};
    int64_t fileNumber{0};
    int64_t cacheVersion{0};
    OH_NN_ReturnCode returnCode = CheckCacheFile(cacheInfoPath, fileNumber, cacheVersion, deviceId);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("OH_NNModel_HasCache get fileNumber or cacheVersion fail.");
        std::filesystem::remove_all(cacheInfoPath);
        return false;
    }

    returnCode = CheckDeviceId(deviceId);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("OH_NNModel_HasCache check deviceId fail.");
        std::filesystem::remove_all(cacheInfoPath);
        return false;
    }

    if (fileNumber <= 0 || static_cast<size_t>(fileNumber) > FILE_NUMBER_MAX) {
        LOGE("OH_NNModel_HasCache fileNumber is invalid or more than 100");
        std::filesystem::remove_all(cacheInfoPath);
        return false;
    }

    // determine whether cache model files exist
    for (int64_t i = 0; i < fileNumber; ++i) {
        std::string cacheModelPath =
            std::string(cacheDir) + "/" + std::string(modelName) + std::to_string(i) + ".nncache";
        exist = (exist && (stat(cacheModelPath.c_str(), &buffer) == 0));
        if (!exist) {
            LOGE("OH_NNModel_HasCache cacheModelPath is not existed.");
            std::filesystem::remove_all(cacheInfoPath);
            return false;
        }
    }

    if (cacheVersion != version) {
        LOGE("OH_NNModel_HasCache version is not match.");
        exist = false;
    }

    return exist;
}

NNRT_API OH_NN_ReturnCode OH_NNModel_BuildFromMetaGraph(OH_NNModel *model, const void *metaGraph,
    const OH_NN_Extension *extensions, size_t extensionSize)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_BuildFromMetaGraph failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (metaGraph == nullptr) {
        LOGE("OH_NNModel_BuildFromMetaGraph failed, passed nullptr to metaGraph.");
        return OH_NN_INVALID_PARAMETER;
    }

    ExtensionConfig extensionConfig;
    std::string ops;
    for (size_t i = 0; i < extensionSize; ++i) {
        std::string name = extensions[i].name;
        if (name == "QuantBuffer") {
            extensionConfig.quantBuffer.data = extensions[i].value;
            extensionConfig.quantBuffer.length = extensions[i].valueSize;
        } else if (name == "ModelName") {
            extensionConfig.modelName.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
        } else if (name == "Profiling") {
            extensionConfig.isProfiling.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
            LOGI("OH_NNModel_BuildFromMetaGraph isProfiling enable.");
        } else if (name == "opLayout") {
            ops.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
            extensionConfig.opLayout.insert({ops, "hiai::ExecuteDevice::CPU"});
            LOGI("OH_NNModel_BuildFromMetaGraph opLayout:%{public}s.", ops.c_str());
        }
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->BuildFromMetaGraph(metaGraph, extensionConfig);
}

NNRT_API OH_NN_ReturnCode OH_NNModel_SetInputsAndOutputsInfo(OH_NNModel *model, const OH_NN_TensorInfo *inputsInfo,
    size_t inputSize, const OH_NN_TensorInfo *outputsInfo, size_t outputSize)
{
    if (model == nullptr) {
        LOGE("OH_NNModel_SetInputsAndOutputsInfo failed, passed nullptr to model.");
        return OH_NN_INVALID_PARAMETER;
    }

    if ((inputsInfo == nullptr) || (inputSize == 0)) {
        LOGE("OH_NNModel_SetInputsAndOutputsInfo failed, inputsInfo is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    if ((outputsInfo == nullptr) || (outputSize == 0)) {
        LOGE("OH_NNModel_SetInputsAndOutputsInfo failed, outputsInfo is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->SetInputsAndOutputsInfo(inputsInfo, inputSize, outputsInfo, outputSize);
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

NNRT_API OH_NN_ReturnCode OH_NN_GetDeviceID(char *nnrtDevice, size_t len)
{
    if (nnrtDevice == nullptr || len == 0) {
        LOGE("nnrtDevice is nullptr or len is 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    char cName[HARDWARE_NAME_MAX_LENGTH] = {0};
    int ret = GetParameter(NNRT_DEVICE_NAME.c_str(), NULL_HARDWARE_NAME.c_str(), cName, HARDWARE_NAME_MAX_LENGTH);
    // 如果成功获取返回值为硬件名称的字节数
    if (ret <= 0) {
        LOGE("GetNNRtDeviceName failed, failed to get parameter.");
        return OH_NN_FAILED;
    }

    std::string deviceName = (std::string)cName + "_" + HARDWARE_VERSION;
    auto secureRet = strcpy_s(nnrtDevice, len, deviceName.c_str());
    if (secureRet != EOK) {
        LOGE("GetNNRtDeviceName failed, failed to get name.");
        return OH_NN_FAILED;
    }
    return OH_NN_SUCCESS;
}