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
#include "executor.h"
#include "inner_model.h"
#include "common/log.h"
#include "quant_param.h"
#include "validation.h"

using namespace OHOS::NeuralNetworkRuntime;

#define NNRT_API __attribute__((visibility("default")))

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

    Buffer buffer;
    std::string modelName;
    std::string isProfiling;
    std::string opLayout;
    std::map<std::string, std::string> opLayouts;
    for (size_t i = 0; i < extensionSize; ++i) {
        std::string name = extensions[i].name;
        if (name == "QuantBuffer") {
            buffer.data = extensions[i].value;
            buffer.length = extensions[i].valueSize;
        } else if (name == "ModelName") {
            modelName.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
        } else if (name == "Profiling") {
            isProfiling.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
            LOGI("OH_NNModel_BuildFromLiteGraph isProfiling enable.");
        } else if (name == "opLayout") {
            opLayout.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
            opLayouts.insert({opLayout, "hiai::ExecuteDevice::CPU"});
            LOGI("OH_NNModel_BuildFromLiteGraph opLayout:%{public}s.", opLayout.c_str());
        }
    }

    auto *pLiteGraph = static_cast<const mindspore::lite::LiteGraph*>(liteGraph);
    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    innerModel->SetTuningStrategy(TuningStrategy::ON_DEVICE_PREPROCESS_TUNING);

    // Once the innerModel built from the liteGraph successfully, the innerModel
    // owns the liteGraph, in which case, the invoker should not delete
    // the liteGraph actively. Otherwise, the invoker still has the ownership.
    return innerModel->BuildFromLiteGraph(pLiteGraph, buffer, modelName, isProfiling, opLayouts);
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

    Buffer buffer;
    std::string modelName;
    std::string isProfiling;
    std::string opLayout;
    std::map<std::string, std::string> opLayouts;
    for (size_t i = 0; i < extensionSize; ++i) {
        std::string name = extensions[i].name;
        if (name == "QuantBuffer") {
            buffer.data = extensions[i].value;
            buffer.length = extensions[i].valueSize;
        } else if (name == "ModelName") {
            modelName.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
        } else if (name == "Profiling") {
            isProfiling.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
            LOGI("OH_NNModel_BuildFromMetaGraph isProfiling enable.");
        } else if (name == "opLayout") {
            opLayout.assign(extensions[i].value, extensions[i].value + extensions[i].valueSize);
            opLayouts.insert({opLayout, "hiai::ExecuteDevice::CPU"});
            LOGI("OH_NNModel_BuildFromMetaGraph opLayout:%{public}s.", opLayout.c_str());
        }
    }

    InnerModel *innerModel = reinterpret_cast<InnerModel*>(model);
    return innerModel->BuildFromMetaGraph(metaGraph, buffer, modelName, isProfiling, opLayouts);
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
