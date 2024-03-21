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

#include "hdi_device_v2_1.h"

#include "hdf_base.h"
#include "mindir.h"
#include "securec.h"

#include "hdi_prepared_model_v2_1.h"
#include "lite_graph_to_hdi_model_v2_1.h"
#include "hdi_returncode_utils_v2_1.h"
#include "memory_manager.h"
#include "transform.h"
#include "common/log.h"
#include "common/utils.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
const size_t OFFLINE_MODEL_MINIMUM_INPUT_SIZE = 2;

namespace {
OH_NN_DeviceType TransHDIDeviceV2_1Type(const V2_1::DeviceType& iDeviceType)
{
    switch (iDeviceType) {
        case V2_1::DeviceType::CPU:
            return OH_NN_CPU;
        case V2_1::DeviceType::GPU:
            return OH_NN_GPU;
        case V2_1::DeviceType::ACCELERATOR:
            return OH_NN_ACCELERATOR;
        default:
            return OH_NN_OTHERS;
    }
}

DeviceStatus TransHDIDeviceV2_1Status(const V2_1::DeviceStatus& iDeviceStatus)
{
    switch (iDeviceStatus) {
        case V2_1::DeviceStatus::AVAILABLE:
            return DeviceStatus::AVAILABLE;
        case V2_1::DeviceStatus::BUSY:
            return DeviceStatus::BUSY;
        case V2_1::DeviceStatus::OFFLINE:
            return DeviceStatus::OFFLINE;
        default:
            return DeviceStatus::UNKNOWN;
    }
}

V2_1::PerformanceMode TransPerformanceMode(const OH_NN_PerformanceMode& mode)
{
    switch (mode) {
        case OH_NN_PERFORMANCE_LOW:
            return V2_1::PerformanceMode::PERFORMANCE_LOW;
        case OH_NN_PERFORMANCE_MEDIUM:
            return V2_1::PerformanceMode::PERFORMANCE_MEDIUM;
        case OH_NN_PERFORMANCE_HIGH:
            return V2_1::PerformanceMode::PERFORMANCE_HIGH;
        case OH_NN_PERFORMANCE_EXTREME:
            return V2_1::PerformanceMode::PERFORMANCE_EXTREME;
        default:
            return V2_1::PerformanceMode::PERFORMANCE_NONE;
    }
}

V2_1::Priority TransPriority(const OH_NN_Priority& priority)
{
    switch (priority) {
        case OH_NN_PRIORITY_LOW:
            return V2_1::Priority::PRIORITY_LOW;
        case OH_NN_PRIORITY_MEDIUM:
            return V2_1::Priority::PRIORITY_MEDIUM;
        case OH_NN_PRIORITY_HIGH:
            return V2_1::Priority::PRIORITY_HIGH;
        default:
            return V2_1::Priority::PRIORITY_NONE;
    }
}

OH_NN_ReturnCode IsOfflineModel(std::shared_ptr<const mindspore::lite::LiteGraph> liteGraph, bool& isOfflineModel)
{
    isOfflineModel = false; // Initialize the returned value
    if (liteGraph == nullptr) {
        LOGE("LiteGraph is empty when identifying the offline model.");
        return OH_NN_NULL_PTR;
    }

    if (liteGraph->all_nodes_.size() == 0) {
        LOGE("Find empty node in the model.");
        return OH_NN_INVALID_PARAMETER;
    }

    // If the model consists of more than 1 node, it will not be considered as offline model.
    if (liteGraph->all_nodes_.size() > 1) {
        isOfflineModel = false;
        return OH_NN_SUCCESS;
    }

    const mindspore::lite::LiteGraph::Node* pNode = liteGraph->all_nodes_[0];
    if (pNode == nullptr) {
        LOGE("Find invalid node in the model.");
        return OH_NN_NULL_PTR;
    }

    const mindspore::lite::NodeType& nodeType = mindspore::lite::MindIR_Primitive_GetType(pNode->primitive_);
    if (nodeType == mindspore::lite::NodeType::NODE_TYPE_CUSTOM) {
        isOfflineModel = true;
    }

    return OH_NN_SUCCESS;
}
}  // unamed namespace

HDIDeviceV2_1::HDIDeviceV2_1(OHOS::sptr<V2_1::INnrtDevice> device) : m_iDevice(device)
{}

OH_NN_ReturnCode HDIDeviceV2_1::GetDeviceName(std::string& name)
{
    auto ret = m_iDevice->GetDeviceName(name);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Get HDI device name failed");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::GetVendorName(std::string& name)
{
    auto ret = m_iDevice->GetVendorName(name);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Get HDI vendor name failed");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::GetVersion(std::string& version)
{
    auto ret = m_iDevice->GetVersion(m_hdiVersion.first, m_hdiVersion.second);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Get HDI version failed");
    }
    version = 'v' + std::to_string(m_hdiVersion.first) + '_' + std::to_string(m_hdiVersion.second);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::GetDeviceType(OH_NN_DeviceType& deviceType)
{
    V2_1::DeviceType iDeviceType;
    auto ret = m_iDevice->GetDeviceType(iDeviceType);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Get HDI device type failed");
    }

    deviceType = TransHDIDeviceV2_1Type(iDeviceType);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::GetDeviceStatus(DeviceStatus& status)
{
    V2_1::DeviceStatus iDeviceStatus;
    auto ret = m_iDevice->GetDeviceStatus(iDeviceStatus);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Get HDI device status failed");
    }
    status = TransHDIDeviceV2_1Status(iDeviceStatus);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
    std::vector<bool>& ops)
{
    if (model == nullptr) {
        LOGE("Model is nullptr, cannot query supported operation.");
        return OH_NN_NULL_PTR;
    }

    bool isOfflineModel {false};
    OH_NN_ReturnCode innerRet = IsOfflineModel(model, isOfflineModel);
    if (innerRet != OH_NN_SUCCESS) {
        LOGE("Check offline model failed.");
        return innerRet;
    }

    // Permanently return a [true] array for offline model.
    if (isOfflineModel) {
        ops.clear();
        ops.emplace_back(true);
        return OH_NN_SUCCESS;
    }

    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {INVALID_FD, 0, 0, 0};
    size_t tensorSize = mindspore::lite::MindIR_LiteGraph_GetConstTensorSize(model.get());
    int32_t ret {0};
    if (tensorSize > 0) {
        ret = m_iDevice->AllocateBuffer(tensorSize, tensorBuffer);
        if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS || tensorBuffer.fd == INVALID_FD) {
            return CheckReturnCode_V2_1(ret, OH_NN_FAILED, "Allocate tensor buffer error when get supported operation");
        }
    }

    auto iModel = NNRt_V2_1::LiteGraph_To_HDIModel(model.get(), tensorBuffer);
    if (iModel == nullptr) {
        LOGE("Parse litegraph to hdi model failed.");
        ReleaseSharedBuffer(tensorBuffer);
        return OH_NN_FAILED;
    }

    ret = m_iDevice->GetSupportedOperation(*iModel, ops);

    NNRt_V2_1::HDIModel_Destroy(&iModel);
    innerRet = ReleaseSharedBuffer(tensorBuffer);
    if (innerRet != OH_NN_SUCCESS) {
        LOGE("Release tensorBuffer failed.");
        return OH_NN_FAILED;
    }
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Get supported operation failed");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::IsFloat16PrecisionSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsFloat16PrecisionSupported(isSupported);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Query fp16 precision supported failed");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::IsPerformanceModeSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsPerformanceModeSupported(isSupported);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Query performance mode supported failed");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::IsPrioritySupported(bool& isSupported)
{
    auto ret = m_iDevice->IsPrioritySupported(isSupported);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Query priority supported failed");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::IsDynamicInputSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsDynamicInputSupported(isSupported);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Query dynamic input supported failed");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::IsModelCacheSupported(bool& isSupported)
{
    auto ret = m_iDevice->IsModelCacheSupported(isSupported);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Query cache model supported failed");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
    const Buffer& quantBuffer, const ModelConfig& config, std::shared_ptr<PreparedModel>& preparedModel)
{
    if (model == nullptr) {
        LOGE("Model is nullptr, cannot prepare model.");
        return OH_NN_INVALID_PARAMETER;
    }

    OHOS::HDI::Nnrt::V2_1::SharedBuffer tensorBuffer {INVALID_FD, 0, 0, 0};
    size_t tensorSize = mindspore::lite::MindIR_LiteGraph_GetConstTensorSize(model.get());
    int32_t ret {0};
    if (tensorSize > 0) {
        ret = m_iDevice->AllocateBuffer(tensorSize, tensorBuffer);
        if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS || tensorBuffer.fd == INVALID_FD) {
            return CheckReturnCode_V2_1(ret, OH_NN_FAILED, "Allocate tensor buffer error when prepare model");
        }
    }

    V2_1::Model* iModel = NNRt_V2_1::LiteGraph_To_HDIModel(model.get(), tensorBuffer);
    if (iModel == nullptr) {
        LOGE("Parse litegraph to hdi model failed.");
        ReleaseSharedBuffer(tensorBuffer);
        return OH_NN_FAILED;
    }

    V2_1::ModelConfig iModelConfig;
    iModelConfig.enableFloat16 = config.enableFloat16;
    iModelConfig.mode = TransPerformanceMode(config.mode);
    iModelConfig.priority = TransPriority(config.priority);
    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel;

    ret = m_iDevice->PrepareModel(*iModel, iModelConfig, iPreparedModel);

    NNRt_V2_1::HDIModel_Destroy(&iModel);
    auto innerRet = ReleaseSharedBuffer(tensorBuffer);
    if (innerRet != OH_NN_SUCCESS) {
        LOGE("Release tensorBuffer failed.");
        return OH_NN_FAILED;
    }
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS || iPreparedModel == nullptr) {
        return CheckReturnCode_V2_1(ret, OH_NN_FAILED, "Prepare model failed");
    }

    preparedModel = CreateSharedPtr<HDIPreparedModelV2_1>(iPreparedModel);
    if (preparedModel == nullptr) {
        LOGE("Prepare model failed, because fail to create preparedModel instance.");
        return OH_NN_MEMORY_ERROR;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::PrepareModel(const void* metaGraph,
                                             const Buffer& quantBuffer,
                                             const ModelConfig& config,
                                             std::shared_ptr<PreparedModel>& preparedModel)
{
    return OH_NN_OPERATION_FORBIDDEN;
}

OH_NN_ReturnCode HDIDeviceV2_1::PrepareModelFromModelCache(const std::vector<Buffer>& modelCache,
    const ModelConfig& config, std::shared_ptr<PreparedModel>& preparedModel)
{
    std::vector<V2_1::SharedBuffer> iBuffers;
    auto memManager = MemoryManager::GetInstance();
    Memory memory;
    OH_NN_ReturnCode ret;
    size_t modelCacheSize = modelCache.size();
    for (size_t i = 0; i < modelCacheSize; i++) {
        ret = memManager->GetMemory(modelCache[i].data, memory);
        if (ret != OH_NN_SUCCESS) {
            LOGE("The %{public}zuth model cache is invalid. Please put valid model cache.", i + 1);
            return ret;
        }
        iBuffers.emplace_back(V2_1::SharedBuffer {memory.fd, memory.length, 0, memory.length});
    }

    V2_1::ModelConfig iModelConfig;
    iModelConfig.enableFloat16 = config.enableFloat16;
    iModelConfig.mode = TransPerformanceMode(config.mode);
    iModelConfig.priority = TransPriority(config.priority);

    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel;
    auto nnrtRet = m_iDevice->PrepareModelFromModelCache(iBuffers, iModelConfig, iPreparedModel);
    if (nnrtRet != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(nnrtRet, OH_NN_FAILED, "Prepare model from cache failed");
    }

    preparedModel = CreateSharedPtr<HDIPreparedModelV2_1>(iPreparedModel);
    if (preparedModel == nullptr) {
        LOGE("Prepare model from model cache failed, because fail to create preparedModel instance.");
        return OH_NN_MEMORY_ERROR;
    }
    return OH_NN_SUCCESS;
}

void* HDIDeviceV2_1::AllocateBuffer(size_t length)
{
    if (length == 0) {
        LOGE("The length param is invalid, length=0");
        return nullptr;
    }

    V2_1::SharedBuffer buffer;
    auto ret = m_iDevice->AllocateBuffer(length, buffer);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, nullptr, "Allocate buffer error");
    }

    auto memManager = MemoryManager::GetInstance();
    auto addr = memManager->MapMemory(buffer.fd, length);
    if (addr == nullptr) {
        LOGE("Map fd to address failed.");
    }
    return addr;
}

OH_NN_ReturnCode HDIDeviceV2_1::AllocateBuffer(size_t length, int& fd)
{
    if (length == 0) {
        LOGE("The length param is invalid, length=0");
        return OH_NN_INVALID_PARAMETER;
    }

    V2_1::SharedBuffer buffer;
    auto ret = m_iDevice->AllocateBuffer(length, buffer);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_MEMORY_ERROR, "Allocate buffer error");
    }

    fd = buffer.fd;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::ReleaseBuffer(int fd, size_t length)
{
    V2_1::SharedBuffer hdiBuffer {fd, length, 0, length};
    auto deviceResult = m_iDevice->ReleaseBuffer(hdiBuffer);
    if (deviceResult != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(deviceResult, OH_NN_FAILED, "Device release buffer error");
    }
    return OH_NN_SUCCESS;
}

void* HDIDeviceV2_1::AllocateTensorBuffer(size_t length, std::shared_ptr<TensorDesc> tensor)
{
    return AllocateBuffer(length);
}

void* HDIDeviceV2_1::AllocateTensorBuffer(size_t length, std::shared_ptr<NNTensor> tensor)
{
    return AllocateBuffer(length);
}

OH_NN_ReturnCode HDIDeviceV2_1::ReleaseBuffer(const void* buffer)
{
    if (buffer == nullptr) {
        LOGE("Buffer is nullptr, no need to release.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto memManager = MemoryManager::GetInstance();
    Memory memory;
    auto ret = memManager->GetMemory(buffer, memory);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Invalid Buffer, it is not NNRt buffer.");
        return ret;
    }

    V2_1::SharedBuffer hdiBuffer {memory.fd, memory.length, 0, memory.length};
    auto deviceResult = m_iDevice->ReleaseBuffer(hdiBuffer);
    if (deviceResult != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(deviceResult, OH_NN_FAILED, "Device release buffer error");
    }

    ret = memManager->UnMapMemory(buffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Unmap memory failed.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::ReleaseSharedBuffer(const V2_1::SharedBuffer& buffer)
{
    if (buffer.fd == INVALID_FD) {
        LOGI("No need to release. fd=%{public}d", INVALID_FD);
        return OH_NN_SUCCESS;
    }

    auto ret = m_iDevice->ReleaseBuffer(buffer);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_FAILED, "Device release buffer error");
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::GetOfflineModelFromLiteGraph(std::shared_ptr<const mindspore::lite::LiteGraph> graph,
                                                             std::vector<std::vector<uint8_t>>& offlineModels)
{
    // graph has been checked in PrepareOfflineModel, no need to check twice.
    offlineModels.clear();

    const size_t inputNum = graph->all_nodes_[0]->input_indices_.size();
    if (inputNum < OFFLINE_MODEL_MINIMUM_INPUT_SIZE) {
        LOGE("LiteGraph with offline model should have at least two input tensors, only get %zu.", inputNum);
        return OH_NN_INVALID_PARAMETER;
    }

    // The offline model is integrated into the last input tensor.
    uint32_t index = graph->all_nodes_[0]->input_indices_[inputNum - 1];
    mindspore::lite::TensorPtr pTensor = graph->all_tensors_[index];
    std::vector<uint8_t> offlineModel = mindspore::lite::MindIR_Tensor_GetData(pTensor);
    if (offlineModel.size() == (size_t) 0) {
        LOGE("Offline model has size of 0, please check the ms model.");
        return OH_NN_INVALID_PARAMETER;
    }
    offlineModels.emplace_back(std::move(offlineModel));

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::AllocateDeviceBufferForOfflineModel(
    const std::vector<std::vector<uint8_t>>& offlineModels, std::vector<Buffer>& deviceBuffers)
{
    // offlineModels is guaranteed to have at least one element in GetOfflineModelFromLiteGraph, no need to check size.
    deviceBuffers.clear();

    for (const std::vector<uint8_t>& offlineModel : offlineModels) {
        const size_t offlineModelSize = offlineModel.size();

        void* newModelBuffer = AllocateBuffer(offlineModelSize);
        if (newModelBuffer == nullptr) {
            // Release allocated model buffer if error happens.
            OH_NN_ReturnCode status {OH_NN_SUCCESS};
            for (const Buffer& deviceBuffer : deviceBuffers) {
                status = ReleaseBuffer(deviceBuffer.data);
                if (status != OH_NN_SUCCESS) {
                    LOGE("Release shared buffer of offline model failed.");
                    return status;
                }
            }

            deviceBuffers.clear();
            LOGE("Error happens when allocating shared buffer for offline model.");
            return OH_NN_MEMORY_ERROR;
        }

        Buffer modelBuffer {nullptr, 0};
        modelBuffer.data = newModelBuffer;
        modelBuffer.length = offlineModelSize;
        deviceBuffers.emplace_back(modelBuffer);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::CopyOfflineModelToDevice(const std::vector<std::vector<uint8_t>>& offlineModels,
                                                         std::vector<Buffer>& deviceBuffers)
{
    if (offlineModels.size() != deviceBuffers.size()) {
        LOGE("CopyOfflineModelToDevice failed, number of offlineModels not equal to allocated buffers.");
        return OH_NN_INVALID_PARAMETER;
    }

    const void* offlineModel {nullptr};
    size_t offlineModelSize {0};
    void* deviceBuffer {nullptr};
    size_t deviceBufferSize {0};

    size_t offlineModelsSize = offlineModels.size();
    for (size_t i = 0; i < offlineModelsSize; i++) {
        offlineModel = offlineModels[i].data();
        offlineModelSize = offlineModels[i].size();
        deviceBuffer = deviceBuffers[i].data;
        deviceBufferSize = deviceBuffers[i].length;

        // Copy offline model to shared buffer of device.
        errno_t errorCode = memcpy_s(deviceBuffer, deviceBufferSize, offlineModel, offlineModelSize);
        if (errorCode != EOK) {
            LOGE("Error happened when copy offline model to device buffer. Error code: %d.", errorCode);
            return OH_NN_MEMORY_ERROR;
        }
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::PrepareOfflineModel(std::vector<Buffer>& deviceBuffers,
                                                    const ModelConfig& config,
                                                    const std::map<std::string, std::vector<int8_t>>& extensions,
                                                    std::shared_ptr<PreparedModel>& preparedModel)
{
    V2_1::ModelConfig iModelConfig;
    iModelConfig.enableFloat16 = config.enableFloat16;
    iModelConfig.mode = TransPerformanceMode(config.mode);
    iModelConfig.priority = TransPriority(config.priority);
    iModelConfig.extensions = extensions;
    OHOS::sptr<V2_1::IPreparedModel> iPreparedModel;

    std::vector<V2_1::SharedBuffer> iBuffers;
    auto memManager = MemoryManager::GetInstance();
    Memory memory;
    OH_NN_ReturnCode ret;
    size_t numOfflineModel = deviceBuffers.size();
    for (size_t i = 0; i < numOfflineModel; i++) {
        ret = memManager->GetMemory(deviceBuffers[i].data, memory);
        if (ret != OH_NN_SUCCESS) {
            LOGE("Retrieve the memory of %zuth device buffer failed.", i);
            return ret;
        }
        iBuffers.emplace_back(V2_1::SharedBuffer {memory.fd, memory.length, 0, memory.length});
    }

    auto preparedRet = m_iDevice->PrepareOfflineModel(iBuffers, iModelConfig, iPreparedModel);

    // Release allocated model buffer after prepare model.
    OH_NN_ReturnCode status {OH_NN_SUCCESS};
    for (const Buffer& deviceBuffer : deviceBuffers) {
        status = ReleaseBuffer(deviceBuffer.data);
        if (status != OH_NN_SUCCESS) {
            LOGE("Release shared buffer of offline model failed.");
            return status;
        }
    }
    deviceBuffers.clear();

    if (preparedRet != V2_1::NNRT_ReturnCode::NNRT_SUCCESS || iPreparedModel == nullptr) {
        return CheckReturnCode_V2_1(preparedRet, OH_NN_FAILED, "Prepare offline model failed");
    }

    preparedModel = CreateSharedPtr<HDIPreparedModelV2_1>(iPreparedModel);
    if (preparedModel == nullptr) {
        LOGE("Prepare model failed, because fail to create preparedModel instance.");
        return OH_NN_MEMORY_ERROR;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV2_1::PrepareOfflineModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                                    const ModelConfig& config,
                                                    std::shared_ptr<PreparedModel>& preparedModel)
{
    if (model == nullptr) {
        LOGE("LiteGraph is empty when identifying the offline model.");
        return OH_NN_NULL_PTR;
    }

    std::vector<std::vector<uint8_t>> offlineModels;
    OH_NN_ReturnCode status = GetOfflineModelFromLiteGraph(model, offlineModels);
    if (status != OH_NN_SUCCESS) {
        LOGE("Error happens when getting offline models from lite graph.");
        return status;
    }

    std::vector<Buffer> deviceBuffers;
    status = AllocateDeviceBufferForOfflineModel(offlineModels, deviceBuffers);
    if (status != OH_NN_SUCCESS) {
        LOGE("Error happens when allocating device buffers for offline model.");
        return status;
    }

    status = CopyOfflineModelToDevice(offlineModels, deviceBuffers);
    if (status != OH_NN_SUCCESS) {
        LOGE("Error happened when copying offline models to device buffers.");

        OH_NN_ReturnCode ret {OH_NN_SUCCESS};
        // Release allocated model buffer if error happens.
        for (const Buffer& deviceBuffer : deviceBuffers) {
            ret = ReleaseBuffer(deviceBuffer.data);
            if (ret != OH_NN_SUCCESS) {
                LOGE("Releasing device buffer failed after copying offline models to device buffers failed.");
                return ret;
            }
        }

        return status;
    }

    // Retrieve offline model configs from Custom primitive and insert to extensions.
    std::string key;
    std::vector<uint8_t> valueFromCustomPrimitive;
    std::vector<int8_t> value;
    std::map<std::string, std::vector<int8_t>> extensions;
    std::vector<const mindspore::schema::Attribute*> attributes =
        mindspore::lite::MindIR_Custom_GetAttr(model->all_nodes_[0]->primitive_);
    for (const auto& attribute : attributes) {
        key = mindspore::lite::MindIR_Attribute_GetName(*attribute);
        valueFromCustomPrimitive = mindspore::lite::MindIR_Attribute_GetData(*attribute);
        value.assign(valueFromCustomPrimitive.begin(), valueFromCustomPrimitive.end());
        extensions.insert(std::pair<std::string, std::vector<int8_t>>(key, value));
    }

    status = PrepareOfflineModel(deviceBuffers, config, extensions, preparedModel);
    if (status != OH_NN_SUCCESS) {
        LOGE("PrepareOfflineModel failed.");
        return status;
    }

    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS
