/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#include "nncompiler.h"

#include <sys/stat.h>
#include <fstream>
#include <climits>
#include <securec.h>

#include "validation.h"
#include "nncompiled_cache.h"
#include "common/utils.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace {
const int CACHE_INPUT_TENSORDESC_OFFSET = 2;
const int CACHE_OUTPUT_TENSORDESC_OFFSET = 1;

struct SerializedTensorDesc {
public:
    SerializedTensorDesc() = default;
    ~SerializedTensorDesc() = default;

    OH_NN_ReturnCode CopyFromTensorDesc(const std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>& tensorDesc)
    {
        if (tensorDesc.first == nullptr) {
            LOGE("CopyFromTensorDesc failed, tensor desc is nullptr.");
            return OH_NN_NULL_PTR;
        }
        OH_NN_ReturnCode ret = tensorDesc.first->GetDataType(&m_dataType);
        if (ret != OH_NN_SUCCESS) {
            LOGE("CopyFromTensorDesc failed, error happened when getting data type from tensor desc.");
            return ret;
        }

        ret = tensorDesc.first->GetFormat(&m_format);
        if (ret != OH_NN_SUCCESS) {
            LOGE("CopyFromTensorDesc failed, error happened when getting format from tensor desc.");
            return ret;
        }

        ret = tensorDesc.first->GetShape(&m_shape, &m_shapeNum);
        if (ret != OH_NN_SUCCESS) {
            LOGE("CopyFromTensorDesc failed, error happened when getting shape from tensor desc.");
            return ret;
        }

        ret = tensorDesc.first->GetName(&m_name);
        if (ret != OH_NN_SUCCESS) {
            LOGE("CopyFromTensorDesc failed, error happened when getting name from tensor desc.");
            return ret;
        }

        m_tensorType = tensorDesc.second;

        return ret;
    }

    OH_NN_ReturnCode CopyToTensorDesc(TensorDesc& tensorDesc) const
    {
        OH_NN_ReturnCode ret = tensorDesc.SetDataType(m_dataType);
        if (ret != OH_NN_SUCCESS) {
            LOGE("CopyToTensorDesc failed, error happened when setting data type to tensor desc.");
            return ret;
        }

        ret = tensorDesc.SetFormat(m_format);
        if (ret != OH_NN_SUCCESS) {
            LOGE("CopyToTensorDesc failed, error happened when setting format to tensor desc.");
            return ret;
        }

        ret = tensorDesc.SetShape(m_shape, m_shapeNum);
        if (ret != OH_NN_SUCCESS) {
            LOGE("CopyToTensorDesc failed, error happened when setting shape to tensor desc.");
            return ret;
        }

        ret = tensorDesc.SetName(m_name);
        if (ret != OH_NN_SUCCESS) {
            LOGE("CopyToTensorDesc failed, error happened when setting name to tensor desc.");
        }

        return ret;
    }

public:
    OH_NN_DataType m_dataType{OH_NN_UNKNOWN};
    OH_NN_Format m_format{OH_NN_FORMAT_NONE};
    OH_NN_TensorType m_tensorType{OH_NN_TENSOR};
    size_t m_shapeNum{0};
    int32_t* m_shape{nullptr};
    const char* m_name{nullptr}; // null-terminated
};

const size_t SIZE_OF_DATATYPE = sizeof(SerializedTensorDesc::m_dataType);
const size_t SIZE_OF_FORMAT = sizeof(SerializedTensorDesc::m_format);
const size_t SIZE_OF_TENSOR_TYPE = sizeof(SerializedTensorDesc::m_tensorType);
const size_t SIZE_OF_SHAPE_NUM = sizeof(SerializedTensorDesc::m_shapeNum);
} // namespace

NNCompiler::NNCompiler(std::shared_ptr<Device> device, size_t backendID)
    : m_device(device),
    m_backendID(backendID) {}

NNCompiler::NNCompiler(const void* model, std::shared_ptr<Device> device, size_t backendID)
    : m_device(device),
    m_backendID(backendID)
{
    m_innerModel = const_cast<InnerModel*>(reinterpret_cast<const InnerModel*>(model));
    m_liteGraph = m_innerModel->GetLiteGraphs();
    m_inputTensorDescs = m_innerModel->GetInputTensorDescs();
    m_outputTensorDescs = m_innerModel->GetOutputTensorDescs();
    m_metaGraph = m_innerModel->GetMetaGraph();
    m_quantBuffer = m_innerModel->GetQuantBuffer();
    m_modelName = m_innerModel->GetModelName();
    m_isProfiling = m_innerModel->GetProfiling();
    m_opLayouts = m_innerModel->GetOpLayouts();
    m_tuningStrategy = m_innerModel->GetTuningStrategy();
}

NNCompiler::~NNCompiler()
{
    if (m_preparedModel != nullptr) {
        m_preparedModel.reset();
    }
    m_inputTensorDescs.clear();
    m_outputTensorDescs.clear();
}

size_t NNCompiler::GetBackendID() const
{
    return m_backendID;
}

OH_NN_ReturnCode NNCompiler::SetCacheDir(const std::string& cacheModelPath, uint32_t version)
{
    if (m_device == nullptr) {
        LOGE("[NNCompiler] SetCacheDir failed, m_device is nullptr");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    bool isSupportedCache {false};
    OH_NN_ReturnCode ret = m_device->IsModelCacheSupported(isSupportedCache);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SetCacheDir failed, fail to call device.");
        return ret;
    }

    if (!isSupportedCache && !cacheModelPath.empty()) {
        LOGE("[NNCompiler] SetCacheDir failed, this device is not support cache setting.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    m_cachePath = cacheModelPath;
    m_cacheVersion = version;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::SetPerformance(OH_NN_PerformanceMode performance)
{
    if (m_device == nullptr) {
        LOGE("[NNCompiler] SetPerformance failed, m_device is nullptr");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    bool isSupportedPerformance {false};
    OH_NN_ReturnCode ret = m_device->IsPerformanceModeSupported(isSupportedPerformance);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SetPerformance failed, fail to call device.");
        return OH_NN_FAILED;
    }

    if (!isSupportedPerformance && (performance != OH_NN_PERFORMANCE_NONE)) {
        LOGE("[NNCompiler] SetPerformance failed, this device is not support performance setting.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!Validation::ValidatePerformanceMode(performance)) {
        LOGE("[NNCompiler] SetPerformance failed, performance=%{public}d is invalid", performance);
        return OH_NN_INVALID_PARAMETER;
    }

    m_performance = performance;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::SetPriority(OH_NN_Priority priority)
{
    if (m_device == nullptr) {
        LOGE("[NNCompiler] SetPriority failed, m_device is nullptr");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    bool isSupportedPriority {false};
    OH_NN_ReturnCode ret = m_device->IsPrioritySupported(isSupportedPriority);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SetPriority failed, fail to call device.");
        return ret;
    }

    if (!isSupportedPriority && (priority != OH_NN_PRIORITY_NONE)) {
        LOGE("[NNCompiler] SetPriority failed, this device is not support priority setting.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (!Validation::ValidatePriority(priority)) {
        LOGE("[NNCompiler] SetPriority failed, priority=%{public}d is invalid.", priority);
        return OH_NN_INVALID_PARAMETER;
    }

    m_priority = priority;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::SetEnableFp16(bool isFp16)
{
    if (m_device == nullptr) {
        LOGE("[NNCompiler] SetEnableFp16 failed, m_device is nullptr");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    bool isSupportedFp16 {false};
    OH_NN_ReturnCode ret = m_device->IsFloat16PrecisionSupported(isSupportedFp16);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SetEnableFp16 failed, fail to call device.");
        return ret;
    }

    if (!isSupportedFp16 && isFp16) {
        LOGE("[NNCompiler] SetEnableFp16 failed, this device is not support float16 precision setting.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    m_enableFp16 = isFp16;
    return OH_NN_SUCCESS;
}

bool NNCompiler::IsBuild() const
{
    return m_isBuild;
}

OH_NN_ReturnCode NNCompiler::IsSupportedModel(const std::shared_ptr<mindspore::lite::LiteGraph>& liteGraph,
                                              bool& isSupportedModel) const
{
    std::vector<bool> supportedList;
    OH_NN_ReturnCode ret = m_device->GetSupportedOperation(liteGraph, supportedList);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] Build failed, error happened when getting supported operation.");
        return ret;
    }

    bool isNotSupport = std::any_of(supportedList.begin(), supportedList.end(), [](bool isSupport) {
        return !isSupport;
    });
    if (isNotSupport) {
        LOGE("[NNCompiler] Build failed, current device not support the model, device id: %{public}zu.", m_backendID);
        isSupportedModel = false;
        return OH_NN_FAILED;
    }

    isSupportedModel = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::CheckModelParameter() const
{
    // If m_innerModel is not passed, the compiler must be construct from cache, jump check m_innerModel.
    if (m_innerModel == nullptr) {
        LOGW("[NNCompiler] Restoring from cache not need to check model.");
        return OH_NN_SUCCESS;
    }

    // m_innerModel is not constructed completely.
    if ((m_liteGraph == nullptr) && (m_metaGraph == nullptr)) {
        LOGE("[NNCompiler] LiteGraph and metaGraph are empty, m_innerModel is not constructed completely.");
        return OH_NN_INVALID_PARAMETER;
    }

    if ((m_liteGraph != nullptr) && (m_metaGraph != nullptr)) {
        LOGE("[NNCompiler] Both LiteGraph and metaGraph are not empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::IsOfflineModel(bool& isOfflineModel) const
{
    // If m_innerModel is not passed, the compiler must be construct from cache, jump check m_innerModel.
    if (m_innerModel == nullptr) {
        LOGW("[NNCompiler] Restoring from cache not need to judge offline model.");
        return OH_NN_SUCCESS;
    }

    isOfflineModel = false; // Initialize the returned value
    if (m_metaGraph != nullptr) {
        isOfflineModel = false;
        return OH_NN_SUCCESS;
    }

    if (m_liteGraph->all_nodes_.size() == 0) {
        LOGE("[NNCompiler] Find empty node in the model.");
        return OH_NN_INVALID_PARAMETER;
    }

    // If the model consists of more than 1 node, it will not be considered as offline model.
    if (m_liteGraph->all_nodes_.size() > 1) {
        isOfflineModel = false;
        return OH_NN_SUCCESS;
    }

    const mindspore::lite::LiteGraph::Node* pNode = m_liteGraph->all_nodes_[0];
    if (pNode == nullptr) {
        LOGE("[NNCompiler] Find invalid node in the model.");
        return OH_NN_NULL_PTR;
    }

    const mindspore::lite::NodeType& nodeType = mindspore::lite::MindIR_Primitive_GetType(pNode->primitive_);
    if (nodeType == mindspore::lite::NodeType::NODE_TYPE_CUSTOM) {
        isOfflineModel = true;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::BuildOfflineModel()
{
    ModelConfig config {m_enableFp16, m_performance, m_priority};
    OH_NN_ReturnCode ret = m_device->PrepareOfflineModel(m_liteGraph, config, m_preparedModel);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] Preparing model failed when building from offline model.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::NormalBuild()
{
    if ((m_liteGraph == nullptr) && (m_metaGraph == nullptr)) {
        LOGW("[NNCompiler] Build failed, both liteGraph and metaGraph are nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if ((m_liteGraph != nullptr) && (m_metaGraph != nullptr)) {
        LOGE("[NNCompiler] Build failed, neither liteGraph nor metaGraph are nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    // 判断是否支持模型
    bool isSupportedModel = true;
    OH_NN_ReturnCode ret = IsSupportedModel(m_liteGraph, isSupportedModel);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] Build failed, error happened when judge if support the model.");
        return ret;
    } else if (!isSupportedModel) {
        LOGE("[NNCompiler] Build failed, current device not support the model.");
        return OH_NN_FAILED;
    }

    ModelConfig config {m_enableFp16, static_cast<OH_NN_PerformanceMode>(m_performance),
        static_cast<OH_NN_Priority>(m_priority), m_isProfiling, m_cachePath, m_opLayouts, m_tuningStrategy};
    if (m_liteGraph != nullptr) {
        ret = m_device->PrepareModel(m_liteGraph, m_quantBuffer, config, m_preparedModel);
    }
    if (m_metaGraph != nullptr) {
        ret = m_device->PrepareModel(m_metaGraph, m_quantBuffer, config, m_preparedModel);
    }
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] Build failed, fail to prepare model when normally building.");
        return ret;
    }
    m_isBuild = true;

    // 保存cache
    if (!m_cachePath.empty()) {
        ret = SaveToCacheFile();
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiler] Build success, but fail to save cache to file.");
            return ret;
        }
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::Build()
{
    if (m_isBuild) {
        LOGE("[NNCompiler] Build failed, cannot build again.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (m_device == nullptr) {
        LOGE("[NNCompiler] Build failed, the m_device is nullptr.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    OH_NN_ReturnCode ret = CheckModelParameter();
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] CheckModelParameter failed, some error happened when checking model parameter.");
        return ret;
    }

    // Prepare from offline model.
    bool isOfflineModel {false};
    ret = IsOfflineModel(isOfflineModel);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] Build failed, fail to identify the offline model.");
        return ret;
    }

    if (isOfflineModel) {
        ret = BuildOfflineModel();
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiler] Build failed, Failed to build offline model.");
            return ret;
        }

        m_isBuild = true;
        return OH_NN_SUCCESS;
    }

    // cache存在，从cache直接复原prepareModel、input/output TensorDesc
    ret = RestoreFromCacheFile();
    if (ret == OH_NN_OPERATION_FORBIDDEN) {
        LOGE("[NNCompiler] Build failed, operation is forbidden.");
        return ret;
    }
    if (ret == OH_NN_SUCCESS) {
        LOGI("[NNCompiler] Build success, restore from cache file.");
        m_isBuild = true;
        return OH_NN_SUCCESS;
    }

    // cache不存在或cache restore失败，走在线构图
    ret = NormalBuild();
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] Build failed, fail to build model online.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

void NNCompiler::ReleaseBuffer(std::vector<Buffer>& buffers) const
{
    for (size_t i = 0; i < buffers.size(); ++i) {
        // release tensor buffer which is allocated by new method.
        delete[] reinterpret_cast<char*>(buffers[i].data);
    }
    buffers.clear();
}

void NNCompiler::ReleaseBufferByDevice(std::vector<Buffer>& buffers) const
{
    for (size_t i = 0; i < buffers.size(); ++i) {
        // release cache buffer which is allocated by idevice.
        m_device->ReleaseBuffer(buffers[i].data);
    }
    buffers.clear();
}

OH_NN_ReturnCode NNCompiler::SaveToCacheFile() const
{
    if (m_cachePath.empty()) {
        LOGE("[NNCompiler] SaveToCacheFile failed, m_cachePath is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_cacheVersion == INVALID_CAHCE_VERSION) {
        LOGE("[NNCompiler] SaveToCacheFile failed, cache version is invalid. Please set a valid cache version.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_preparedModel == nullptr) {
        LOGE("[NNCompiler] SaveToCacheFile failed, m_preparedModel is nullptr. Please construct prepareModel first.");
        return OH_NN_FAILED;
    }

    std::vector<Buffer> caches;
    std::vector<Buffer> tensorBuffers;
    OH_NN_ReturnCode ret = m_preparedModel->ExportModelCache(caches);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SaveToCacheFile failed, error happened when exporting model cache.");
        return ret;
    }

    NNCompiledCache compiledCache;
    ret = compiledCache.SetBackend(m_backendID);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SaveToCacheFile failed, fail to set backend.");
        return ret;
    }

    Buffer inputTensorDescBuffer;
    ret = SerializeTensorsToBuffer(m_inputTensorDescs, inputTensorDescBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SaveToCacheFile failed, error happened when serializing input tensor desc.");
        return ret;
    }
    caches.emplace_back(inputTensorDescBuffer);
    tensorBuffers.emplace_back(inputTensorDescBuffer);

    Buffer outputTensorDescBuffer;
    ret = SerializeTensorsToBuffer(m_outputTensorDescs, outputTensorDescBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SaveToCacheFile failed, error happened when serializing output tensor desc.");
        ReleaseBuffer(tensorBuffers);
        return ret;
    }
    caches.emplace_back(outputTensorDescBuffer);
    tensorBuffers.emplace_back(outputTensorDescBuffer);

    compiledCache.SetModelName(m_modelName);
    ret = compiledCache.Save(caches, m_cachePath, m_cacheVersion);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SaveToCacheFile failed, error happened when saving model cache.");
        ReleaseBuffer(tensorBuffers);
        return ret;
    }

    ReleaseBuffer(tensorBuffers);
    LOGI("[NNCompiler] Export model cache successfully.");
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::RestoreFromCacheFile()
{
    if (m_cachePath.empty()) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, path is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_cacheVersion == INVALID_CAHCE_VERSION) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, cache version is invalid. Please set a valid cache version.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_preparedModel != nullptr) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, m_preparedModel is not nullptr.");
        return OH_NN_FAILED;
    }

    NNCompiledCache compiledCache;
    OH_NN_ReturnCode ret = compiledCache.SetBackend(m_backendID);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, fail to set backend.");
        return ret;
    }

    std::vector<Buffer> caches;
    compiledCache.SetModelName(m_modelName);
    ret = compiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, error happened when restoring model cache.");
        ReleaseBufferByDevice(caches);
        return ret;
    }

    size_t cacheNum = caches.size();
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> inputTensorDescs;
    ret = DeserializedTensorsFromBuffer(caches[cacheNum - CACHE_INPUT_TENSORDESC_OFFSET], inputTensorDescs);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, error happened when deserializing input tensor desc.");
        ReleaseBufferByDevice(caches);
        return ret;
    }

    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> outputTensorDescs;
    ret = DeserializedTensorsFromBuffer(caches[cacheNum - CACHE_OUTPUT_TENSORDESC_OFFSET], outputTensorDescs);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, error happened when deserializing output tensor desc.");
        ReleaseBufferByDevice(caches);
        return ret;
    }

    ModelConfig config;
    config.enableFloat16 = m_enableFp16;
    config.mode = m_performance;
    config.priority = m_priority;
    std::vector<Buffer> modelOnlyCaches(caches.begin(), caches.end() - CACHE_INPUT_TENSORDESC_OFFSET);
    ret = m_device->PrepareModelFromModelCache(modelOnlyCaches, config, m_preparedModel);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, error happened when preparing model from cache.");
        ReleaseBufferByDevice(caches);
        return ret;
    }
    ReleaseBufferByDevice(caches);

    m_inputTensorDescs = inputTensorDescs;
    m_outputTensorDescs = outputTensorDescs;
    LOGI("[NNCompiler] Restore model cache successfully.");
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::SaveToCacheBuffer(const void* buffer, size_t length, size_t* modelSize) const
{
    LOGE("[NNCompiler] SaveToCacheBuffer is not supported currently.");
    return OH_NN_UNSUPPORTED;
}

OH_NN_ReturnCode NNCompiler::RestoreFromCacheBuffer(const void* buffer, size_t length)
{
    LOGE("[NNCompiler] RestoreFromCacheBuffer is not supported currently.");
    return OH_NN_UNSUPPORTED;
}

OH_NN_ReturnCode NNCompiler::SetExtensionConfig(const std::unordered_map<std::string, std::vector<char>>& configs)
{
    LOGI("[NNCompiler] SetExtensionConfig successfully.");
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::SetOptions(const std::vector<std::shared_ptr<void>>& options)
{
    LOGE("[NNCompiler] SetOptions is not supported for NN compiler currently.");
    return OH_NN_UNSUPPORTED;
}

NNExecutor* NNCompiler::CreateExecutor()
{
    if (m_device == nullptr) {
        LOGE("[NNCompiler] CreateExecutor failed, m_device is nullptr");
        return nullptr;
    }

    if (m_preparedModel == nullptr) {
        LOGE("[NNCompiler] CreateExecutor failed, m_device is nullptr");
        return nullptr;
    }

    if (m_inputTensorDescs.empty()) {
        LOGE("[NNCompiler] CreateExecutor failed, m_inputTensorDescs is empty");
        return nullptr;
    }

    if (m_outputTensorDescs.empty()) {
        LOGE("[NNCompiler] CreateExecutor failed, m_outputTensorDescs is empty");
        return nullptr;
    }

    NNExecutor* nnExecutor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);
    if (nnExecutor == nullptr) {
        LOGE("[NNCompiler] CreateExecutor failed, error happend when allocating NN Executor.");
        return nullptr;
    }

    return nnExecutor;
}

OH_NN_ReturnCode NNCompiler::SerializeTensorsToBuffer(
    const std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& tensorDescs, Buffer& buffer) const
{
    std::vector<SerializedTensorDesc> immediateTensorDescs;
    OH_NN_ReturnCode ret = OH_NN_SUCCESS;
    for (const auto& tensorDesc : tensorDescs) {
        SerializedTensorDesc immediateTensorDesc;
        ret = immediateTensorDesc.CopyFromTensorDesc(tensorDesc);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiler] SerializeInputsToBuffer failed, error happened when copying tensorDesc to "
                 "SerializedTensorDesc.");
            immediateTensorDescs.clear();
            return ret;
        }
        immediateTensorDescs.emplace_back(immediateTensorDesc);
    }

    size_t totalSize = 0;
    for (const auto& tensorDesc : immediateTensorDescs) {
        totalSize += SIZE_OF_DATATYPE;
        totalSize += SIZE_OF_FORMAT;
        totalSize += SIZE_OF_TENSOR_TYPE;
        totalSize += SIZE_OF_SHAPE_NUM;
        totalSize += tensorDesc.m_shapeNum * sizeof(int32_t);
        totalSize += strlen(tensorDesc.m_name) + 1;
    }

    // Allocate memory for the serialized data
    char* serializedData = new (std::nothrow) char[totalSize];
    if (serializedData == nullptr) {
        LOGE("[NNCompiler] SerializeInputsToBuffer failed, failed to create serialized data.");
        return OH_NN_NULL_PTR;
    }
    char* currentPos = serializedData;

    // Serialize each tensor description
    for (const auto& tensorDesc : immediateTensorDescs) {
        auto memRet = memcpy_s(currentPos, SIZE_OF_DATATYPE, &tensorDesc.m_dataType, SIZE_OF_DATATYPE);
        if (memRet != EOK) {
            LOGE("[NNCompiler] SerializeInputsToBuffer failed, failed to memcpy_s data type.");
            delete[] serializedData;
            return OH_NN_MEMORY_ERROR;
        }
        currentPos += SIZE_OF_DATATYPE;

        memRet = memcpy_s(currentPos, SIZE_OF_FORMAT, &tensorDesc.m_format, SIZE_OF_FORMAT);
        if (memRet != EOK) {
            LOGE("[NNCompiler] SerializeInputsToBuffer failed, failed to memcpy_s format.");
            delete[] serializedData;
            return OH_NN_MEMORY_ERROR;
        }
        currentPos += SIZE_OF_FORMAT;

        memRet = memcpy_s(currentPos, SIZE_OF_TENSOR_TYPE, &tensorDesc.m_tensorType, SIZE_OF_TENSOR_TYPE);
        if (memRet != EOK) {
            LOGE("[NNCompiler] SerializeInputsToBuffer failed, failed to memcpy_s tensor type.");
            delete[] serializedData;
            return OH_NN_MEMORY_ERROR;
        }
        currentPos += SIZE_OF_TENSOR_TYPE;

        memRet = memcpy_s(currentPos, SIZE_OF_SHAPE_NUM, &tensorDesc.m_shapeNum, SIZE_OF_SHAPE_NUM);
        if (memRet != EOK) {
            LOGE("[NNCompiler] SerializeInputsToBuffer failed, failed to memcpy_s shape num.");
            delete[] serializedData;
            return OH_NN_MEMORY_ERROR;
        }
        currentPos += SIZE_OF_SHAPE_NUM;

        size_t sizeOfShape = tensorDesc.m_shapeNum * sizeof(int32_t);
        memRet = memcpy_s(currentPos, sizeOfShape, tensorDesc.m_shape, sizeOfShape);
        if (memRet != EOK) {
            LOGE("[NNCompiler] SerializeInputsToBuffer failed, failed to memcpy_s shape.");
            delete[] serializedData;
            return OH_NN_MEMORY_ERROR;
        }
        currentPos += sizeOfShape;

        memRet = strcpy_s(currentPos, strlen(tensorDesc.m_name) + 1, tensorDesc.m_name);
        if (memRet != EOK) {
            LOGE("[NNCompiler] SerializeInputsToBuffer failed, failed to memcpy_s name.");
            delete[] serializedData;
            return OH_NN_MEMORY_ERROR;
        }
        currentPos += strlen(tensorDesc.m_name) + 1;
    }

    buffer.data = serializedData;
    buffer.length = totalSize;

    return OH_NN_SUCCESS;
}

void ReleaseDescShape(std::vector<SerializedTensorDesc>& immediateTensorDescs)
{
    for (auto desc : immediateTensorDescs) {
        delete[] desc.m_shape;
    }
    immediateTensorDescs.clear();
}

OH_NN_ReturnCode NNCompiler::DeserializedTensorsFromBuffer(
    const Buffer& buffer, std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& tensorDescs)
{
    std::vector<SerializedTensorDesc> immediateTensorDescs;
    const char* ptr = static_cast<const char*>(buffer.data);
    const char* end = ptr + buffer.length;
    while (ptr < end) {
        SerializedTensorDesc desc;

        auto memRet = memcpy_s(&desc.m_dataType, SIZE_OF_DATATYPE, ptr, sizeof(desc.m_dataType));
        if (memRet != EOK) {
            LOGE("[NNCompiler] DeserializedTensorsFromBuffer failed, failed to memcpy_s data type.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += sizeof(desc.m_dataType);

        memRet = memcpy_s(&desc.m_format, SIZE_OF_FORMAT, ptr, sizeof(desc.m_format));
        if (memRet != EOK) {
            LOGE("[NNCompiler] DeserializedTensorsFromBuffer failed, failed to memcpy_s format.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += sizeof(desc.m_format);

        memRet = memcpy_s(&desc.m_tensorType, SIZE_OF_TENSOR_TYPE, ptr, sizeof(desc.m_tensorType));
        if (memRet != EOK) {
            LOGE("[NNCompiler] DeserializedTensorsFromBuffer failed, failed to memcpy_s tensor type.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += sizeof(desc.m_tensorType);

        memRet = memcpy_s(&desc.m_shapeNum, SIZE_OF_SHAPE_NUM, ptr, sizeof(desc.m_shapeNum));
        if (memRet != EOK) {
            LOGE("[NNCompiler] DeserializedTensorsFromBuffer failed, failed to memcpy_s shape num.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += sizeof(desc.m_shapeNum);

        desc.m_shape = new (std::nothrow) int32_t[desc.m_shapeNum];
        if (desc.m_shape == nullptr) {
            LOGE("[NNCompiler] DeserializedTensorsFromBuffer failed, failed to create shape buffer.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_NULL_PTR;
        }
        memRet = memcpy_s(desc.m_shape, desc.m_shapeNum * sizeof(int32_t), ptr, desc.m_shapeNum * sizeof(int32_t));
        if (memRet != EOK) {
            LOGE("[NNCompiler] DeserializedTensorsFromBuffer failed, failed to memcpy_s shape.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += desc.m_shapeNum * sizeof(int32_t);

        desc.m_name = ptr;
        ptr += std::strlen(desc.m_name) + 1; // +1 for null terminator

        immediateTensorDescs.push_back(desc);
    }

    OH_NN_ReturnCode ret {OH_NN_SUCCESS};
    for (const auto& immediateTensorDesc : immediateTensorDescs) {
        std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> tensorDescPair;
        tensorDescPair.first = CreateSharedPtr<TensorDesc>();
        if (tensorDescPair.first == nullptr) {
            LOGE("[NNCompiler] DeserializedTensorsFromBuffer failed, failed to create tensor desc.");
            tensorDescs.clear();
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_NULL_PTR;
        }
        ret = immediateTensorDesc.CopyToTensorDesc(*(tensorDescPair.first.get()));
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiler] DeserializedTensorsFromBuffer failed, error happened when copying "
                 "SerializedTensorDesc to TensorDesc.");
            tensorDescs.clear();
            ReleaseDescShape(immediateTensorDescs);
            return ret;
        }
        tensorDescPair.second = immediateTensorDesc.m_tensorType;

        tensorDescs.emplace_back(tensorDescPair);
    }

    ReleaseDescShape(immediateTensorDescs);
    return ret;
}

} // NeuralNetworkRuntime
} // OHOS
