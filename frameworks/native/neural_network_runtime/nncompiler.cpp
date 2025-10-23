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
#include "neural_network_runtime/neural_network_runtime.h"

#include <sys/stat.h>
#include <fstream>
#include <climits>
#include <securec.h>

#include "validation.h"
#include "nncompiled_cache.h"
#include "utils.h"
#include "nlohmann/json.hpp"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace {
const int CACHE_INPUT_TENSORDESC_OFFSET = 2;
const int CACHE_OUTPUT_TENSORDESC_OFFSET = 1;
constexpr int32_t  NUMBER_CACHE_INFO_MEMBERS = 3;
constexpr int32_t NUMBER_CACHE_INFO_EXTENSION_MEMBERS = 2;
const std::string EXTENSION_KEY_MODEL_NAME = "ModelName";
const std::string EXTENSION_KEY_FM_SHARED = "NPU_FM_SHARED";
const std::string EXTENSION_KEY_IS_EXCEED_RAMLIMIT = "isExceedRamLimit";
constexpr size_t INPUT_OUTPUT_MAX_NUM = 200;
constexpr size_t MORE_MODEL_MAX_LIMIT = 201 * 1024 * 1024; // 201MB
constexpr size_t MODEL_MAX_LIMIT = 200 * 1024 * 1024; // 200MB
constexpr size_t CHECK_SUM_ZERO = 0;
constexpr size_t CHECK_SUM_ONE = 1;
constexpr size_t CHECK_SUM_TWO = 2;
constexpr size_t EXTRACT_NODE_LAYER = 3;
constexpr int32_t MINDSPORE_CONST_NODE_TYPE = 0;

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
    m_extensionConfig = m_innerModel->GetExtensionConfig();
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
        LOGE("[NNCompiler] Build failed, both liteGraph and metaGraph are nullptr.");
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
        static_cast<OH_NN_Priority>(m_priority), m_cachePath, m_extensionConfig};
    if (m_liteGraph != nullptr) {
        ret = m_device->PrepareModel(m_liteGraph, config, m_preparedModel);
    }
    if (m_metaGraph != nullptr) {
        ret = m_device->PrepareModel(m_metaGraph, config, m_preparedModel);
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

    ret = OnlineBuild();
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] OnlineBuild failed, Failed to build model online.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::OnlineBuild()
{
    // cache存在，从cache直接复原prepareModel、input/output TensorDesc
    OH_NN_ReturnCode ret = RestoreFromCacheFile();
    if (ret == OH_NN_INVALID_FILE) {
        char path[PATH_MAX];
        if (realpath(m_cachePath.c_str(), path) == nullptr) {
            LOGE("[NNCompiler] Build failed, fail to get the real path of cacheDir.");
            return OH_NN_INVALID_PARAMETER;
        }

        std::string cachePath = path;
        std::string cacheInfo = cachePath + "/" + m_extensionConfig.modelName + "cache_info.nncache";
        if (std::filesystem::exists(cacheInfo)) {
            LOGW("[NNCompiler] cache file is failed, fail to delete cache file.");
            std::filesystem::remove_all(cacheInfo);
        }
    }

    if (ret == OH_NN_OPERATION_FORBIDDEN) {
        LOGE("[NNCompiler] Build failed, operation is forbidden.");
        return ret;
    }
    if (ret == OH_NN_SUCCESS) {
        LOGD("[NNCompiler] Build success, restore from cache file.");
        m_isBuild = true;
    }

    // cache不存在或cache restore失败，走在线构图
    if (!m_isBuild) {
        ret = NormalBuild();
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiler] Build failed, fail to build model online.");
            return ret;
        }
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

    size_t cacheNumber = caches.size();
    if (cacheNumber == 0 || cacheNumber > NN_CACHE_FILE_NUMBER_MAX) {
        LOGE("[NNCompiler] Caches size is equal 0 or greater than 100.");
        return OH_NN_FAILED;
    }

    NNCompiledCache compiledCache;
    ret = compiledCache.SetBackend(m_backendID);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SaveToCacheFile failed, fail to set backend.");
        return ret;
    }

    if ((m_inputTensorDescs.size() > INPUT_OUTPUT_MAX_NUM) || (m_outputTensorDescs.size() > INPUT_OUTPUT_MAX_NUM)) {
        LOGE("[NNCompiler] SaveToCacheFile failed, m_inputTensorDescs or m_outputTensorDescs is more than 200.");
        return OH_NN_INVALID_PARAMETER;
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

    compiledCache.SetModelName(m_extensionConfig.modelName);
    compiledCache.SetIsExceedRamLimit(m_extensionConfig.isExceedRamLimit);
    ret = compiledCache.Save(caches, m_cachePath, m_cacheVersion);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] SaveToCacheFile failed, error happened when saving model cache.");
        ReleaseBuffer(tensorBuffers);
        return ret;
    }

    ReleaseBuffer(tensorBuffers);
    ret = m_preparedModel->ReleaseBuiltModel();
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] ReleaseBuiltModel failed, error happened when release model cache.");
        return ret;
    }

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
    compiledCache.SetModelName(m_extensionConfig.modelName);
    ret = compiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, error happened when restoring model cache.");
        compiledCache.ReleaseCacheBuffer(caches);
        return ret;
    }

    size_t cacheNum = caches.size();
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> inputTensorDescs;
    ret = DeserializedTensorsFromBuffer(caches[cacheNum - CACHE_INPUT_TENSORDESC_OFFSET], inputTensorDescs);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, error happened when deserializing input tensor desc.");
        compiledCache.ReleaseCacheBuffer(caches);
        return ret;
    }

    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> outputTensorDescs;
    ret = DeserializedTensorsFromBuffer(caches[cacheNum - CACHE_OUTPUT_TENSORDESC_OFFSET], outputTensorDescs);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, error happened when deserializing output tensor desc.");
        compiledCache.ReleaseCacheBuffer(caches);
        return ret;
    }

    ModelConfig config;
    config.enableFloat16 = m_enableFp16;
    config.mode = m_performance;
    config.priority = m_priority;
    config.extensionConfig.isNpuFmShared = m_extensionConfig.isNpuFmShared;
    std::vector<Buffer> modelOnlyCaches(caches.begin(), caches.end() - CACHE_INPUT_TENSORDESC_OFFSET);
    bool isUpdatable = false;
    ret = m_device->PrepareModelFromModelCache(modelOnlyCaches, config, m_preparedModel, isUpdatable);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] RestoreFromCacheFile failed, error happened when preparing model from cache.");
        compiledCache.ReleaseCacheBuffer(caches);
        return ret;
    }

    if (isUpdatable) {
        LOGI("isUpdatable is true");

        int currentOpVersion = 0;
        ret = m_device->ReadOpVersion(currentOpVersion);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiledCache] GenerateCacheModel failed, fail to read op version.");
            return ret;
        }

        NNCompiledCacheInfo modelCacheInfo;
        std::string cacheInfoPath = m_cachePath + "/" + m_extensionConfig.modelName + "cache_info.nncache";
        ret = compiledCache.CheckCacheInfo(modelCacheInfo, cacheInfoPath);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNCompiledCache] isUpdatable is true to check cache info failed.");
            return ret;
        }

        LOGI("isUpdatable currentOpVersion is: %{public}d", currentOpVersion);
        LOGI("isUpdatable modelCacheInfo opVersion is %{public}d", static_cast<int>(modelCacheInfo.opVersion));

        if (currentOpVersion > modelCacheInfo.opVersion) {
            const size_t cacheNumber = caches.size();
            uint32_t cacheSize = NUMBER_CACHE_INFO_MEMBERS + cacheNumber + NUMBER_CACHE_INFO_EXTENSION_MEMBERS;
            uint32_t infoCharNumber = cacheSize * sizeof(int64_t);

            nlohmann::json cacheInfo;

            cacheInfo["data"]["fileNumber"] = modelCacheInfo.fileNumber;
            cacheInfo["data"]["version"] = modelCacheInfo.version - 1;
            cacheInfo["data"]["deviceId"] = modelCacheInfo.deviceId;

            for (size_t i = 0; i < modelCacheInfo.modelCheckSum.size(); ++i) {
                cacheInfo["data"]["modelCheckSum"][i] = modelCacheInfo.modelCheckSum[i];
            }

            cacheInfo["data"]["opVersion"] = currentOpVersion;
            cacheInfo["data"]["isExceedRamLimit"] = modelCacheInfo.isExceedRamLimit ? 1 : 0;

            const size_t dataLength = cacheInfo["data"].dump().length();
            char cacheInfoData[dataLength + 1];
            if (strncpy_s(cacheInfoData, dataLength+1, cacheInfo["data"].dump().c_str(), dataLength) != 0) {
                LOGE("ParseStr failed due to strncpy_s error");
                return OH_NN_INVALID_PARAMETER;
            }

            cacheInfo["CheckSum"] = static_cast<int64_t>(CacheInfoGetCrc16(cacheInfoData, dataLength));

            ret = compiledCache.WriteCacheInfo(infoCharNumber, cacheInfo, m_cachePath);
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NNCompiledCache] isUpdatable is true to write cache info failed.");
                return ret;
            }
        }
    }

    compiledCache.ReleaseCacheBuffer(caches);

    m_inputTensorDescs = inputTensorDescs;
    m_outputTensorDescs = outputTensorDescs;
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
    if (configs.find(EXTENSION_KEY_MODEL_NAME) != configs.end()) {
        std::vector<char> value = configs.at(EXTENSION_KEY_MODEL_NAME);
        if (value.empty()) {
            LOGE("[NNCompiler] SetExtensionConfig get empty model name from configs");
            return OH_NN_INVALID_PARAMETER;
        }
        m_extensionConfig.modelName.assign(value.data(), value.data() + value.size());
    }
    if (configs.find(EXTENSION_KEY_FM_SHARED) != configs.end()) {
        m_extensionConfig.isNpuFmShared = true;
        LOGI("[NNCompiler] SetExtensionConfig NpuFmShared enabled.");
    }
    if (configs.find(EXTENSION_KEY_IS_EXCEED_RAMLIMIT) != configs.end()) {
        std::vector<char> value = configs.at(EXTENSION_KEY_IS_EXCEED_RAMLIMIT);
        if (value.empty()) {
            LOGE("[NNCompiler] SetExtensionConfig get empty model name from configs");
            return OH_NN_INVALID_PARAMETER;
        }

        if (value[0] == '1') {
            m_extensionConfig.isExceedRamLimit = true;
        } else {
            m_extensionConfig.isExceedRamLimit = false;
        }
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::SetOptions(const std::vector<std::shared_ptr<void>>& options)
{
    return OH_NN_UNSUPPORTED;
}

OH_NN_ReturnCode NNCompiler::GetModelName(std::string& modelName)
{
    modelName = m_extensionConfig.modelName;
    return OH_NN_SUCCESS;
}

NNExecutor* NNCompiler::CreateExecutor()
{
    if (m_device == nullptr) {
        LOGE("[NNCompiler] CreateExecutor failed, m_device is nullptr");
        return nullptr;
    }

    if (m_preparedModel == nullptr) {
        LOGE("[NNCompiler] CreateExecutor failed, m_preparedModel is nullptr");
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
        m_backendID, m_device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs,
        m_cachePath, m_cacheVersion, m_extensionConfig, m_enableFp16, m_performance, m_priority);
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

size_t NNCompiler::DataTypeSize(mindspore::lite::DataType dataType)
{
    switch (dataType) {
        case mindspore::lite::DATA_TYPE_BOOL:
            return sizeof(bool);
        case mindspore::lite::DATA_TYPE_INT8:
            return sizeof(int8_t);
        case mindspore::lite::DATA_TYPE_UINT8:
            return sizeof(uint8_t);
        case mindspore::lite::DATA_TYPE_INT16:
            return sizeof(int16_t);
        case mindspore::lite::DATA_TYPE_UINT16:
        case mindspore::lite::DATA_TYPE_FLOAT16:
            return sizeof(uint16_t);
        case mindspore::lite::DATA_TYPE_INT32:
            return sizeof(int32_t);
        case mindspore::lite::DATA_TYPE_UINT32:
            return sizeof(uint32_t);
        case mindspore::lite::DATA_TYPE_INT64:
            return sizeof(int64_t);
        case mindspore::lite::DATA_TYPE_UINT64:
            return sizeof(uint64_t);
        case mindspore::lite::DATA_TYPE_FLOAT32:
            return sizeof(float);
        case mindspore::lite::DATA_TYPE_FLOAT64:
            return sizeof(double);
        case mindspore::lite::DATA_TYPE_UNKNOWN:
            return 0;
        default:
            LOGE("Not support the type: %{public}d", dataType);
            return 0;
    }
}

size_t NNCompiler::GetFileSize(const char* fileName)
{
    if (fileName == nullptr) {
        return 0;
    }

    // 这是一个存储文件(夹)信息的结构体，其中有文件大小和创建时间、访问时间、修改时间等
    struct stat statbuf;

    // 提供文件名字符串， 获得文件属性结构体
    stat(fileName, &statbuf);
    if (statbuf.st_size == 0) {
        return 0;
    }

    // 获取文件大小
    size_t fileSize = static_cast<size_t>(statbuf.st_size);

    return fileSize;
}

size_t NNCompiler::GetModelSizeFromModel(InnerModel* innerModel)
{
    auto liteGraph = innerModel->GetLiteGraphs();
    if (liteGraph == nullptr) {
        LOGE("GetModelSizeFromModel failed, failed to get liteGraph");
        return 0;
    }

    size_t modelSize = 0;
    std::vector<int32_t> shape;
    mindspore::lite::DataType dtype = mindspore::lite::DATA_TYPE_UNKNOWN;
    size_t num = 1;
    LOGD("GetOnlineModelSize, all_tensors_size: %{public}zu.", liteGraph->all_tensors_.size());
    for (const auto& tensor : liteGraph->all_tensors_) {
        if (tensor == nullptr) {
            LOGE("GetmodelSizeFromModel failed, failed to nullptr in model tensor");
            return 0;
        }

        // non-const node type, skip
        if (mindspore::lite::MindIR_Tensor_GetNodeType(tensor) != MINDSPORE_CONST_NODE_TYPE) {
            continue;
        }

        shape = mindspore::lite::MindIR_Tensor_GetDims(tensor);
        dtype = mindspore::lite::MindIR_Tensor_GetDataType(tensor);
        size_t tensorSize = std::accumulate(shape.begin(), shape.end(), num, std::multiplies<size_t>());
        if ((std::numeric_limits<size_t>::max() - modelSize) <= tensorSize) {
            LOGE("model size exceed max limit size, please check.");
            return 0;
        }
        modelSize +=  (tensorSize * DataTypeSize(dtype));
    }

    LOGD("GetModelSizeFromModel, modelSize: %{public}zu.", modelSize);
    return modelSize;
}

size_t NNCompiler::GetModelSizeFromFile(std::string& path)
{
    // 读取omc文件大小
    if (path.empty()) {
        LOGE("[GetModelSizeFromFile] failed, path is empty.");
        return 0;
    }

    // 获取模型文件大小
    size_t modelSize = GetFileSize(path.c_str());

    // 获取权重文件大小
    const std::string& weightPath = path;
    struct stat buffer;
    if (stat(weightPath.c_str(), &buffer) == 0) {
        modelSize += static_cast<size_t>(buffer.st_size);
    } else {
        LOGD("[GetModelSizeFromFile] weight file not exists: %{public}s.", weightPath.c_str());
    }

    LOGD("GetModelSizeFromFile, modelSize: %{public}zu.", modelSize);
    return modelSize;
}

size_t NNCompiler::GetModelSizeFromCache(std::string& path, const std::string& modelName)
{
    size_t modelSize = 0;
    if (std::filesystem::is_directory(path)) {
        if (path.empty()) {
            LOGE("GetModelSizeFromCache failed, path is nullptr");
            return 0;
        }

        std::string modelPath = path + "/" + modelName + "cache_info.nncache";
        char modelCachePath[PATH_MAX];
        if (realpath(modelPath.c_str(), modelCachePath) == nullptr) {
            LOGE("GetModelSizeFromCache failed to get the real path of cacheDir.");
            return 0;
        }

        std::string cacheInfoPath(modelCachePath);

        // cacheInfoPath is validated outside.
        std::ifstream infoCacheFile(cacheInfoPath.c_str(), std::ios::in | std::ios::binary);
        if (!infoCacheFile) {
            LOGE("[GetModelSizeFromCache] checkCacheInfo failed, error happened when opening cache info file.");
            return 0;
        }

        std::string content((std::istreambuf_iterator<char>(infoCacheFile)), std::istreambuf_iterator<char>());
        infoCacheFile.close();

        if (!nlohmann::json::accept(content)) {
            LOGE("[GetModelSizeFromCache] checkCacheInfo JSON parse error.");
            return 0;
        }

        // parse the JSON string
        nlohmann::json j = nlohmann::json::parse(content);

        int64_t isExceedRamLimit = -1;
        if (j["data"].find("isExceedRamLimit") == j["data"].end()) {
            LOGW("[GetModelSizeFromCache] checkCacheInfo read cache isExceedRamLimit failed.");
        }

        isExceedRamLimit = j["data"]["isExceedRamLimit"].get<int64_t>();
        modelSize = isExceedRamLimit == 1 ? MORE_MODEL_MAX_LIMIT : MODEL_MAX_LIMIT;
    } else {
        modelSize = GetModelSizeFromFile(path);
    }
    return modelSize;
}

size_t NNCompiler::GetModelSize()
{
    size_t modelSize = 0;
    if (m_innerModel != nullptr) {
        modelSize = GetModelSizeFromModel(m_innerModel);
    } else {
        modelSize = GetModelSizeFromCache(m_cachePath, m_extensionConfig.modelName);
    }

    return modelSize;
}

std::vector<mindspore::lite::LiteGraph::Node*> NNCompiler::GetNodeIndices(
    const std::shared_ptr<mindspore::lite::LiteGraph>& liteGraph, size_t layer)
{
    std::vector<mindspore::lite::LiteGraph::Node*> nodes;
    size_t inputTensorSize = liteGraph->input_indices_.size();
    size_t outputTensorSize = liteGraph->output_indices_.size();

    size_t allnodeSize = liteGraph->all_nodes_.size();
    if (((inputTensorSize + outputTensorSize) * layer) >= allnodeSize) {
        LOGI("The all node size in model is too small, return all nodes.");
        return liteGraph->all_nodes_;
    }

    for (size_t i = 0; i < inputTensorSize * layer; ++i) {
        nodes.emplace_back(liteGraph->all_nodes_[i]);
    }

    for (size_t j = allnodeSize - 1; j >= (allnodeSize - (outputTensorSize * layer)); --j) {
        nodes.emplace_back(liteGraph->all_nodes_[j]);
    }

    LOGD("nodes size : %{public}zu.", nodes.size());
    return nodes;
}

size_t NNCompiler::GetOnlineModelID(const std::shared_ptr<mindspore::lite::LiteGraph>& liteGraph)
{
    size_t inputSize = liteGraph->input_indices_.size();
    size_t outputSize = liteGraph->output_indices_.size();
    size_t allTensorSize = liteGraph->all_tensors_.size();
    size_t allNodesSize = liteGraph->all_nodes_.size();

    std::string onlineModelId = "";
    onlineModelId.append(std::to_string(inputSize));
    onlineModelId.append(std::to_string(outputSize));
    onlineModelId.append(std::to_string(allTensorSize));
    onlineModelId.append(std::to_string(allNodesSize));

    std::vector<mindspore::lite::LiteGraph::Node*> nodes = GetNodeIndices(liteGraph, EXTRACT_NODE_LAYER);

    for (auto node : nodes) {
        onlineModelId.append(node->name_);
    }

    return std::hash<std::string>{}(onlineModelId);
}

OH_NN_ReturnCode NNCompiler::GetNNRtModelIDFromModel(InnerModel* innerModel, size_t& nnrtModelID)
{
    if (innerModel == nullptr) {
        LOGE("GetNNRtModelIDFromModel failed, model is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto liteGraph = innerModel->GetLiteGraphs();
    if (liteGraph == nullptr) {
        LOGE("GetNNRtModelIDFromModel failed, failed to get liteGraph.");
        return OH_NN_INVALID_PARAMETER;
    }

    nnrtModelID = GetOnlineModelID(liteGraph);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNCompiler::GetNNRtModelIDFromCache(const std::string& path, const std::string& modelName,
    size_t& nnrtModelID)
{
    if (path.empty()) {
        LOGE("GetNNRtModelIDFromCache failed, path is empty");
        return OH_NN_INVALID_PARAMETER;
    }

    if (modelName.empty()) {
        LOGE("GetNNRtModelIDFromCache failed, modelName is empty");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!std::filesystem::is_directory(path)) {
        LOGW("GetNNRtModelIDFromCache cvache path is not directory.");
        nnrtModelID = std::hash<std::string>{}(path);
        return OH_NN_SUCCESS;
    }

    std::string modelPath = path + "/" + modelName + "cache_info.nncache";
    char modelCachePath[PATH_MAX];
    if (realpath(modelPath.c_str(), modelCachePath) == nullptr) {
        LOGE("GetNNRtModelIDFromCache fail to get real path of cacheDir.");
        return OH_NN_INVALID_PARAMETER;
    }

    NNCompiledCache compiledCache;
    NNCompiledCacheInfo cacheInfo;
    OH_NN_ReturnCode retCode = compiledCache.SetBackend(m_backendID);
    if (retCode != OH_NN_SUCCESS) {
        LOGE("GetNNRtmodelIDFromCache failed, fail to set backend.");
        return retCode;
    }

    retCode = compiledCache.CheckCacheInfo(cacheInfo, modelCachePath);
    if (retCode != OH_NN_SUCCESS) {
        LOGE("GetNNRtmodelIDFromCache failed, fail to CheckCacheInfo.");
        return retCode;
    }

    if (cacheInfo.modelCheckSum.size() != NUMBER_CACHE_INFO_MEMBERS) {
        LOGE("GetNNRtmodelIDFromCache failed, fail to modelCheckSum.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::string cacheStr = std::to_string(cacheInfo.modelCheckSum[CHECK_SUM_ZERO]) +
        std::to_string(cacheInfo.modelCheckSum[CHECK_SUM_ONE]) +
        std::to_string(cacheInfo.modelCheckSum[CHECK_SUM_TWO]);
    nnrtModelID = std::hash<std::string>{}(cacheStr);

    return OH_NN_SUCCESS;
}

size_t NNCompiler::GetOnlineModelID()
{
    size_t nnrtModeId = 0;
    OH_NN_ReturnCode ret = GetNNRtModelIDFromCache(m_cachePath, m_extensionConfig.modelName, nnrtModeId);
    if (ret != OH_NN_SUCCESS && m_innerModel != nullptr) {
        ret = GetNNRtModelIDFromModel(m_innerModel, nnrtModeId);
    }

    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNCompiler] GetOnlineModelID failed.");
        return 0;
    }

    return nnrtModeId;
}
} // NeuralNetworkRuntime
} // OHOS
