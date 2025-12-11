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


#include "nnexecutor.h"
#include "nntensor.h"
#include "nncompiled_cache.h"
#include "cpp_type.h"
#include "neural_network_runtime_inner.h"
#include "nnrt_client.h"
#include "log.h"

#include "securec.h"
#include "utils.h"
#include "scoped_trace.h"
#include "transform.h"

namespace OHOS {
constexpr size_t EXTENSION_MAX_SIZE = 200;
constexpr int AUTOUNLOAD_TIME = 10 * 60 * 1000;

namespace NeuralNetworkRuntime {
constexpr int CACHE_INPUT_TENSORDESC_OFFSET = 2;
constexpr int CACHE_OUTPUT_TENSORDESC_OFFSET = 1;
constexpr size_t CHECK_SUM_ZERO = 0;
constexpr size_t CHECK_SUM_ONE = 1;
constexpr size_t CHECK_SUM_TWO = 2;
constexpr int32_t  NUMBER_CACHE_INFO_MEMBERS = 3;

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

uint64_t GenRandom(void)
{
    uint64_t random = 0;
    int fd = open("/dev/random", O_RDONLY);
    if (fd >= 0) {
        read(fd, &random, sizeof(random));
        close(fd);
    }
    return random;
}

NNExecutor::NNExecutor(size_t backendID, std::shared_ptr<Device> device, std::shared_ptr<PreparedModel> preparedModel,
    const std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& inputTensorDescs,
    const std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& outputTensorDescs,
    std::string cachePath, uint32_t cacheVersion, ExtensionConfig extensionConfig, bool enableFp16,
    OH_NN_PerformanceMode performance, OH_NN_Priority priority)
    : m_backendID(backendID),
    m_device(device),
    m_preparedModel(preparedModel),
    m_inputTensorDescs(inputTensorDescs),
    m_outputTensorDescs(outputTensorDescs),
    m_cachePath(cachePath),
    m_cacheVersion(cacheVersion),
    m_extensionConfig(extensionConfig),
    m_enableFp16(enableFp16),
    m_performance(performance),
    m_priority(priority) {
        m_executorid = GenRandom();
        m_autoUnloadRunner = OHOS::AppExecFwk::EventRunner::Create
            ("nnexecutor_autounload" + std::to_string(m_executorid));
        m_autoUnloadHandler = std::make_shared<OHOS::AppExecFwk::EventHandler>(m_autoUnloadRunner);
        auto autoUnloadTask = [this]() {
            DeinitModel("DelayUnload");
        };
        if (m_autoUnloadHandler != nullptr) {
            m_autoUnloadHandler->PostTask(autoUnloadTask,
                "nnexecutor_autounload" + std::to_string(m_executorid), AUTOUNLOAD_TIME);
        }

        GetModelID(m_originHiaiModelId);
    }

OH_NN_ReturnCode NNExecutor::GetInputDimVec() const
{
    std::vector<std::vector<uint32_t>> minInputDimsVec;
    std::vector<std::vector<uint32_t>> maxInputDimsVec;
    OH_NN_ReturnCode oldRet = m_preparedModel->GetInputDimRanges(minInputDimsVec, maxInputDimsVec);
    if (oldRet != OH_NN_SUCCESS) {
        LOGW("GetInputDimVec failed, current version don't support get input dim ranges.");
        return OH_NN_OPERATION_FORBIDDEN;
    }
    size_t inputSize = minInputDimsVec.size();
    if (inputSize != maxInputDimsVec.size()) {
        LOGE("GetInputDimVece failed, size of minInputDimsVec is not equal to maxInputDimsVec.");
        return OH_NN_INVALID_PARAMETER;
    }
    for (size_t i = 0; i < inputSize; i++) {
        std::vector<size_t> minInputDimVec;
        std::vector<size_t> maxInputDimVec;
        size_t minInputDimVecSize = minInputDimsVec[i].size();
        if (minInputDimVecSize != maxInputDimsVec[i].size()) {
            LOGE("GetInputDimVec failed, size of the min input dims is not equal to the max input"
                " dims of the %{public}zuth input.", i);
            return OH_NN_INVALID_PARAMETER;
        }
        for (size_t j = 0; j < minInputDimVecSize; j++) {
            minInputDimVec.emplace_back(static_cast<size_t>(minInputDimsVec[i][j]));
            maxInputDimVec.emplace_back(static_cast<size_t>(maxInputDimsVec[i][j]));
        }
        m_minInputDimsVec.emplace_back(minInputDimVec);
        m_maxInputDimsVec.emplace_back(maxInputDimVec);
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::GetInputDimRange(
    size_t inputIndex, size_t** minInputDims, size_t** maxInputDims, size_t* shapeNum) const
{
    if (minInputDims == nullptr) {
        LOGE("NNExecutor::GetInputDimRange failed, minInputDims is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (maxInputDims == nullptr) {
        LOGE("NNExecutor::GetInputDimRange failed, maxInputDims is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shapeNum == nullptr) {
        LOGE("NNExecutor::GetInputDimRange failed, shapeNum is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_minInputDimsVec.empty()) {
        OH_NN_ReturnCode ret = GetInputDimVec();
        if (ret != OH_NN_SUCCESS) {
            LOGE("NNExecutor::GetInputDimRange failed, GetInputDimVec failed.");
            return ret;
        }
    }

    if (inputIndex >= m_minInputDimsVec.size()) {
        LOGE("NNExecutor::GetInputDimRange failed, inputIndex[%{public}zu] is out of range.", inputIndex);
        return OH_NN_INVALID_PARAMETER;
    }

    *shapeNum = m_minInputDimsVec[inputIndex].size();
    if (*shapeNum != m_maxInputDimsVec[inputIndex].size()) {
        LOGE("NNExecutor::GetInputDimRange failed, size of the min input dims is not equal to the max input"
             " dims of the %{public}zuth input.", inputIndex);
        return OH_NN_INVALID_PARAMETER;
    }
    *minInputDims = m_minInputDimsVec[inputIndex].data();
    *maxInputDims = m_maxInputDimsVec[inputIndex].data();
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::GetOutputShape(uint32_t outputIndex, int32_t** shape, uint32_t* shapeNum) const
{
    if (outputIndex >= m_outputTensorDescs.size()) {
        LOGE("NNExecutor::GetOutputShape failed, outputIndex must be smaller than m_outputTensorDescs.size.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_outputTensorDescs[outputIndex].first == nullptr) {
        LOGE("NNExecutor::GetOutputShape failed, tensor desc of output %{public}u is nullptr.", outputIndex);
        return OH_NN_INVALID_PARAMETER;
    }

    auto tensorDesc = m_outputTensorDescs[outputIndex].first;
    size_t shapeNumTmp = 0;
    auto ret = tensorDesc->GetShape(shape, &shapeNumTmp);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNExecutor::GetOutputShape failed, failed to get shape from tensor desc.");
        return ret;
    }
    *shapeNum = static_cast<uint32_t>(shapeNumTmp);

    return OH_NN_SUCCESS;
}

size_t NNExecutor::GetInputNum() const
{
    return m_inputTensorDescs.size();
}

size_t NNExecutor::GetOutputNum() const
{
    return m_outputTensorDescs.size();
}

NN_TensorDesc* NNExecutor::CreateInputTensorDesc(size_t index) const
{
    if (index >= m_inputTensorDescs.size()) {
        LOGE("NNExecutor::CreateInputTensorDesc failed, index must be smaller than m_inputTensorDescs.size.");
        return nullptr;
    }
    if (m_inputTensorDescs[index].first == nullptr) {
        LOGE("NNExecutor::CreateInputTensorDesc failed, tensor desc of input %{public}zu is nullptr.", index);
        return nullptr;
    }

    TensorDesc* tensorDescImpl = new (std::nothrow) TensorDesc();
    if (tensorDescImpl == nullptr) {
        LOGE("NNExecutor::CreateInputTensorDesc failed, failed to create tensor desc.");
        return nullptr;
    }

    // Copy the member attributes to new tensor description
    *tensorDescImpl = *(m_inputTensorDescs[index].first.get());

    return reinterpret_cast<NN_TensorDesc*>(tensorDescImpl);
}

NN_TensorDesc* NNExecutor::CreateOutputTensorDesc(size_t index) const
{
    if (index >= m_outputTensorDescs.size()) {
        LOGE("NNExecutor::CreateOutputTensorDesc failed, index must be smaller than m_outputTensorDescs.size.");
        return nullptr;
    }
    if (m_outputTensorDescs[index].first == nullptr) {
        LOGE("NNExecutor::CreateOutputTensorDesc failed, tensor desc of output %{public}zu is nullptr.", index);
        return nullptr;
    }

    TensorDesc* tensorDescImpl = new (std::nothrow) TensorDesc();
    if (tensorDescImpl == nullptr) {
        LOGE("NNExecutor::CreateOutputTensorDesc failed, failed to create tensor desc.");
        return nullptr;
    }

    // Copy the member attributes to new tensor description
    *tensorDescImpl = *(m_outputTensorDescs[index].first.get());

    return reinterpret_cast<NN_TensorDesc*>(tensorDescImpl);
}

OH_NN_ReturnCode NNExecutor::SetExtensionConfig(const std::unordered_map<std::string, std::vector<char>>& configs)
{
    if (m_executorConfig == nullptr) {
        m_executorConfig = new (std::nothrow) ExecutorConfig();
        if (m_executorConfig == nullptr) {
            LOGE("[NNExecutor] SetExtensionConfig, m_executorConfig create failed.");
            return OH_NN_FAILED;
        }
    }

    if (configs.size() > EXTENSION_MAX_SIZE) {
        LOGE("[NNExecutor] SetExtensionConfig, configs size more than 200.");
        return OH_NN_FAILED;
    }

    for (auto config : configs) {
        char* configData = reinterpret_cast<char*>(config.second.data());
        if (configData == nullptr) {
            LOGE("[NNExecutor] SetExtensionConfig, key: %s, configData is nullptr.", config.first.c_str());
            return OH_NN_FAILED;
        }

        if (!config.first.compare("callingPid")) {
            m_executorConfig->callingPid = std::atoi(configData);
            LOGD("[NNExecutor] SetExtensionConfig, callingPid: %{public}d.", m_executorConfig->callingPid);
        }

        if (!config.first.compare("hiaiModelId")) {
            m_executorConfig->hiaiModelId = std::atoi(configData);
            LOGD("[NNExecutor] SetExtensionConfig, hiaiModelId: %{public}d.", m_executorConfig->hiaiModelId);
        }

        if (!config.first.compare("isNeedModelLatency")) {
            m_executorConfig->isNeedModelLatency = static_cast<bool>(*configData);
            LOGD("[NNExecutor] SetExtensionConfig, isNeedModelLatency: %{public}d.",
                m_executorConfig->isNeedModelLatency);
        }
    }

    return OH_NN_SUCCESS;
}

ExecutorConfig* NNExecutor::GetExecutorConfig() const
{
    return m_executorConfig;
}

OH_NN_ReturnCode NNExecutor::SetOnRunDone(NN_OnRunDone onRunDone)
{
    LOGE("NNExecutor::SetOnRunDone failed, SetOnRunDone is not supported.");
    return OH_NN_OPERATION_FORBIDDEN;
}

OH_NN_ReturnCode NNExecutor::SetOnServiceDied(NN_OnServiceDied onServiceDied)
{
    LOGE("NNExecutor::SetOnServiceDied failed, SetOnServiceDied is not supported.");
    return OH_NN_OPERATION_FORBIDDEN;
}


OH_NN_ReturnCode NNExecutor::ReloadAippModel(uint32_t modelId)
{
    if (Reload() != OH_NN_SUCCESS) {
        return OH_NN_INVALID_PARAMETER;
    }
    
    auto _ret = GetModelID(modelId);
    LOGI("AutoReload pid=%{public}d originHiaiModelId=%{public}d hiaiModelId=%{public}d",
        getpid(), m_originHiaiModelId, modelId);
    if (_ret != OH_NN_SUCCESS) {
        LOGW("GetModelID failed, some error happen when get model id for device.");
    }

    _ret = ReinitScheduling(modelId, &m_executorConfig->isNeedModelLatency, m_cachePath.c_str());
    if (_ret != OH_NN_SUCCESS) {
        LOGW("ReinitScheduling failed, some error happen when ReinitScheduling model.");
    }

    _ret = SetDeinitModelCallBack();
    if (_ret != OH_NN_SUCCESS) {
        LOGW("SetDeinitModelCallBack failed, some error happen when ReinitScheduling model.");
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::RunAippModel(NN_Tensor* inputTensors[], size_t inputSize,
                                          NN_Tensor* outputTensors[], size_t outputSize, const char* aippStrings)
{
    std::vector<NN_Tensor*> inputTensorsVec;
    for (size_t i = 0; i < inputSize; ++i) {
        if (inputTensors[i] == nullptr) {
            LOGE("RunSyncWithAipp failed, failed to check input");
            return OH_NN_INVALID_PARAMETER;
        }
        inputTensorsVec.emplace_back(inputTensors[i]);
    }
    
    std::vector<NN_Tensor*> outputTensorsVec;
    for (size_t i = 0; i < outputSize; ++i) {
        if (outputTensors == nullptr) {
            LOGE("RunSyncWithAipp failed, failed to check output");
            return OH_NN_INVALID_PARAMETER;
        }
        outputTensorsVec.emplace_back(outputTensors[i]);
    }

    std::vector<std::vector<int32_t>> outputsDims;
    std::vector<bool> isSufficientDataBuffer;
    OH_NN_ReturnCode ret = m_preparedModel->SetAippString(aippStrings);
    if (ret != OH_NN_SUCCESS) {
        LOGE("RunSyncWithAipp failed, failed to set aipp para in prepared model.");
        return ret;
    }

    ret = m_preparedModel->Run(inputTensorsVec, outputTensorsVec, outputsDims, isSufficientDataBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("RunSyncWithAipp failed, failed to run in prepared model.");
        return ret;
    }

    // Set the output NNTensor2_0's dimensions from output IOTensor if it is dynamic
    // NNTensor2_0::SetDimensions will check if the tensor buffer is enough for the new dimensions.
    if (outputsDims.size() != outputSize) {
        LOGE("RunSyncWithAipp failed, size of outputsDims is not equal to outputTensors.");
        return OH_NN_INVALID_PARAMETER;
    }

    for (size_t i = 0; i < outputSize; ++i) {
        ret = m_outputTensorDescs[i].first->SetShape(outputsDims[i].data(), outputsDims[i].size());
        if (ret != OH_NN_SUCCESS) {
            LOGE("RunSyncWithAipp failed, error happened when setting inner output tensor's dimensions,"
                " output id: %zu.", i);
            return ret;
        }
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::RunSyncWithAipp(NN_Tensor* inputTensors[], size_t inputSize,
                                             NN_Tensor* outputTensors[], size_t outputSize, const char* aippStrings)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    {
        uint32_t modelId;
        GetModelID(modelId);
        m_autoUnloadHandler->RemoveTask("nnexecutor_autounload" + std::to_string(m_executorid));
        if (m_inputTensorDescs.size() != inputSize) {
            LOGE("RunSyncWithAipp failed, inputSize:%{public}zu is not equal to model inputsize:%{public}zu",
                inputSize, m_inputTensorDescs.size());
            return OH_NN_INVALID_PARAMETER;
        }
        if (m_outputTensorDescs.size() != outputSize) {
            LOGE("RunSyncWithAipp failed, outputSize:%{public}zu is not equal to model output "
                "size:%{public}zu", outputSize, m_outputTensorDescs.size());
            return OH_NN_INVALID_PARAMETER;
        }

        OH_NN_ReturnCode ret {OH_NN_FAILED};
        if (m_preparedModel == nullptr) {
            ret = ReloadAippModel(modelId);
            if (ret != OH_NN_SUCCESS) {
                return ret;
            }
        }
        
        ret = CheckInputDimRanges(inputTensors, inputSize);
        if (ret != OH_NN_OPERATION_FORBIDDEN && ret != OH_NN_SUCCESS) {
            LOGE("RunSyncWithAipp failed, failed to check input dim ranges.");
            return ret;
        }

        ret = RunAippModel(inputTensors, inputSize, outputTensors, outputSize, aippStrings);
        if (ret != OH_NN_SUCCESS) {
            return ret;
        }
    }
    auto autoUnloadTask = [this]() {
        DeinitModel("DelayUnload");
    };
    m_autoUnloadHandler->PostTask(autoUnloadTask,
        "nnexecutor_autounload" + std::to_string(m_executorid), AUTOUNLOAD_TIME);

    return OH_NN_SUCCESS;
}

void ReleaseDescShape(std::vector<SerializedTensorDesc>& immediateTensorDescs)
{
    for (auto desc : immediateTensorDescs) {
        delete[] desc.m_shape;
    }
    immediateTensorDescs.clear();
}

OH_NN_ReturnCode NNExecutor::DeserializedTensorsFromBuffer(
    const Buffer& buffer, std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& tensorDescs)
{
    std::vector<SerializedTensorDesc> immediateTensorDescs;
    const char* ptr = static_cast<const char*>(buffer.data);
    const char* end = ptr + buffer.length;
    while (ptr < end) {
        SerializedTensorDesc desc;

        auto memRet = memcpy_s(&desc.m_dataType, SIZE_OF_DATATYPE, ptr, sizeof(desc.m_dataType));
        if (memRet != EOK) {
            LOGE("[NNExecutor] DeserializedTensorsFromBuffer failed, failed to memcpy_s data type.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += sizeof(desc.m_dataType);

        memRet = memcpy_s(&desc.m_format, SIZE_OF_FORMAT, ptr, sizeof(desc.m_format));
        if (memRet != EOK) {
            LOGE("[NNExecutor] DeserializedTensorsFromBuffer failed, failed to memcpy_s format.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += sizeof(desc.m_format);

        memRet = memcpy_s(&desc.m_tensorType, SIZE_OF_TENSOR_TYPE, ptr, sizeof(desc.m_tensorType));
        if (memRet != EOK) {
            LOGE("[NNExecutor] DeserializedTensorsFromBuffer failed, failed to memcpy_s tensor type.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += sizeof(desc.m_tensorType);

        memRet = memcpy_s(&desc.m_shapeNum, SIZE_OF_SHAPE_NUM, ptr, sizeof(desc.m_shapeNum));
        if (memRet != EOK) {
            LOGE("[NNExecutor] DeserializedTensorsFromBuffer failed, failed to memcpy_s shape num.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_MEMORY_ERROR;
        }
        ptr += sizeof(desc.m_shapeNum);

        desc.m_shape = new (std::nothrow) int32_t[desc.m_shapeNum];
        if (desc.m_shape == nullptr) {
            LOGE("[NNExecutor] DeserializedTensorsFromBuffer failed, failed to create shape buffer.");
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_NULL_PTR;
        }
        memRet = memcpy_s(desc.m_shape, desc.m_shapeNum * sizeof(int32_t), ptr, desc.m_shapeNum * sizeof(int32_t));
        if (memRet != EOK) {
            LOGE("[NNExecutor] DeserializedTensorsFromBuffer failed, failed to memcpy_s shape.");
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
            LOGE("[NNExecutor] DeserializedTensorsFromBuffer failed, failed to create tensor desc.");
            tensorDescs.clear();
            ReleaseDescShape(immediateTensorDescs);
            return OH_NN_NULL_PTR;
        }
        ret = immediateTensorDesc.CopyToTensorDesc(*(tensorDescPair.first.get()));
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNExecutor] DeserializedTensorsFromBuffer failed, error happened when copying "
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

OH_NN_ReturnCode NNExecutor::Reload()
{
    if (m_cachePath.empty()) {
        LOGE("[NNExecutor] RestoreFromCacheFile failed, path is empty.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_cacheVersion == INVALID_CAHCE_VERSION) {
        LOGE("[NNExecutor] RestoreFromCacheFile failed, cache version is invalid. Please set a valid cache version.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_preparedModel != nullptr) {
        LOGE("[NNExecutor] RestoreFromCacheFile failed, m_preparedModel is not nullptr.");
        return OH_NN_FAILED;
    }

    NNCompiledCache compiledCache;
    OH_NN_ReturnCode ret = compiledCache.SetBackend(m_backendID);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNExecutor] RestoreFromCacheFile failed, fail to set backend.");
        return ret;
    }

    std::vector<Buffer> caches;
    compiledCache.SetModelName(m_extensionConfig.modelName);
    ret = compiledCache.Restore(m_cachePath, m_cacheVersion, caches);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNExecutor] RestoreFromCacheFile failed, error happened when restoring model cache.");
        compiledCache.ReleaseCacheBuffer(caches);
        return ret;
    }

    size_t cacheNum = caches.size();
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> inputTensorDescs;
    ret = DeserializedTensorsFromBuffer(caches[cacheNum - CACHE_INPUT_TENSORDESC_OFFSET], inputTensorDescs);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNExecutor] RestoreFromCacheFile failed, error happened when deserializing input tensor desc.");
        compiledCache.ReleaseCacheBuffer(caches);
        return ret;
    }

    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> outputTensorDescs;
    ret = DeserializedTensorsFromBuffer(caches[cacheNum - CACHE_OUTPUT_TENSORDESC_OFFSET], outputTensorDescs);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNExecutor] RestoreFromCacheFile failed, error happened when deserializing output tensor desc.");
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
        LOGE("[NNExecutor] RestoreFromCacheFile failed, error happened when preparing model from cache.");
        compiledCache.ReleaseCacheBuffer(caches);
        return ret;
    }

    compiledCache.ReleaseCacheBuffer(caches);

    m_inputTensorDescs = inputTensorDescs;
    m_outputTensorDescs = outputTensorDescs;
    LOGI("[NNExecutor] Restore model cache successfully.");
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::RunSync(NN_Tensor* inputTensors[], size_t inputSize,
    NN_Tensor* outputTensors[], size_t outputSize)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    {
        uint32_t modelId;
        GetModelID(modelId);
        m_autoUnloadHandler->RemoveTask("nnexecutor_autounload" + std::to_string(m_executorid));
        if (m_inputTensorDescs.size() != inputSize) {
            LOGE("NNExecutor::RunSync failed, inputSize:%{public}zu is not equal to model input size:%{public}zu",
                inputSize, m_inputTensorDescs.size());
            return OH_NN_INVALID_PARAMETER;
        }
        if (m_outputTensorDescs.size() != outputSize) {
            LOGE("NNExecutor::RunSync failed, outputSize:%{public}zu is not equal to model output size:%{public}zu",
                outputSize, m_outputTensorDescs.size());
            return OH_NN_INVALID_PARAMETER;
        }

        if (m_preparedModel == nullptr) {
            if (Reload() != OH_NN_SUCCESS) {
                return OH_NN_INVALID_PARAMETER;
            }
            auto _ret = GetModelID(modelId);
            LOGI("AutoReload pid=%{public}ld originHiaiModelId=%{public}u hiaiModelId=%{public}u",
                static_cast<long>(getpid()), m_originHiaiModelId, modelId);
            if (_ret != OH_NN_SUCCESS) {
                LOGW("GetModelID failed, some error happen when get model id for device.");
            }
            _ret = ReinitScheduling(modelId, &m_executorConfig->isNeedModelLatency, m_cachePath.c_str());
            if (_ret != OH_NN_SUCCESS) {
                LOGW("ReinitScheduling failed, some error happen when ReinitScheduling model.");
            }
            _ret = SetDeinitModelCallBack();
            if (_ret != OH_NN_SUCCESS) {
                LOGW("SetDeinitModelCallBack failed, some error happen when ReinitScheduling model.");
            }
        }

        OH_NN_ReturnCode ret {OH_NN_FAILED};
        ret = CheckInputDimRanges(inputTensors, inputSize);
        if (ret != OH_NN_OPERATION_FORBIDDEN && ret != OH_NN_SUCCESS) {
            LOGE("NNExecutor::RunSync failed, failed to check input dim ranges.");
            return ret;
        }

        OHOS::NeuralNetworkRuntime::IOTensor tensor;
        std::vector<NN_Tensor*> inputTensorsVec;
        for (size_t i = 0; i < inputSize; ++i) {
            if (inputTensors[i] == nullptr) {
                LOGE("NNExecutor::RunSync failed, input[%{public}zu] is nullptr.", i);
                return OH_NN_INVALID_PARAMETER;
            }

            inputTensorsVec.emplace_back(inputTensors[i]);
        }

        std::vector<NN_Tensor*> outputTensorsVec;
        for (size_t i = 0; i < outputSize; ++i) {
            if (outputTensors[i] == nullptr) {
                LOGE("NNExecutor::RunSync failed, output[%{public}zu] is nullptr.", i);
                return OH_NN_INVALID_PARAMETER;
            }
            outputTensorsVec.emplace_back(outputTensors[i]);
        }

        std::vector<std::vector<int32_t>> outputsDims;
        std::vector<bool> isSufficientDataBuffer;

        ret = m_preparedModel->Run(inputTensorsVec, outputTensorsVec, outputsDims, isSufficientDataBuffer);
        if (ret != OH_NN_SUCCESS) {
            LOGE("NNExecutor::RunSync failed, failed to run in prepared model.");
            return ret;
        }

        // Set the output NNTensor2_0's dimensions from output IOTensor if it is dynamic.
        // NNTensor2_0::SetDimensions will check if the tensor buffer is enough for the new dimensions.
        if (outputsDims.size() != outputSize) {
            LOGE("NNExecutor::RunSync failed, size of outputsDims is not equal to outputTensors.");
            return OH_NN_INVALID_PARAMETER;
        }
        for (size_t i = 0; i < outputSize; ++i) {
            NNTensor2_0* nnTensor = reinterpret_cast<NNTensor2_0*>(outputTensors[i]);
            TensorDesc* nnTensorDesc = nnTensor->GetTensorDesc();
            if (nnTensorDesc == nullptr) {
                LOGE("NNExecutor::RunSync failed, failed to get desc from tensor.");
                return OH_NN_NULL_PTR;
            }
            ret = nnTensorDesc->SetShape(outputsDims[i].data(), outputsDims[i].size());
            if (ret != OH_NN_SUCCESS) {
                LOGE("NNExecutor::RunSync failed, error happened when setting output tensor's dimensions,"
                    " output id: %zu.", i);
                return ret;
            }
            ret = m_outputTensorDescs[i].first->SetShape(outputsDims[i].data(), outputsDims[i].size());
            if (ret != OH_NN_SUCCESS) {
                LOGE("NNExecutor::RunSync failed, error happened when setting inner output tensor's dimensions,"
                    " output id: %zu.", i);
                return ret;
            }
        }
    }
    auto autoUnloadTask = [this]() {
        DeinitModel("DelayUnload");
    };
    m_autoUnloadHandler->PostTask(autoUnloadTask,
        "nnexecutor_autounload" + std::to_string(m_executorid), AUTOUNLOAD_TIME);

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::RunAsync(NN_Tensor* inputTensors[], size_t inputSize,
    NN_Tensor* outputTensors[], size_t outputSize, int32_t timeout, void* userData)
{
    LOGE("NNExecutor::RunAsync failed, RunAsync is not supported.");
    return OH_NN_OPERATION_FORBIDDEN;
}

OH_NN_ReturnCode NNExecutor::GetModelID(uint32_t& modelId) const
{
    if (m_preparedModel != nullptr) {
        OH_NN_ReturnCode ret = m_preparedModel->GetModelID(modelId);
        if (ret != OH_NN_SUCCESS) {
            LOGE("GetModelID failed, some error happen when get model id for device.");
            return ret;
        }
        return OH_NN_SUCCESS;
    }

    return OH_NN_OPERATION_FORBIDDEN;
}

size_t NNExecutor::GetBackendID()
{
    return m_backendID;
}

OH_NN_ReturnCode NNExecutor::CheckInputDimRanges(NN_Tensor* inputTensors[], size_t inputSize)
{
    std::vector<std::vector<uint32_t>> minInputDims;
    std::vector<std::vector<uint32_t>> maxInputDims;
    OH_NN_ReturnCode oldRet = m_preparedModel->GetInputDimRanges(minInputDims, maxInputDims);
    if (oldRet != OH_NN_SUCCESS) {
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (inputSize != minInputDims.size()) {
        LOGE("NNExecutor::CheckInputDimRanges failed, size of minInputDims:%{public}zu is not equal to "
             "inputSize:%{public}zu.", minInputDims.size(), inputSize);
        return OH_NN_INVALID_PARAMETER;
    }

    if (inputSize != maxInputDims.size()) {
        LOGE("NNExecutor::CheckInputDimRanges failed, size of maxInputDims:%{public}zu is not equal to "
             "inputSize:%{public}zu.", maxInputDims.size(), inputSize);
        return OH_NN_INVALID_PARAMETER;
    }

    const NNTensor2_0* nnTensor = nullptr;
    OH_NN_ReturnCode ret {OH_NN_FAILED};
    for (size_t i = 0; i < inputSize; ++i) {
        const std::vector<uint32_t>& minSingleInputDims = minInputDims[i];
        const std::vector<uint32_t>& maxSingleInputDims = maxInputDims[i];
        nnTensor = reinterpret_cast<const NNTensor2_0*>(inputTensors[i]);
        if (nnTensor == nullptr) {
            LOGE("NNExecutor::CheckInputDimRanges failed, input %{public}zu is nullptr.", i);
            return OH_NN_NULL_PTR;
        }
        ret = nnTensor->CheckDimRanges(minSingleInputDims, maxSingleInputDims);
        if (ret != OH_NN_SUCCESS) {
            LOGE("NNExecutor::CheckInputDimRanges failed, failed to check input dim ranges of input %{public}zu", i);
            return ret;
        }
    }

    return OH_NN_SUCCESS;
}

bool NNExecutor::CompareAttribute(
    const std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>& tensorDesc, const NNTensor& tensor) const
{
    OH_NN_DataType dataType;
    auto ret = tensorDesc.first->GetDataType(&dataType);
    if (ret != OH_NN_SUCCESS) {
        LOGE("CompareAttribute failed, failed to get data type from tensor desc.");
        return false;
    }
    if (dataType != tensor.GetDataType()) {
        LOGI("Tensors have different data type: %d and %d.", dataType, tensor.GetDataType());
        return false;
    }

    int32_t* shape {nullptr};
    size_t shapeNum {0};
    ret = tensorDesc.first->GetShape(&shape, &shapeNum);
    if (ret != OH_NN_SUCCESS) {
        LOGE("CompareAttribute failed, failed to get shape from tensor desc.");
        return false;
    }
    const std::vector<int32_t> dimensions = tensor.GetDimensions();
    if (shapeNum != dimensions.size()) {
        LOGI("Tensors have differents dimension counts: %zu and %zu.", shapeNum, dimensions.size());
        return false;
    }

    size_t dimensionsSize = dimensions.size();
    for (size_t i = 0; i < dimensionsSize; i++) {
        if ((shape[i] != -1) && (shape[i] != dimensions[i])) {
            LOGI("Tensors have different dimension: dimension index: %zu, dimension value: %d and %d.",
                 i, shape[i], dimensions[i]);
            return false;
        }
    }

    if (tensorDesc.second != tensor.GetType()) {
        LOGI("Tensors have different type: %{public}d and %{public}d.", tensorDesc.second, tensor.GetType());
        return false;
    }

    return true;
}

OH_NN_ReturnCode NNExecutor::BuildInputTensor(uint32_t index, const OH_NN_Tensor& nnTensor,
    std::shared_ptr<NNTensor> inputTensor) const
{
    // Note: inputs have only shapes info.
    if (index >= m_inputTensorDescs.size()) {
        LOGE("BuildInputTensor failed, input index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_inputTensorDescs[index].first == nullptr) {
        LOGE("BuildInputTensor failed, tensor desc of input %{public}u is nullptr.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    // Build a tensor from nnTensor.
    auto ret = inputTensor->BuildFromOHNNTensor(nnTensor);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildInputTensor failed, please check input nnTensor.");
        return ret;
    }

    if (inputTensor->IsDynamicShape()) {
        LOGE("BuildInputTensor failed, input nnTensor should has certain dimensions which cannot contain -1.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_Format format;
    ret = m_inputTensorDescs[index].first->GetFormat(&format);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildInputTensor failed, failed to get tensor format from desc.");
        return ret;
    }
    inputTensor->SetFormat(format);

    if (!CompareAttribute(m_inputTensorDescs[index], *inputTensor)) {
        LOGE("BuildInputTensor failed, input has different attributes from the one in the constructed model.");
        return OH_NN_INVALID_PARAMETER;
    }

    const char* name {nullptr};
    ret = m_inputTensorDescs[index].first->GetName(&name);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildInputTensor failed, failed to get tensor name from desc.");
        return ret;
    }
    inputTensor->SetName(name);
    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode NNExecutor::SetInputTensorWithCurrentBuffer(uint32_t index,
    std::shared_ptr<NNTensor> inputTensor, const void* buffer, size_t dataLength, size_t curBufferLength)
{
    void* curBuffer = m_inputTensors[index].tensor->GetBuffer();
    errno_t status = memcpy_s(curBuffer, dataLength, buffer, dataLength);
    // Current buffer inside m_inputTensors is managed by executor, no need to release if memcpy failed.
    if (status != EOK) {
        LOGE("SetInputTensorWithCurrentBuffe failed, copy data from user buffer to device buffer failed. "
             "Error code: %d.", status);
        return OH_NN_MEMORY_ERROR;
    }

    // Set the new tensor with the buffer of current tensor
    inputTensor->SetBuffer(curBuffer, curBufferLength);

    // The memory is reused here. Thus, current tensor's buffer must set to nullptr, in case the memory is released
    // twice.
    m_inputTensors[index].tensor->SetBuffer(nullptr, 0);

    // Set to the new tensor, and release current one.
    m_inputTensors[index].tensor = inputTensor;
    return OH_NN_SUCCESS;
}


void NNExecutor::SetInputTensorWithNewBuffer(uint32_t index,
    std::shared_ptr<NNTensor> inputTensor, const void* inputBuffer, size_t length, bool isInnerMem)
{
    // Release the memory inside the tensor first, if it is allocated by Executor during SetInput().
    if (m_inputTensors.find(index) != m_inputTensors.end()) {
        if (m_inputTensors[index].isInnerMem) {
            void* curBuffer = m_inputTensors[index].tensor->GetBuffer();
            m_device->ReleaseBuffer(curBuffer);
        }
        // Set current tensor's buffer to nullptr in case the NNTensor release the driver memory in destruction.
        m_inputTensors[index].tensor->SetBuffer(nullptr, 0);
    }

    // Set new input tensor data buffer
    inputTensor->SetBuffer(inputBuffer, length);

    // Create or update the input tensor
    ExeTensor exeTensor{inputTensor, nullptr, 0, isInnerMem};
    m_inputTensors[index] = exeTensor;
}


OH_NN_ReturnCode NNExecutor::CheckInputDimRanges(uint32_t index, const OH_NN_Tensor& nnTensor) const
{
    std::vector<std::vector<uint32_t>> minInputDims;
    std::vector<std::vector<uint32_t>> maxInputDims;
    auto ret = m_preparedModel->GetInputDimRanges(minInputDims, maxInputDims);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Get the dimension ranges of input %u failed. ErrorCode=%d", index, ret);
        return ret;
    }

    if (index >= minInputDims.size()) {
        LOGE("index is %u, which exceeds the size of minInputDims:%zu.", index, minInputDims.size());
        return OH_NN_INVALID_PARAMETER;
    }

    if (index >= maxInputDims.size()) {
        LOGE("index is %u, which exceeds the size of maxInputDims:%zu.", index, maxInputDims.size());
        return OH_NN_INVALID_PARAMETER;
    }

    const std::vector<uint32_t>& minSingleInputDims = minInputDims[index];
    const std::vector<uint32_t>& maxSingleInputDims = maxInputDims[index];

    std::vector<int32_t> tensorShape = ConstructVectorFromArray(nnTensor.dimensions, nnTensor.dimensionCount);
    size_t tensorShapeSize = tensorShape.size();
    if (minSingleInputDims.size() != tensorShapeSize || maxSingleInputDims.size() != tensorShapeSize) {
        LOGE("Size of minSingleInputDims, maxSingleInputDims and tensorShape of input %u are not equal.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    for (size_t j = 0; j < tensorShapeSize; ++j) {
        // Dimensions cannot be negative
        if (tensorShape[j] < 0) {
            LOGE("Dimension %zu of input %u is %d.", j, index, tensorShape[j]);
            return OH_NN_INVALID_PARAMETER;
        }
        uint32_t dim = static_cast<uint32_t>(tensorShape[j]);
        if (dim < minSingleInputDims[j] || dim > maxSingleInputDims[j]) {
            LOGE("Dimension %zu of input %u is %u, which is out of range [%u, %u]",
                j, index, dim, minSingleInputDims[j], maxSingleInputDims[j]);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode NNExecutor::SetInput(uint32_t index, const OH_NN_Tensor& nnTensor, const void* buffer, size_t length)
{
    auto nnRet = CheckInputDimRanges(index, nnTensor);
    if (nnRet == OH_NN_OPERATION_FORBIDDEN) {
        LOGI("Skip input dimension bounds check.");
    } else if (nnRet != OH_NN_SUCCESS) {
        LOGE("SetInput failed, Check the range of the %uth input dimension ranges failed.", index);
        return nnRet;
    }

    std::shared_ptr<NNTensor> inputTensor = CreateSharedPtr<NNTensor>();
    if (inputTensor == nullptr) {
        LOGE("SetInput failed, error happened when creating NNTensor.");
        return OH_NN_MEMORY_ERROR;
    }

    auto ret = BuildInputTensor(index, nnTensor, inputTensor);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetInput failed, please check input index or nnTensor.");
        return ret;
    }

    // dataLength will be larger than 0 after BuildInputTensor()
    size_t dataLength = inputTensor->GetDataLength();
    if (length == 0 || length < dataLength) {
        LOGE("SetInput failed, the given buffer length is too small to store the input nnTensor data.");
        return OH_NN_INVALID_PARAMETER;
    }

    // Get length of current buffer if it is allocate by SetInput() before.
    size_t curBufferLength = 0;
    if ((m_inputTensors.find(index) != m_inputTensors.end()) && (m_inputTensors[index].isInnerMem)) {
        curBufferLength = m_inputTensors[index].tensor->GetBufferLength();
    }

    // (dataLength <= curBufferLength) returns true if and only if current buffer is allocated by SetInput() before
    // and is larger than user buffer.
    if (dataLength <= curBufferLength) {
        ret = SetInputTensorWithCurrentBuffer(index, inputTensor, buffer, dataLength, curBufferLength);
        if (ret != OH_NN_SUCCESS) {
            LOGE("SetInput failed, error happened when setting input with current buffer.");
            return ret;
        }
        m_isRun = false;
        return OH_NN_SUCCESS;
    }

    /**
     * Buffer needs to allocated or reallocated if:
     *
     * - Current buffer is not enough.
     * - SetInput() has not been called for the input before.
     * - The buffer held in m_inputTensors is allocated and set by CreateInputMemory() and SetInputFromMemory().
     */
    void* inputBuffer = m_device->AllocateTensorBuffer(length, inputTensor);
    if (inputBuffer == nullptr) {
        LOGE("SetInput failed, error happened when allocating input device buffer.");
        return OH_NN_MEMORY_ERROR;
    }

    errno_t status = memcpy_s(inputBuffer, dataLength, buffer, dataLength);
    if (status != EOK) {
        LOGE("SetInput failed, copy data from user buffer failed. Error code: %d.", status);
        m_device->ReleaseBuffer(inputBuffer);
        return OH_NN_MEMORY_ERROR;
    }

    SetInputTensorWithNewBuffer(index, inputTensor, inputBuffer, length, true);
    m_isRun = false;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::SetInputFromMemory(
    uint32_t index, const OH_NN_Tensor& nnTensor, const OH_NN_Memory& memory)
{
    auto nnRet = CheckInputDimRanges(index, nnTensor);
    if (nnRet == OH_NN_OPERATION_FORBIDDEN) {
        LOGI("Skip input dimension bounds check.");
    } else if (nnRet != OH_NN_SUCCESS) {
        LOGE("SetInputFromMemory failed, Check the range of the %uth input dimension ranges failed.", index);
        return nnRet;
    }

    // Build a input tensor
    std::shared_ptr<NNTensor> inputTensor = CreateSharedPtr<NNTensor>();
    if (inputTensor == nullptr) {
        LOGE("SetInputFromMemory failed, error happened when creating NNTensor.");
        return OH_NN_MEMORY_ERROR;
    }

    auto ret = BuildInputTensor(index, nnTensor, inputTensor);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetInputFromMemory failed, please check input index or nnTensor");
        return ret;
    }

    // check data length
    size_t dataLength = inputTensor->GetDataLength();
    if (memory.length == 0 || memory.length < dataLength) {
        LOGE("SetInputFromMemory failed,"
             " the length in the given memory is too small to store the input nnTensor data.");
        return OH_NN_INVALID_PARAMETER;
    }

    SetInputTensorWithNewBuffer(index, inputTensor, const_cast<const void*>(memory.data), memory.length, false);
    m_isRun = false;
    return OH_NN_SUCCESS;
}

std::shared_ptr<NNTensor> NNExecutor::BuildNNTensorFromDesc(
    const std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>& tensorDesc)
{
    std::shared_ptr<NNTensor> tensor = CreateSharedPtr<NNTensor>();
    if (tensor == nullptr) {
        LOGE("BuildNNTensorFromDesc failed, error happened when creating NNTensor.");
        return nullptr;
    }

    // Build a tensor from nnTensor.
    NN_TensorDesc* tensorDescCast = reinterpret_cast<NN_TensorDesc*>(tensorDesc.first.get());
    auto ret = tensor->BuildFromTensorDesc(tensorDescCast);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildNNTensorFromDesc failed, please check input nnTensor.");
        return nullptr;
    }

    OH_NN_Format format;
    tensorDesc.first->GetFormat(&format);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildNNTensorFromDesc failed, failed to get tensor format from desc.");
        return nullptr;
    }
    tensor->SetFormat(format);

    ret = tensor->SetTensorType(tensorDesc.second);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildNNTensorFromDesc failed, failed to set tensor type.");
        return nullptr;
    }

    if (!CompareAttribute(tensorDesc, *tensor)) {
        LOGE("BuildNNTensorFromDesc failed, input has different attributes from the one in the constructed model.");
        return nullptr;
    }

    const char* name {nullptr};
    ret = tensorDesc.first->GetName(&name);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildNNTensorFromDesc failed, failed to get tensor name from desc.");
        return nullptr;
    }
    tensor->SetName(name);
    return tensor;
}

OH_NN_ReturnCode NNExecutor::SetOutput(uint32_t index, void* buffer, size_t length)
{
    if (index >= m_outputTensorDescs.size()) {
        LOGE("SetOutput failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_outputTensorDescs[index].first == nullptr) {
        LOGE("NNExecutor::SetOutput failed, tensor desc of output %{public}u is nullptr.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    size_t dataLength {0};
    auto ret = m_outputTensorDescs[index].first->GetByteSize(&dataLength);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetOutputFromMemory failed, failed to get byte size from tensor desc.");
        return ret;
    }
    if (length == 0 || length < dataLength) {
        LOGE("SetOutput failed, the given buffer length is too small to store the output tensor data.");
        return OH_NN_INVALID_PARAMETER;
    }

    // If output tensor does not exist, or inner device buffer size is not enough,
    // or device buffer is set by SetOutputFromMemory() before,
    // allocate a new device buffer and set it to output tensor, and update the user buffer.
    if (m_outputTensors.find(index) != m_outputTensors.end()) {
        if (m_outputTensors[index].isInnerMem) {
            size_t curBufferLength =  m_outputTensors[index].tensor->GetBufferLength();
            if (length <= curBufferLength) {
                // If current device buffer size is enough, only update the user buffer.
                m_outputTensors[index].userBuffer = buffer;
                m_outputTensors[index].userBufferLength = length;
                m_isRun = false;
                return OH_NN_SUCCESS;
            } else {
                // If current device buffer size is not enough,
                // release current device buffer and then allocate a new one below.
                void* curBuffer = m_outputTensors[index].tensor->GetBuffer();
                m_device->ReleaseBuffer(curBuffer);
            }
        }
    } else {
        // If output tensor does not exist, create a new null output tensor.
        ExeTensor exeTensor;
        m_outputTensors[index] = exeTensor;
        m_outputTensors[index].tensor = BuildNNTensorFromDesc(m_outputTensorDescs[index]);
        if (m_outputTensors[index].tensor == nullptr) {
            LOGE("SetOutput failed, failed to build nntensor from desc.");
            return OH_NN_NULL_PTR;
        }
    }

    void* deviceOutputBuffer = m_device->AllocateTensorBuffer(length, m_outputTensorDescs[index].first);
    if (deviceOutputBuffer == nullptr) {
        LOGE("SetOutput failed, allocating output device buffer failed.");
        return OH_NN_MEMORY_ERROR;
    }

    m_outputTensors[index].tensor->SetBuffer(deviceOutputBuffer, length);
    m_outputTensors[index].userBuffer = buffer;
    m_outputTensors[index].userBufferLength = length;
    m_outputTensors[index].isInnerMem = true;
    m_isRun = false;
    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode NNExecutor::SetOutputFromMemory(uint32_t index, const OH_NN_Memory& memory)
{
    if (index >= m_outputTensorDescs.size()) {
        LOGE("SetOutputFromMemory failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_outputTensorDescs[index].first == nullptr) {
        LOGE("NNExecutor::SetOutputFromMemory failed, tensor desc of output %{public}u is nullptr.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    size_t dataLength {0};
    auto ret = m_outputTensorDescs[index].first->GetByteSize(&dataLength);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetOutputFromMemory failed, failed to get byte size from tensor desc.");
        return ret;
    }
    if (memory.length == 0 || memory.length < dataLength) {
        LOGE("SetOutputFromMemory failed, the memory is too small to store the output tensor data.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_outputTensors.find(index) != m_outputTensors.end()) {
        if (m_outputTensors[index].isInnerMem) {
            // If it is inner buffer, releate it
            void* curBuffer = m_outputTensors[index].tensor->GetBuffer();
            m_device->ReleaseBuffer(curBuffer);
        }
    } else {
        // If output tensor does not exist, create a new null output tensor.
        ExeTensor exeTensor;
        m_outputTensors[index] = exeTensor;
        m_outputTensors[index].tensor = BuildNNTensorFromDesc(m_outputTensorDescs[index]);
        if (m_outputTensors[index].tensor == nullptr) {
            LOGE("SetOutputFromMemory failed, failed to build nntensor from desc.");
            return OH_NN_NULL_PTR;
        }
    }

    // Set the output tensor with memory
    m_outputTensors[index].tensor->SetBuffer(const_cast<const void*>(memory.data), memory.length);
    m_outputTensors[index].userBuffer = nullptr;
    m_outputTensors[index].userBufferLength = 0;
    m_outputTensors[index].isInnerMem = false;
    m_isRun = false;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::CreateInputMemory(uint32_t index, size_t length, OH_NN_Memory** memory)
{
    if (index >= m_inputTensorDescs.size()) {
        LOGE("CreateInputMemory failed, input index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_inputTensorDescs[index].first == nullptr) {
        LOGE("CreateInputMemory failed, tensor desc of input %{public}u is nullptr.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    // Allocate device buffer
    void* deviceInputBuffer = m_device->AllocateTensorBuffer(length, m_inputTensorDescs[index].first);
    if (deviceInputBuffer == nullptr) {
        LOGE("CreateInputMemory failed, allocating intput device buffer failed.");
        return OH_NN_MEMORY_ERROR;
    }

    *memory = new(std::nothrow) OH_NN_Memory{deviceInputBuffer, length};
    if (*memory == nullptr) {
        LOGE("CreateInputMemory failed, constructing OH_NN_Memory failed.");
        m_device->ReleaseBuffer(deviceInputBuffer);
        return OH_NN_MEMORY_ERROR;
    }

    // Save the buffer address for check when destroying it.
    m_inputCreatedMem[index].emplace_back(deviceInputBuffer);

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode NNExecutor::DestroyInputMemory(uint32_t index, OH_NN_Memory** memory)
{
    if (index >= m_inputTensorDescs.size()) {
        LOGE("DestroyInputMemory failed, input index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_inputCreatedMem.find(index) == m_inputCreatedMem.end()) {
        LOGE("DestroyInputMemory failed, the memory has not been created with the index.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<void*>& inputCreatedMem = m_inputCreatedMem[index];
    auto pos = std::find(inputCreatedMem.begin(), inputCreatedMem.end(), (*memory)->data);
    if (pos == inputCreatedMem.end()) {
        LOGE("DestroyInputMemory failed, the index does not match the memory.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto ret = m_device->ReleaseBuffer((*memory)->data);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Release input buffer failed.");
        return ret;
    }

    inputCreatedMem.erase(pos);
    delete *memory;
    *memory = nullptr;

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode NNExecutor::CreateOutputMemory(uint32_t index, size_t length, OH_NN_Memory** memory)
{
    if (index >= m_outputTensorDescs.size()) {
        LOGE("CreateOutputMemory failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_outputTensorDescs[index].first == nullptr) {
        LOGE("NNExecutor::CreateOutputMemory failed, tensor desc of output %{public}u is nullptr.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    // Allocate device buffer
    void* deviceOutputBuffer = m_device->AllocateTensorBuffer(length, m_outputTensorDescs[index].first);
    if (deviceOutputBuffer == nullptr) {
        LOGE("CreateOutputMemory failed, allocating output device buffer failed.");
        return OH_NN_MEMORY_ERROR;
    }

    *memory = new(std::nothrow) OH_NN_Memory{deviceOutputBuffer, length};
    if (*memory == nullptr) {
        LOGE("CreateOutputMemory failed, constructing OH_NN_Memory failed.");
        m_device->ReleaseBuffer(deviceOutputBuffer);
        return OH_NN_MEMORY_ERROR;
    }

    // Save the buffer address for check when destroying it.
    m_outputCreatedMem[index].emplace_back(deviceOutputBuffer);

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode NNExecutor::DestroyOutputMemory(uint32_t index, OH_NN_Memory** memory)
{
    if (index >= m_outputTensorDescs.size()) {
        LOGE("DestroyOutputMemory failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_outputCreatedMem.find(index) == m_outputCreatedMem.end()) {
        LOGE("DestroyOutputMemory failed, the memory has not been created with the index.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<void*>& outputCreatedMem = m_outputCreatedMem[index];
    auto pos = std::find(outputCreatedMem.begin(), outputCreatedMem.end(), (*memory)->data);
    if (pos == outputCreatedMem.end()) {
        LOGE("DestroyOutputMemory failed, the index does not match the memory.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto ret = m_device->ReleaseBuffer((*memory)->data);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Release output buffer failed.");
        return ret;
    }

    outputCreatedMem.erase(pos);
    delete *memory;
    *memory = nullptr;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::Run(const std::vector<std::shared_ptr<NNTensor>>& inputTensors,
    std::vector<std::shared_ptr<NNTensor>>& outputTensors)
{
    OH_NN_ReturnCode ret {OH_NN_FAILED};
    IOTensor tensor;
    std::vector<IOTensor> inputIOTensors;
    size_t inputSize = inputTensors.size();
    size_t outputSize = outputTensors.size();
    for (size_t i = 0; i < inputSize; ++i) {
        inputTensors[i]->ConvertToIOTensor(tensor);
        inputIOTensors.emplace_back(std::move(tensor));
    }

    std::vector<IOTensor> outputIOTensors;
    for (size_t i = 0; i < outputSize; ++i) {
        outputTensors[i]->ConvertToIOTensor(tensor);
        outputIOTensors.emplace_back(std::move(tensor));
    }

    std::vector<std::vector<int32_t>> outputsDims;
    std::vector<bool> isSufficientDataBuffer;
    ret = m_preparedModel->Run(inputIOTensors, outputIOTensors, outputsDims, isSufficientDataBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("PrepardModel Run() failed.");
        return ret;
    }

    // Set the output NNTensor's dimensions from output IOTensor if it is dynamic.
    // NNTensor::SetDimensions will check if the tensor buffer is enough for the new dimensions.
    if (outputsDims.size() != outputSize) {
        LOGE("ExecutionPlan run failed, size of outputsDims is not equal to outputTensors.");
        return OH_NN_INVALID_PARAMETER;
    }
    for (size_t i = 0; i < outputSize; ++i) {
        ret = outputTensors[i]->SetDimensions(outputsDims[i]);
        if (ret != OH_NN_SUCCESS) {
            LOGE("Run failed, error happened when setting output tensor's dimensions, output id: %zu.", i);
            return ret;
        }
        ret = m_outputTensorDescs[i].first->SetShape(outputsDims[i].data(), outputsDims[i].size());
        if (ret != OH_NN_SUCCESS) {
            LOGE("Run failed, error happened when setting inner output tensor's dimensions,"
                 " output id: %zu.", i);
            return ret;
        }
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::Run()
{
    NNRT_TRACE_NAME("Execution");
    if (m_inputTensorDescs.size() != m_inputTensors.size()) {
        LOGE("Run failed, some input tensors have not been set.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_outputTensorDescs.size() != m_outputTensors.size()) {
        LOGE("Run failed, some output tensors have not been set.");
        return OH_NN_INVALID_PARAMETER;
    }

    // Build the NNTensor pointer vector: inputTensors and outputTensors
    std::vector<std::shared_ptr<NNTensor>> inputTensors;
    std::vector<std::shared_ptr<NNTensor>> outputTensors;
    size_t inputSize = m_inputTensors.size();
    size_t outputSize = m_outputTensors.size();
    for (size_t i = 0; i < inputSize; ++i) {
        inputTensors.emplace_back(m_inputTensors[i].tensor);
    }
    for (size_t i = 0; i < outputSize; ++i) {
        outputTensors.emplace_back(m_outputTensors[i].tensor);
    }

    // Predict
    auto ret = Run(inputTensors, outputTensors);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Run failed, error happened when executing the inference.");
        return ret;
    }

    errno_t status{EOK};
    // Copy inner device buffer to user buffer if using SetOutput()
    for (size_t i = 0; i < outputSize; ++i) {
        if (m_outputTensors[i].isInnerMem) {
            auto size = outputTensors[i]->GetDataLength();
            if (size > m_outputTensors[i].userBufferLength) {
                LOGE("Output buffer size is not enough. Your size=%zu, but actual output size=%zu.",
                    m_outputTensors[i].userBufferLength, size);
                return OH_NN_INVALID_PARAMETER;
            }

            void* deviceBuffer = outputTensors[i]->GetBuffer();
            if (deviceBuffer == nullptr) {
                LOGE("Output buffer is nullptr.");
                return OH_NN_FAILED;
            }

            status = memcpy_s(m_outputTensors[i].userBuffer, m_outputTensors[i].userBufferLength, deviceBuffer, size);
            if (status != EOK) {
                LOGE("Run failed, memory copy from device buffer to user buffer failed. Error code: %d.", status);
                return OH_NN_MEMORY_ERROR;
            }
        }
    }

    m_isRun = true;
    return OH_NN_SUCCESS;
}

NNExecutor::~NNExecutor()
{
    for (auto& it : m_inputTensors) {
        if ((it.second).isInnerMem) {
            m_device->ReleaseBuffer((it.second).tensor->GetBuffer());
        }
        (it.second).tensor->SetBuffer(nullptr, 0);
        (it.second).tensor.reset();
        (it.second).userBuffer = nullptr;
    }
    m_inputTensors.clear();

    for (auto& it : m_outputTensors) {
        if ((it.second).isInnerMem) {
            m_device->ReleaseBuffer((it.second).tensor->GetBuffer());
        }
        (it.second).tensor->SetBuffer(nullptr, 0);
        (it.second).tensor.reset();
        (it.second).userBuffer = nullptr;
    }
    m_outputTensors.clear();

    for (auto& it : m_inputCreatedMem) {
        it.second.clear();
    }
    m_inputCreatedMem.clear();

    for (auto& it : m_outputCreatedMem) {
        it.second.clear();
    }
    m_outputCreatedMem.clear();

    if (m_executorConfig != nullptr) {
        delete m_executorConfig;
        m_executorConfig = nullptr;
    }

    UnSetDeinitModelCallBack();

    uint32_t modelId;
    GetModelID(modelId);
}

OH_NN_ReturnCode NNExecutor::DestroyPreparedModel()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_autoUnloadHandler != nullptr) {
        m_autoUnloadHandler->RemoveTask("nnexecutor_autounload" + std::to_string(m_executorid));
        m_autoUnloadHandler.reset();
    }

    if (m_preparedModel == nullptr) {
        LOGE("DestroyPreparedModel failed, m_preparedModel is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_preparedModel.reset();
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::SetDeinitModelCallBack()
{
    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGW("SetDeinitModelCallBack failed, fail to get nnrt service, skip SetDeinitModelCallBack.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.SetDeinitModelCallBack == nullptr) {
        LOGE("SetDeinitModelCallBack failed, nnrtService SetDeinitModelCallBack func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_preparedModel == nullptr) {
        LOGE("SetDeinitModelCallBack failed, m_preparedModel is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    int ret = nnrtService.SetDeinitModelCallBack(m_executorid, reinterpret_cast<Executor*>(this));
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("SetDeinitModelCallBack failed, some error happened when SetDeinitModelCallBack.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::UnSetDeinitModelCallBack()
{
    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGW("UnSetDeinitModelCallBack failed, fail to get nnrt service, skip UnSetDeinitModelCallBack.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.UnSetDeinitModelCallBack == nullptr) {
        LOGE("UnSetDeinitModelCallBack failed, nnrtService UnSetDeinitModelCallBack func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    int ret = nnrtService.UnSetDeinitModelCallBack(m_executorid);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("UnSetDeinitModelCallBack failed, some error happened when UnSetDeinitModelCallBack.");
        return static_cast<OH_NN_ReturnCode>(ret);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::GetNNRtModelIDFromCache(const std::string& path, const std::string& modelName,
    size_t& nnrtModelID)
{
    if (path.empty()) {
        LOGE("GetNNRtmodelIDFromCache failed, path is empty");
        return OH_NN_INVALID_PARAMETER;
    }

    if (modelName.empty()) {
        LOGE("GetNNRtmodelIDFromCache failed, modelName is empty");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!std::filesystem::is_directory(path)) {
        LOGW("GetNNRtmodelIDFromCache cvache path is not directory.");
        nnrtModelID = std::hash<std::string>{}(path);
        return OH_NN_SUCCESS;
    }

    std::string modelPath = path + "/" + modelName + "cache_info.nncache";
    char modelCachePath[PATH_MAX];
    if (realpath(modelPath.c_str(), modelCachePath) == nullptr) {
        LOGE("GetNNRtmodelIDFromCache fail to get real path of cacheDir.");
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

OH_NN_ReturnCode NNExecutor::ReinitScheduling(uint32_t hiaimodelID, bool* needModelLatency, const char* cachePath)
{
    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (!nnrtService.IsServiceAvaliable()) {
        LOGE("[HiaiExecutorImpl] ReinitScheduling failed, fail to get nnrt service, skip ReinitScheduling.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.AutoReinitSetModelID == nullptr) {
        LOGE("[HiaiExecutorImpl] ReinitScheduling failed, nnrtService AutoReinitSetModelId func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    size_t nnrtmodelID = 0;
    OH_NN_ReturnCode retCode = GetNNRtModelIDFromCache(m_cachePath, m_extensionConfig.modelName, nnrtmodelID);
    if ((retCode != OH_NN_SUCCESS) || (nnrtmodelID == 0)) {
        LOGE("[HiaiExecutorImpl] ReinitScheduling is failedm fail to GetNNRtModelIDFromCache.");
        return OH_NN_INVALID_PARAMETER;
    }

    int ret = nnrtService.AutoReinitSetModelID(m_originHiaiModelId, hiaimodelID, nnrtmodelID);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("[HiaiExecutorImpl] ReinitScheduling failed, some error happened when AutoReinitSetModelID.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (nnrtService.IsSupportScheduling == nullptr) {
        LOGE("[HiaiExecutorImpl] ReinitScheduling failed, nnrtService IsSupportScheduling func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    bool supportStat = false;
    ret = nnrtService.IsSupportScheduling(&supportStat);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("ReinitScheduling failed, some error happened when judge if support scheduling.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (!supportStat) {
        LOGW("device not support scheduling, jumper over scheduling.");
        return OH_NN_SUCCESS;
    }

    if (nnrtService.AutoReinitScheduling == nullptr) {
        LOGE("ReinitScheduling failed, nnrtService IsSupportScheduling func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
 
    ret = nnrtService.AutoReinitScheduling(m_originHiaiModelId, hiaimodelID, needModelLatency, cachePath);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("ReinitScheduling failed, some error happened when scheduling.");
        return OH_NN_INVALID_PARAMETER;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNExecutor::DeinitScheduling(uint32_t hiaimodelID)
{
    NNRtServiceApi& nnrtService = NNRtServiceApi::GetInstance();
    if (nnrtService.AutoUnload == nullptr) {
        LOGE("[HiaiExecutorImpl] AutoUnload failed, nnrtService AutoUnload func is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    int ret = nnrtService.AutoUnload(m_originHiaiModelId, hiaimodelID);
    if (ret != static_cast<int>(OH_NN_SUCCESS)) {
        LOGE("[HiaiExecutorImpl] AutoUnload failed, some error happen when AutoUnload hiaiModelId.");
        return OH_NN_INVALID_PARAMETER;
    }

    return OH_NN_SUCCESS;
}

bool NNExecutor::DeinitModel(std::string mode)
{
    if (m_preparedModel == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_preparedModel != nullptr &&
        OH_NNModel_HasCache(m_cachePath.c_str(),
                            m_extensionConfig.modelName.c_str(),
                            m_cacheVersion)) {
        uint32_t modelId;
        auto _ret = GetModelID(modelId);
        if (_ret != OH_NN_SUCCESS) {
            LOGW("GetModelID failed, some error happen when get model id for device.");
        }

        _ret = DeinitScheduling(modelId);
        if (_ret != OH_NN_SUCCESS) {
            LOGW("DeinitScheduling failed, some error happen when DeinitScheduling model.");
        }
        m_preparedModel.reset();
        if (mode == "FrozenDeinit") {
            if (m_autoUnloadHandler != nullptr) {
                m_autoUnloadHandler->RemoveTask("nnexecutor_autounload" + std::to_string(m_executorid));
                LOGI("FrozenDeinit pid=%{public}ld originHiaiModelId=%{public}u hiaiModelId=%{public}u",
                    static_cast<long>(getpid()), m_originHiaiModelId, modelId);
            }
        } else {
            LOGI("AutoUnload pid=%{public}ld originHiaiModelId=%{public}u hiaiModelId=%{public}u",
                static_cast<long>(getpid()), m_originHiaiModelId, modelId);
        }
    }

    return true;
}
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS