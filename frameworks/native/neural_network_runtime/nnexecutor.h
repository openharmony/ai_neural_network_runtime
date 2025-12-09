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

#ifndef NEURAL_NETWORK_RUNTIME_NNEXECUTOR_H
#define NEURAL_NETWORK_RUNTIME_NNEXECUTOR_H

#include <mutex>
#include "executor.h"
#include "device.h"
#include "prepared_model.h"
#include "nn_tensor.h"
#include "log.h"

#include "event_handler.h"
#include "event_runner.h"

#include <chrono>
namespace OHOS {
namespace NeuralNetworkRuntime {
class NNExecutor : public Executor {
public:
    NNExecutor(size_t backendID,
               std::shared_ptr<Device> device,
               std::shared_ptr<PreparedModel> preparedModel,
               const std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& inputTensorDescs,
               const std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& outputTensorDescs,
               std::string cachePath, uint32_t cacheVersion, ExtensionConfig extensionConfig, bool enableFp16,
               OH_NN_PerformanceMode performance, OH_NN_Priority priority);
    ~NNExecutor() override;

    OH_NN_ReturnCode GetInputDimRange(size_t inputIndex,
                                      size_t** minInputDims,
                                      size_t** maxInputDims,
                                      size_t* shapeNum) const override;
    OH_NN_ReturnCode GetOutputShape(uint32_t outputIndex, int32_t** shape, uint32_t* shapeNum) const override;

    size_t GetInputNum() const override;
    size_t GetOutputNum() const override;
    NN_TensorDesc* CreateInputTensorDesc(size_t index) const override;
    NN_TensorDesc* CreateOutputTensorDesc(size_t index) const override;

    OH_NN_ReturnCode SetOnRunDone(NN_OnRunDone onRunDone) override;
    OH_NN_ReturnCode SetOnServiceDied(NN_OnServiceDied onServiceDied) override;
    OH_NN_ReturnCode RunSync(NN_Tensor* inputTensors[],
                             size_t inputSize,
                             NN_Tensor* outputTensors[],
                             size_t outputSize) override;
    OH_NN_ReturnCode RunSyncWithAipp(NN_Tensor* inputTensors[],
                             size_t inputSize,
                             NN_Tensor* outputTensors[],
                             size_t outputSize,
                             const char* aippStrings) override;
    OH_NN_ReturnCode RunAsync(NN_Tensor* inputTensors[],
                              size_t inputSize,
                              NN_Tensor* outputTensors[],
                              size_t outputSize,
                              int32_t timeout,
                              void* userData) override;
    OH_NN_ReturnCode GetModelID(uint32_t& modelId) const override;
    size_t GetBackendID() override;
    OH_NN_ReturnCode SetExtensionConfig(const std::unordered_map<std::string, std::vector<char>>& configs) override;
    ExecutorConfig* GetExecutorConfig() const override;

    // The following APIs are compatible with older versions
    OH_NN_ReturnCode SetInput(uint32_t index, const OH_NN_Tensor& nnTensor, const void* buffer, size_t length);
    OH_NN_ReturnCode SetInputFromMemory(uint32_t index, const OH_NN_Tensor& nnTensor, const OH_NN_Memory& memory);
    OH_NN_ReturnCode SetOutput(uint32_t index, void* buffer, size_t length);
    OH_NN_ReturnCode SetOutputFromMemory(uint32_t index, const OH_NN_Memory& memory);

    OH_NN_ReturnCode CreateInputMemory(uint32_t index, size_t length, OH_NN_Memory** memory);
    OH_NN_ReturnCode CreateOutputMemory(uint32_t index, size_t length, OH_NN_Memory** memory);
    OH_NN_ReturnCode DestroyInputMemory(uint32_t index, OH_NN_Memory** memory);
    OH_NN_ReturnCode DestroyOutputMemory(uint32_t index, OH_NN_Memory** memory);

    OH_NN_ReturnCode Run();

    bool DeinitModel(std::string mode) override;
    OH_NN_ReturnCode SetDeinitModelCallBack() override;
    OH_NN_ReturnCode UnSetDeinitModelCallBack() override;
    OH_NN_ReturnCode DestroyPreparedModel() override;

private:
    OH_NN_ReturnCode GetInputDimVec() const;
    OH_NN_ReturnCode CheckInputDimRanges(NN_Tensor* inputTensors[], size_t inputSize);

    // The following APIs are compatible with older versions
    OH_NN_ReturnCode Run(const std::vector<std::shared_ptr<NNTensor>>& inputTensors,
                         std::vector<std::shared_ptr<NNTensor>>& outputTensors);
    bool CompareAttribute(
        const std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>& tensorDesc, const NNTensor& tensor) const;
    std::shared_ptr<NNTensor> BuildNNTensorFromDesc(
        const std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>& tensorDesc);
    OH_NN_ReturnCode BuildInputTensor(uint32_t index, const OH_NN_Tensor& nnTensor,
                                      std::shared_ptr<NNTensor> inputTensor) const;
    OH_NN_ReturnCode SetInputTensorWithCurrentBuffer(uint32_t index, std::shared_ptr<NNTensor> inputTensor,
                                                     const void* buffer, size_t dataLength, size_t curBufferLength);
    void SetInputTensorWithNewBuffer(uint32_t index, std::shared_ptr<NNTensor> inputTensor,
                                     const void* inputBuffer, size_t length, bool isInnerMem);
    OH_NN_ReturnCode CheckInputDimRanges(uint32_t index, const OH_NN_Tensor& nnTensor) const;
    OH_NN_ReturnCode DeserializedTensorsFromBuffer(
        const Buffer& buffer, std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& tensorDescs);
    OH_NN_ReturnCode Reload();
    OH_NN_ReturnCode ReinitScheduling(uint32_t hiaimodelID, bool* needModelLatency, const char* cachePath);
    OH_NN_ReturnCode DeinitScheduling(uint32_t hiaimodelID);
    OH_NN_ReturnCode GetNNRtModelIDFromCache(const std::string& path, const std::string& modelName,
        size_t& nnrtModelID);

private:
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::string m_cachePath;
    uint32_t m_cacheVersion {0};
    ExtensionConfig m_extensionConfig;
    bool m_enableFp16 {false};
    OH_NN_PerformanceMode m_performance {OH_NN_PERFORMANCE_NONE};
    OH_NN_Priority m_priority {OH_NN_PRIORITY_NONE};
    uint32_t m_originHiaiModelId;

    // The following parameters are provided for compatibility with older versions
    struct ExeTensor {
        std::shared_ptr<NNTensor> tensor {nullptr};
        void* userBuffer {nullptr};
        size_t userBufferLength {0};
        bool isInnerMem {false};
    };
    bool m_isRun {false};
    ExecutorConfig* m_executorConfig {nullptr};
    std::unordered_map<int, ExeTensor> m_inputTensors;
    std::unordered_map<int, ExeTensor> m_outputTensors;
    std::unordered_map<int, std::vector<void*>> m_inputCreatedMem;
    std::unordered_map<int, std::vector<void*>> m_outputCreatedMem;
    mutable std::vector<std::vector<size_t>> m_minInputDimsVec;
    mutable std::vector<std::vector<size_t>> m_maxInputDimsVec;

    std::shared_ptr<OHOS::AppExecFwk::EventRunner> m_autoUnloadRunner;
    std::shared_ptr<OHOS::AppExecFwk::EventHandler> m_autoUnloadHandler;
    uint64_t m_executorid;
    std::mutex m_mutex;
    std::string m_aippPara;
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif  // NEURAL_NETWORK_RUNTIME_NNEXECUTOR_H
