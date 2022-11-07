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
#ifndef NEURAL_NETWORK_RUNTIME_COMPILATION_H
#define NEURAL_NETWORK_RUNTIME_COMPILATION_H

#include "inner_model.h"
#include "execution_plan.h"

#include "device.h"
#include "cpp_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
struct ModelCacheInfo {
    uint64_t fileNumber = 0;
    uint64_t version = 0;
    uint64_t deviceId = 0;
    std::vector<unsigned short> modelCheckSum;
};

class Compilation {
public:
    explicit Compilation(const InnerModel* innerModel);

    OH_NN_ReturnCode SetDevice(size_t deviceId);
    OH_NN_ReturnCode SetCacheDir(const std::string& cacheModelPath, uint32_t version);
    OH_NN_ReturnCode SetPerformance(OH_NN_PerformanceMode performance);
    OH_NN_ReturnCode SetPriority(OH_NN_Priority priority);
    OH_NN_ReturnCode SetEnableFp16(bool isFp16);

    OH_NN_ReturnCode Build();

    bool IsBuild() const;
    bool IsDynamicShape() const;
    std::vector<std::shared_ptr<NNTensor>>GetInputTensors() const;
    std::vector<std::shared_ptr<NNTensor>>GetOutputTensors() const;
    std::shared_ptr<ExecutionPlan> GetExecutionPlan() const;

private:
    std::shared_ptr<mindspore::lite::LiteGraph> m_liteGraph {nullptr};
    OH_NN_Priority m_priority {OH_NN_PRIORITY_NONE};
    OH_NN_PerformanceMode m_performance {OH_NN_PERFORMANCE_NONE};
    bool m_enableFp16 {false};
    std::shared_ptr<Device> m_device {nullptr};
    std::string m_cachePath;
    uint32_t m_version {0};
    size_t m_deviceId {0};
    bool m_isBuild {false};
    std::shared_ptr<ExecutionPlan> m_executionPlan {nullptr};
    std::vector<std::shared_ptr<NNTensor>> m_inputTensors;
    std::vector<std::shared_ptr<NNTensor>> m_outputTensors;

private:
    OH_NN_ReturnCode GenerateCacheFiles(const std::vector<ModelBuffer>& modelBuffer) const;
    OH_NN_ReturnCode GenerateCacheModel(size_t cacheNumber, std::unique_ptr<uint64_t[]>& cacheInfo,
        std::vector<ModelBuffer> modelBuffer) const;
    OH_NN_ReturnCode GenerateCacheInfo(uint32_t cacheSize, std::unique_ptr<uint64_t[]>& cacheInfo) const;
    OH_NN_ReturnCode CheckCacheInfo(ModelCacheInfo& modelCacheInfo, const std::string& cacheInfoPath) const;
    OH_NN_ReturnCode ReadCacheModelFile(const std::string& file, ModelBuffer& modelBuffer) const;
    OH_NN_ReturnCode RemoveCacheFiles(uint32_t fileNumber) const;
    unsigned short GetCrc16(const unsigned char* buffer, size_t length) const;
    OH_NN_ReturnCode CheckCacheModel(const ModelCacheInfo& modelCacheInfo,
        std::vector<ModelBuffer>& modelBuffers) const;
    OH_NN_ReturnCode NormalBuild(std::shared_ptr<PreparedModel>& preparedModel);
    OH_NN_ReturnCode GenCacheBuild(std::shared_ptr<PreparedModel>& preparedModel);
    OH_NN_ReturnCode ReGenCacheBuild(uint32_t fileNumber, std::shared_ptr<PreparedModel>& preparedModel);
    OH_NN_ReturnCode LoadCacheBuild(std::shared_ptr<PreparedModel>& preparedModel, const ModelCacheInfo& cacheInfo);
    OH_NN_ReturnCode InnerBuild();
    OH_NN_ReturnCode GetCacheFileLength(std::ifstream& ifs, int& fsize) const;
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_COMPILATION_H