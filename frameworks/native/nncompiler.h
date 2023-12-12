#include "compiler.h"

#include "mindir.h"
#include "device.h"
#include "inner_model.h"
#include "prepared_model.h"
#include "nnexecutor.h"

#ifndef NEURAL_NETWORK_RUNTIME_NNCOMPILER_H
#define NEURAL_NETWORK_RUNTIME_NNCOMPILER_H

namespace OHOS {
namespace NeuralNetworkRuntime {

class NNCompiler : public Compiler {
public:
    NNCompiler() = delete;
    NNCompiler(std::shared_ptr<Device> device, size_t backendID);
    NNCompiler(const void* model, std::shared_ptr<Device> device, size_t backendID);
    ~NNCompiler();

    size_t GetBackendID() const override;

    OH_NN_ReturnCode SetCacheDir(const std::string& cacheModelPath, uint32_t version) override;
    OH_NN_ReturnCode SetPerformance(OH_NN_PerformanceMode performance) override;
    OH_NN_ReturnCode SetPriority(OH_NN_Priority priority) override;
    OH_NN_ReturnCode SetEnableFp16(bool isFp16) override;

    bool IsBuild() const override;
    OH_NN_ReturnCode Build() override;

    OH_NN_ReturnCode SaveToCacheFile() const override;
    OH_NN_ReturnCode RestoreFromCacheFile() override;
    OH_NN_ReturnCode SaveToCacheBuffer(const void* buffer, size_t length, size_t* modelSize) const override;
    OH_NN_ReturnCode RestoreFromCacheBuffer(const void* buffer, size_t length) override;

    OH_NN_ReturnCode SetExtensionConfig(const std::unordered_map<std::string, std::vector<char>>& configs) override;
    OH_NN_ReturnCode SetOptions(const std::vector<std::shared_ptr<void>>& options) override;

    NNExecutor* CreateExecutor();
    OH_NN_ReturnCode IsSupporttedModel(const std::shared_ptr<mindspore::lite::LiteGraph>& liteGraph,
                                       bool& isSupporttedModel) const;

private:
    void ReleaseBuffer(std::vector<Buffer>& buffers) const;
    void ReleaseBufferByDevice(std::vector<Buffer>& buffers) const;
    OH_NN_ReturnCode SerializeTensorsToBuffer(
        const std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& tensorDescs,
        Buffer& buffer) const;
    OH_NN_ReturnCode DeserializedTensorsFromBuffer(
        const Buffer& buffer, std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>& tensorDescs);

private:
    bool m_isBuild {false};
    bool m_enableFp16 {false};
    std::string m_cachePath;
    uint32_t m_cacheVersion {0};
    std::shared_ptr<Device> m_device {nullptr};
    size_t m_backendID {0};
    OH_NN_Priority m_priority {OH_NN_PRIORITY_NONE};
    OH_NN_PerformanceMode m_performance {OH_NN_PERFORMANCE_NONE};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    Buffer m_quantBuffer {nullptr, 0};
    std::string m_modelName;
    void* m_metaGraph {nullptr};
    std::shared_ptr<mindspore::lite::LiteGraph> m_liteGraph {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
};
} // NeuralNetworkRuntime
} // OHOS

#endif // NEURAL_NETWORK_RUNTIME_NNCOMPILER_H