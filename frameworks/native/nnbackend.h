#include "backend.h"
#include "executor.h"
#include "tensor.h"
#include "tensor_desc.h"
#include "device.h"
#include "nncompiler.h"

#ifndef NEURAL_NETWORK_RUNTIME_NNBACKEND_H
#define NEURAL_NETWORK_RUNTIME_NNBACKEND_H

namespace OHOS {
namespace NeuralNetworkRuntime {
class NNBackend : public Backend {
public:
    explicit NNBackend(const std::shared_ptr<Device>& device, size_t backendID);
    ~NNBackend();

    // Backend Info
    size_t GetBackendID() const override;
    OH_NN_ReturnCode GetBackendName(std::string& backendName) const override;
    OH_NN_ReturnCode GetBackendType(OH_NN_DeviceType& backendType) const override;
    OH_NN_ReturnCode GetBackendStatus(DeviceStatus& status) const override;

    // Create & Destory compiler
    Compiler* CreateCompiler(Compilation* compilation) override;
    OH_NN_ReturnCode DestroyCompiler(Compiler* compiler) override;

    // Create & Destory Executor
    Executor* CreateExecutor(Compilation* compilation) override;
    OH_NN_ReturnCode DestroyExecutor(Executor* executor) override;

    // Create & Destory Tensor
    Tensor* CreateTensor(TensorDesc* desc) override;
    OH_NN_ReturnCode DestroyTensor(Tensor* tensor) override;

    // external methods
    std::shared_ptr<Device> GetDevice() const;
    OH_NN_ReturnCode GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                           std::vector<bool>& ops);

private:
    std::shared_ptr<Device> m_device;
    size_t m_backendID;
};
} // NeuralNetworkRuntime
} // OHOS

#endif // NEURAL_NETWORK_RUNTIME_NNBACKEND_H