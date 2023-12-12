#include "backend_registrar.h"

#include "common/log.h"
#include "backend_manager.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
BackendRegistrar::BackendRegistrar(const CreateBackend creator)
{
    auto& backendManager = BackendManager::GetInstance();
    OH_NN_ReturnCode ret = backendManager.RegisterBackend(creator);
    if (ret != OH_NN_SUCCESS) {
        LOGW("[BackendRegistrar] Register backend failed. ErrorCode=%{public}d", ret);
    }
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS