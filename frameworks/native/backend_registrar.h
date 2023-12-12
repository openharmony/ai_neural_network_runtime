#include <vector>
#include <memory>
#include <functional>

#include "backend.h"

#ifndef NEURAL_NETWORK_CORE_BACKEND_REGISTRAR_H
#define NEURAL_NETWORK_CORE_BACKEND_REGISTRAR_H

namespace OHOS {
namespace NeuralNetworkRuntime {
using CreateBackend = std::function<std::shared_ptr<Backend>()>;

class BackendRegistrar {
public:
    explicit BackendRegistrar(const CreateBackend creator);
    ~BackendRegistrar() = default;
};

#define REGISTER_BACKEND(backend, creator)                                                            \
    namespace {                                                                                       \
    static OHOS::NeuralNetworkRuntime::BackendRegistrar g_##backendName##_backend_registrar(creator); \
    } // namespace
} // NeuralNetworkRuntime
} // OHOS

#endif // NEURAL_NETWORK_CORE_BACKEND_REGISTRAR_H