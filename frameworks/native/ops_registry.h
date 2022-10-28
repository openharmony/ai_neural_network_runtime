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

#ifndef HETERNEURAL_NETWORK_OPS_REGISTRY_H
#define HETERNEURAL_NETWORK_OPS_REGISTRY_H

#include <functional>
#include <memory>
#include <unordered_map>

#include "ops_builder.h"
#include "interfaces/kits/c/neural_network_runtime.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class OpsRegistry {
public:
    struct Registrar {
        Registrar() = delete;
        Registrar(OH_NN_OperationType opsType, std::function<std::unique_ptr<OpsBuilder>()> createFunc);
    };

public:
    static OpsRegistry& GetSingleton();
    std::unique_ptr<OpsBuilder> GetOpsBuilder(OH_NN_OperationType type) const;

private:
    OpsRegistry() {};
    OpsRegistry(const OpsRegistry&) = delete;
    OpsRegistry& operator=(const OpsRegistry&) = delete;

private:
    std::unordered_map<OH_NN_OperationType, std::function<std::unique_ptr<OpsBuilder>()>> m_opsRegedit;
};

#define CREATE_FUNC(T) ([]()->std::unique_ptr<OpsBuilder> {return std::make_unique<T>();})
#define REGISTER_OPS(T, opsType) static OpsRegistry::Registrar g_##T(opsType, CREATE_FUNC(T))
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespcae OHOS
#endif // HETERNEURAL_NETWORK_OPS_REGISTRY_H