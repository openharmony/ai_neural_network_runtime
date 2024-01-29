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

#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
OpsRegistry::Registrar::Registrar(OH_NN_OperationType opsType, std::function<std::unique_ptr<OpsBuilder>()> createFunc)
{
    OpsRegistry& registry = OpsRegistry::GetSingleton();
    if (registry.m_opsRegedit.find(opsType) != registry.m_opsRegedit.end()) {
        LOGW("Operantion has been registered, cannot register twice. Operation type: %d", opsType);
    } else {
        registry.m_opsRegedit[opsType] = createFunc;
    }
}

OpsRegistry& OpsRegistry::GetSingleton()
{
    static OpsRegistry opsRegistry;
    return opsRegistry;
}

std::unique_ptr<OpsBuilder> OpsRegistry::GetOpsBuilder(OH_NN_OperationType type) const
{
    if (m_opsRegedit.find(type) != m_opsRegedit.end()) {
        return m_opsRegedit.at(type)();
    }
    return nullptr;
}
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS