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

#include "node_registry.h"

#include "utils/hdf_log.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
NodeRegistry& NodeRegistry::GetSingleton()
{
    static NodeRegistry registry;
    return registry;
}

NodeRegistry::Registrar::Registrar(NodeType type, std::function<PrimUniquePtr(const std::vector<int8_t>&)> nodeFunc)
{
    auto& registry = NodeRegistry::GetSingleton();
    if (registry.m_nodeRegs.find(type) != registry.m_nodeRegs.end()) {
        HDF_LOGW("Node has been registered. nodeType=%d", type);
    } else {
        registry.m_nodeRegs[type] = nodeFunc;
    }
}

std::function<PrimUniquePtr(const std::vector<int8_t>&)> NodeRegistry::GetNodeFunc(NodeType type) const
{
    if (m_nodeRegs.find(type) == m_nodeRegs.end()) {
        HDF_LOGW("Node type is not found. nodeType=%d", type);
        return nullptr;
    }

    return m_nodeRegs.at(type);
}

bool NodeRegistry::IsNodeTypeExist(NodeType type) const
{
    if (m_nodeRegs.find(type) == m_nodeRegs.end()) {
        return false;
    }
    return true;
}
} // namespace V1_0
} // namespace Nnrt
} // namespace HDI
} // namespace OHOS