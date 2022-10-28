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

#ifndef OHOS_HDI_NNR_NODE_REGISTRY_H
#define OHOS_HDI_NNR_NODE_REGISTRY_H

#include <memory>
#include <functional>
#include <unordered_map>

#include "v1_0/nnrt_types.h"
#include "mindspore_schema/model_generated.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
using PrimUniquePtr = std::unique_ptr<mindspore::schema::PrimitiveT>;
class NodeRegistry {
public:
    struct Registrar {
        Registrar() = delete;
        Registrar(NodeType type, std::function<PrimUniquePtr(const std::vector<int8_t>&)> nodeFunc);
    };

public:
    static NodeRegistry& GetSingleton();
    std::function<PrimUniquePtr(const std::vector<int8_t>&)> GetNodeFunc(NodeType type) const;
    bool IsNodeTypeExist(NodeType type) const;

private:
    NodeRegistry() {};
    NodeRegistry(const NodeRegistry&) = delete;
    NodeRegistry& operator=(const NodeRegistry&) = delete;

private:
    std::unordered_map<NodeType, std::function<PrimUniquePtr(const std::vector<int8_t>&)>> m_nodeRegs;
};

#define REGISTER_NODE(nodeName, nodeType, funcPtr) static NodeRegistry::Registrar g_##nodeName(nodeType, funcPtr)
} // namespace V1_0
} // namespace Nnrt
} // namespace HDI
} // namespace OHOS
#endif // OHOS_HDI_NNR_NODE_REGISTRY_H