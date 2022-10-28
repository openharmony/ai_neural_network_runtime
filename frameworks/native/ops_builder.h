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

#ifndef NEURAL_NETWORK_RUNTIME_OPS_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_OPS_BUILDER_H

#include <memory>
#include <unordered_map>

#include "nn_tensor.h"
#include "common/log.h"
#include "interfaces/kits/c/neural_network_runtime.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
using LiteGraphPrimitvePtr = std::unique_ptr<void, void(*)(void*)>;
void DestroyLiteGraphPrimitive(void* primitive);

// QuantType Enum
enum class OpsQuantType: int {
    QUANT_NONE = 0,
    QUANT_ALL = 1
};

class OpsBuilder {
public:
    OpsBuilder() = default;
    virtual ~OpsBuilder() = default;

    // Other operation builders inherit from OpsBuilder, delete these special construction and assignment functions.
    OpsBuilder(const OpsBuilder& opsBuilder) = delete;
    OpsBuilder& operator=(const OpsBuilder& opsBuilder) = delete;
    OpsBuilder(OpsBuilder&& opsBuilder) = delete;
    OpsBuilder& operator=(OpsBuilder&& opsBuilder) = delete;

    virtual OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                                   const std::vector<uint32_t>& inputsIndex,
                                   const std::vector<uint32_t>& outputsIndex,
                                   const std::vector<std::shared_ptr<NNTensor>>& allTensors) = 0;
    virtual LiteGraphPrimitvePtr GetPrimitive() = 0;

    virtual void GetInputIndex(std::vector<uint32_t>& inputsIndex,
                               const std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID) const;
    virtual void GetOutputIndex(std::vector<uint32_t>& outputsIndex,
                                const std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID) const;
    virtual std::string GetName() const;
    virtual OpsQuantType GetQuantType() const;

protected:
    OH_NN_ReturnCode CheckIOIndex(const std::vector<uint32_t>& inputsIndex,
                                  const std::vector<uint32_t>& outputsIndex,
                                  const std::vector<std::shared_ptr<NNTensor>>& allTensors,
                                  const size_t inputNum,
                                  const size_t outputNum) const;
    void SetQuantType(const std::vector<uint32_t>& outputsIndex,
                      const std::vector<std::shared_ptr<NNTensor>>& allTensors);

protected:
    std::string m_name;
    std::vector<uint32_t> m_inputsIndex;
    std::vector<uint32_t> m_outputsIndex;
    OpsQuantType m_quantType {OpsQuantType::QUANT_NONE};
    bool m_isBuild {false};
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_OPS_BUILDER_H