/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#ifndef NEURAL_NETWORK_RUNTIME_LRN_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_LRN_BUILDER_H

#include "mindir.h"

#include "ops_builder.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class LRNBuilder : public OpsBuilder {
public:
    LRNBuilder();
    ~LRNBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;

    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetDepthRadius(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetBias(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetAlpha(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetBeta(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetNormRegion(std::shared_ptr<NNTensor> tensor);

private:
    int64_t m_depthRadius {0};
    float m_bias {0.0f};
    float m_alpha {0.0f};
    float m_beta {0.0f};
    std::string m_normRegion {"ACROSS_CHANNELS"};
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_LRN_BUILDER_H
