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

#ifndef NEURAL_NETWORK_RUNTIME_LAYERNORM_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_LAYERNORM_BUILDER_H

#include "frameworks/native/ops_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class LayerNormBuilder : public OpsBuilder {
public:
    LayerNormBuilder();
    ~LayerNormBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;
    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetBeginNormAxis(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetEpsilon(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetBeginParamsAxis(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode ValidateGammaAndBetaShape(const std::vector<uint32_t>& inputsIndex,
        int beginAxis, const std::vector<std::shared_ptr<NNTensor>>& allTensors) const;

private:
    int m_beginNormAxis{1};
    float m_epsilon{1e-7};
    bool m_elementwiseAffine{false};
    int m_beginParamsAxis{1};
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_LAYERNORM_BUILDER_H