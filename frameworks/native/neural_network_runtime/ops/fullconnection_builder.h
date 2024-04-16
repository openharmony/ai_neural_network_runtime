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

#ifndef NEURAL_NETWORK_RUNTIME_FULLCONNECTION_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_FULLCONNECTION_BUILDER_H

#include "mindir.h"

#include "ops_builder.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class FullConnectionBuilder : public OpsBuilder {
public:
    typedef OH_NN_ReturnCode (FullConnectionBuilder::*FuncPtr)(const std::shared_ptr<NNTensor>&);

    FullConnectionBuilder();
    ~FullConnectionBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;

    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetFullConnectionInput(const std::vector<uint32_t>& inputsIndex,
                                            const std::vector<uint32_t>& outputsIndex,
                                            const std::vector<std::shared_ptr<NNTensor>>& allTensors);
    OH_NN_ReturnCode SetHasBias(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetUseAxis(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetFullConnectionActivation(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetAxis(const std::shared_ptr<NNTensor>& tensor);

private:
    bool m_hasBias {false};
    bool m_useAxis {false};
    int64_t m_axis {0};
    mindspore::lite::ActivationType m_activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    std::unordered_map<OH_NN_TensorType, FuncPtr> m_paramMap = {
        {OH_NN_FULL_CONNECTION_ACTIVATIONTYPE, &FullConnectionBuilder::SetFullConnectionActivation},
        {OH_NN_FULL_CONNECTION_HAS_BIAS, &FullConnectionBuilder::SetHasBias},
        {OH_NN_FULL_CONNECTION_USE_AXIS, &FullConnectionBuilder::SetUseAxis},
        {OH_NN_FULL_CONNECTION_AXIS, &FullConnectionBuilder::SetAxis}
    };

    bool m_axisIsSet {false};
    bool m_useAxisIsSet {false};
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_FULLCONNECTION_BUILDER_H