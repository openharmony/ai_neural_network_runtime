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

#ifndef NEURAL_NETWORK_RUNTIME_EXECUTION_PLAN_H
#define NEURAL_NETWORK_RUNTIME_EXECUTION_PLAN_H

#include "frameworks/native/nn_tensor.h"
#include "interfaces/kits/c/neural_network_runtime_type.h"
#include "prepared_model.h"
#include "device.h"


namespace OHOS {
namespace NeuralNetworkRuntime {
class ExecutionPlan {
public:
    ExecutionPlan(std::shared_ptr<PreparedModel> preparedModel, std::shared_ptr<Device> device)
        : m_preparedModel(preparedModel),
          m_device(device) {};

    OH_NN_ReturnCode Run(const std::vector<std::shared_ptr<NNTensor>>& inputTensors,
                         std::vector<std::shared_ptr<NNTensor>>& outputTensors);

    std::shared_ptr<Device> GetInputDevice() const;
    std::shared_ptr<Device> GetOutputDevice() const;

private:
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::shared_ptr<Device> m_device {nullptr};
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif