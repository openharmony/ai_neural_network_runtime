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

#ifndef NEURAL_NETWORK_RUNTIME_PREPARED_MODEL_H
#define NEURAL_NETWORK_RUNTIME_PREPARED_MODEL_H

#include <vector>

#include "interfaces/kits/c/neural_network_runtime_type.h"
#include "cpp_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class PreparedModel {
public:
    PreparedModel() = default;
    virtual ~PreparedModel() = default;

    virtual OH_NN_ReturnCode ExportModelCache(std::vector<ModelBuffer>& modelCache) = 0;

    virtual OH_NN_ReturnCode Run(const std::vector<IOTensor>& inputs,
                                 const std::vector<IOTensor>& outputs,
                                 std::vector<std::vector<int32_t>>& outputsDims,
                                 std::vector<bool>& isOutputBufferEnough) = 0;
};
} // OHOS
} // namespace NeuralNetworkRuntime
#endif // NEURAL_NETWORK_RUNTIME_PREPARED_MODEL_H