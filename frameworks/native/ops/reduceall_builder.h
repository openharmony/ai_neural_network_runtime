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

#ifndef NEURAL_NETWORK_RUNTIME_REDUCEALL_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_REDUCEALL_BUILDER_H

#include "mindir.h"

#include "frameworks/native/ops_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class ReduceAllBuilder : public OpsBuilder {
public:
    ReduceAllBuilder();
    ~ReduceAllBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;

    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetKeepDims(std::shared_ptr<NNTensor> tensor);

private:
    bool m_keepDims{false};
    mindspore::lite::ReduceMode m_mode{mindspore::lite::REDUCE_MODE_ALL};
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_REDUCEALL_BUILDER_H