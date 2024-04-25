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

#ifndef NEURAL_NETWORK_RUNTIME_SPACE_TO_DEPTH_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_SPACE_TO_DEPTH_BUILDER_H

#include "ops_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class SpaceToDepthBuilder : public OpsBuilder {
public:
    typedef OH_NN_ReturnCode (SpaceToDepthBuilder::*FuncPtr)(const std::shared_ptr<NNTensor>&);

    SpaceToDepthBuilder();
    ~SpaceToDepthBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;

    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetBlockSize(const std::shared_ptr<NNTensor>& tensor);

private:
    int64_t m_blockSize {0};
    std::unordered_map<OH_NN_TensorType, FuncPtr> m_paramMap = {
        {OH_NN_SPACE_TO_DEPTH_BLOCK_SIZE, &SpaceToDepthBuilder::SetBlockSize}
    };
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_SPACE_TO_DEPTH_BUILDER_H
