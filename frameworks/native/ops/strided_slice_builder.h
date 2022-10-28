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

#ifndef NEURAL_NETWORK_RUNTIME_STRIDEDSLICE_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_STRIDEDSLICE_BUILDER_H

#include "frameworks/native/ops_builder.h"
#include "frameworks/native/ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class StridedSliceBuilder : public OpsBuilder {
public:
    StridedSliceBuilder();
    ~StridedSliceBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;

    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetInputOutput(const std::vector<uint32_t>& inputsIndex,
                                    const std::vector<uint32_t>& outputsIndex,
                                    const std::vector<std::shared_ptr<NNTensor>>& allTensors);
    OH_NN_ReturnCode SetBeginMask(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetEndMask(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetEllipsisMask(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetNewAxisMask(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetShrinkAxisMask(std::shared_ptr<NNTensor> tensor);

private:
    int64_t m_begin_mask = {0};
    int64_t m_end_mask = {0};
    int64_t m_ellipsis_mask = {0};
    int64_t m_new_axis_mask = {0};
    int64_t m_shrink_axis_mask = {0};
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_STRIDEDSLICE_BUILDER_H
