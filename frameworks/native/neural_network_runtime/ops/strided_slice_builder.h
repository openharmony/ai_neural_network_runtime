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

#include "ops_builder.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class StridedSliceBuilder : public OpsBuilder {
public:
    typedef OH_NN_ReturnCode (StridedSliceBuilder::*FuncPtr)(const std::shared_ptr<NNTensor>&);

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
    OH_NN_ReturnCode SetBeginMask(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetEndMask(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetEllipsisMask(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetNewAxisMask(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetShrinkAxisMask(const std::shared_ptr<NNTensor>& tensor);

private:
    int64_t m_begin_mask = {0};
    int64_t m_end_mask = {0};
    int64_t m_ellipsis_mask = {0};
    int64_t m_new_axis_mask = {0};
    int64_t m_shrink_axis_mask = {0};
    std::unordered_map<OH_NN_TensorType, FuncPtr> m_paramMap = {
        {OH_NN_STRIDED_SLICE_BEGIN_MASK, &StridedSliceBuilder::SetBeginMask},
        {OH_NN_STRIDED_SLICE_END_MASK, &StridedSliceBuilder::SetEndMask},
        {OH_NN_STRIDED_SLICE_ELLIPSIS_MASK, &StridedSliceBuilder::SetEllipsisMask},
        {OH_NN_STRIDED_SLICE_NEW_AXIS_MASK, &StridedSliceBuilder::SetNewAxisMask},
        {OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK, &StridedSliceBuilder::SetShrinkAxisMask}
    };
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_STRIDEDSLICE_BUILDER_H
