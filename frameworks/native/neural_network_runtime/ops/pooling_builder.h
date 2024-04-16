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

#ifndef NEURAL_NETWORK_RUNTIME_POOLING_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_POOLING_BUILDER_H

#include "mindir.h"

#include "ops_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class PoolingBuilder : public OpsBuilder {
public:
    typedef OH_NN_ReturnCode (PoolingBuilder::*FuncPtr)(const std::shared_ptr<NNTensor>&);

    PoolingBuilder() = default;
    virtual ~PoolingBuilder() = default;

    OH_NN_ReturnCode PoolingBuild(const std::vector<uint32_t>& paramsIndex,
                                  const std::vector<uint32_t>& inputsIndex,
                                  const std::vector<uint32_t>& outputsIndex,
                                  const std::vector<std::shared_ptr<NNTensor>>& allTensors);

    OH_NN_ReturnCode SetInputAndOutput(const std::vector<uint32_t>& inputsIndex,
                                       const std::vector<uint32_t>& outputsIndex,
                                       const std::vector<std::shared_ptr<NNTensor>>& allTensors);

    OH_NN_ReturnCode SetKernel(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetStrides(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetPadModeOrPaddings(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetRoundMode(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetActivation(const std::shared_ptr<NNTensor>& tensor);
    OH_NN_ReturnCode SetGlobal(const std::shared_ptr<NNTensor>& tensor);

protected:
    std::vector<int64_t> m_kernelSize;
    std::vector<int64_t> m_pad;
    std::vector<int64_t> m_strides;
    mindspore::lite::PadMode m_padMode {mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType m_activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    mindspore::lite::RoundMode m_roundMode {mindspore::lite::ROUND_MODE_FLOOR};
    mindspore::lite::Format m_format {mindspore::lite::FORMAT_NCHW};
    bool m_global {false};
    std::unordered_map<OH_NN_TensorType, FuncPtr> m_paramMap = {
        {OH_NN_MAX_POOL_KERNEL_SIZE, &PoolingBuilder::SetKernel},
        {OH_NN_MAX_POOL_STRIDE, &PoolingBuilder::SetStrides},
        {OH_NN_MAX_POOL_PAD_MODE, &PoolingBuilder::SetPadModeOrPaddings},
        {OH_NN_MAX_POOL_PAD, &PoolingBuilder::SetPadModeOrPaddings},
        {OH_NN_MAX_POOL_ACTIVATION_TYPE, &PoolingBuilder::SetActivation},
        {OH_NN_MAX_POOL_ROUND_MODE, &PoolingBuilder::SetRoundMode},
        {OH_NN_MAX_POOL_GLOBAL, &PoolingBuilder::SetGlobal},

        {OH_NN_AVG_POOL_KERNEL_SIZE, &PoolingBuilder::SetKernel},
        {OH_NN_AVG_POOL_STRIDE, &PoolingBuilder::SetStrides},
        {OH_NN_AVG_POOL_PAD_MODE, &PoolingBuilder::SetPadModeOrPaddings},
        {OH_NN_AVG_POOL_PAD, &PoolingBuilder::SetPadModeOrPaddings},
        {OH_NN_AVG_POOL_ACTIVATION_TYPE, &PoolingBuilder::SetActivation},
        {OH_NN_AVG_POOL_ROUND_MODE, &PoolingBuilder::SetRoundMode},
        {OH_NN_AVG_POOL_GLOBAL, &PoolingBuilder::SetGlobal}
    };
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_POOLING_BUILDER_H
