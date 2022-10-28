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

#ifndef NEURAL_NETWORK_RUNTIME_DEPTHWISE_CONV2D_NATIVE_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_DEPTHWISE_CONV2D_NATIVE_BUILDER_H

#include "frameworks/native/ops_builder.h"
#include "frameworks/native/ops_registry.h"
#include "mindir.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class DepthwiseConv2DNativeBuilder : public OpsBuilder {
public:
    DepthwiseConv2DNativeBuilder();
    ~DepthwiseConv2DNativeBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;
    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetInputAndOutput(const std::vector<uint32_t>& inputsIndex,
        const std::vector<uint32_t>& outputsIndex, const std::vector<std::shared_ptr<NNTensor>>& allTensors);
    OH_NN_ReturnCode SetIsPadMode(std::shared_ptr<NNTensor> tensor,
        bool &isPadMode);
    OH_NN_ReturnCode SetPadModeOrPaddings(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetKernelSize(const std::vector<uint32_t>& inputsIndex,
        const std::vector<std::shared_ptr<NNTensor>>& allTensors);
    OH_NN_ReturnCode SetDilation(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetStrides(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetActivation(std::shared_ptr<NNTensor> tensor);

private:
    int64_t m_inChannel{0};
    int64_t m_outChannel{0};
    std::vector<int64_t> m_kernelSize;
    std::vector<int64_t> m_strides;
    std::vector<int64_t> m_pad;
    std::vector<int64_t> m_dilation;
    mindspore::lite::PadMode m_padMode{mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType m_activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_DEPTHWISE_CONV2D_NATIVE_BUILDER_H