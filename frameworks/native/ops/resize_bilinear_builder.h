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

#ifndef NEURAL_NETWORK_RUNTIME_RESIZE_BILINEAR_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_RESIZE_BILINEAR_BUILDER_H

#include "mindir.h"

#include "frameworks/native/ops_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class ResizeBilinearBuilder : public OpsBuilder {
public:
    ResizeBilinearBuilder();
    ~ResizeBilinearBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;

    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetNewHeight(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetNewWidth(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetPreserveAspectRatio(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetCoordinateTransformMode(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetExcludeOutside(std::shared_ptr<NNTensor> tensor);

private:
    mindspore::lite::ResizeMethod m_method {mindspore::lite::RESIZE_METHOD_LINEAR};
    uint64_t m_newHeight{0};
    uint64_t m_newWidth{0};
    bool m_preserveAspectRatio{false};
    mindspore::lite::CoordinateTransformMode m_coordinateTransformMode {
        mindspore::lite::COORDINATE_TRANSFORM_MODE_ASYMMETRIC};
    uint64_t m_excludeOutside{0};
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_RESIZE_BILINEAR_BUILDER_H
