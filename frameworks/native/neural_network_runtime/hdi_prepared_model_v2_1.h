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


#ifndef NEURAL_NETWORK_RUNTIME_HDI_PREPARED_MODEL_V2_1_H
#define NEURAL_NETWORK_RUNTIME_HDI_PREPARED_MODEL_V2_1_H

#include <vector>

#include <v2_1/innrt_device.h>
#include <v2_1/iprepared_model.h>
#include <v2_1/nnrt_types.h>

#include "cpp_type.h"
#include "prepared_model.h"
#include "refbase.h"

namespace V2_1 = OHOS::HDI::Nnrt::V2_1;

namespace OHOS {
namespace NeuralNetworkRuntime {
class HDIPreparedModelV2_1 : public PreparedModel {
public:
    explicit HDIPreparedModelV2_1(OHOS::sptr<V2_1::IPreparedModel> hdiPreparedModel);

    OH_NN_ReturnCode ExportModelCache(std::vector<Buffer>& modelCache) override;

    OH_NN_ReturnCode Run(const std::vector<IOTensor>& inputs,
                         const std::vector<IOTensor>& outputs,
                         std::vector<std::vector<int32_t>>& outputsDims,
                         std::vector<bool>& isOutputBufferEnough) override;

    OH_NN_ReturnCode Run(const std::vector<NN_Tensor*>& inputs,
                         const std::vector<NN_Tensor*>& outputs,
                         std::vector<std::vector<int32_t>>& outputsDims,
                         std::vector<bool>& isOutputBufferEnough) override;

    OH_NN_ReturnCode GetInputDimRanges(std::vector<std::vector<uint32_t>>& minInputDims,
                                       std::vector<std::vector<uint32_t>>& maxInputDims) override;

private:
    // first: major version, second: minor version
    std::pair<uint32_t, uint32_t> m_hdiVersion;
    OHOS::sptr<V2_1::IPreparedModel> m_hdiPreparedModel {nullptr};
};
} // namespace NeuralNetworkRuntime
} // OHOS
#endif // NEURAL_NETWORK_RUNTIME_HDI_PREPARED_MODEL_V2_1_H
