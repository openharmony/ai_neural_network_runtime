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

#ifndef NEURAL_NETWORK_RUNTIME_QUANT_PARAMS_H
#define NEURAL_NETWORK_RUNTIME_QUANT_PARAMS_H

#include <vector>

#include "cpp_type.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class QuantParams {
public:
    QuantParams() = default;
    ~QuantParams() = default;

    void SetScales(const std::vector<double>& scales);
    void SetZeroPoints(const std::vector<int32_t>& zeroPoints);
    void SetNumBits(const std::vector<uint32_t>& numBits);

    std::vector<double> GetScales() const;
    std::vector<int32_t> GetZeroPoints() const;
    std::vector<uint32_t> GetNumBits() const;

    OH_NN_ReturnCode CopyToCompat(std::vector<OHOS::NeuralNetworkRuntime::QuantParam>& compatQuantParams) const;

private:
    std::vector<double> m_scales;
    std::vector<int32_t> m_zeroPoints;
    std::vector<uint32_t> m_numBits;
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_QUANT_PARAMS_H