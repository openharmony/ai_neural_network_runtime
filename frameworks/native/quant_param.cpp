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

#include "quant_param.h"

#include "common/log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
void QuantParams::SetScales(const std::vector<double>& scales)
{
    m_scales = scales;
}

void QuantParams::SetZeroPoints(const std::vector<int32_t>& zeroPoints)
{
    m_zeroPoints = zeroPoints;
}

void QuantParams::SetNumBits(const std::vector<uint32_t>& numBits)
{
    m_numBits = numBits;
}

std::vector<double> QuantParams::GetScales() const
{
    return m_scales;
}

std::vector<int32_t> QuantParams::GetZeroPoints() const
{
    return m_zeroPoints;
}

std::vector<uint32_t> QuantParams::GetNumBits() const
{
    return m_numBits;
}

OH_NN_ReturnCode QuantParams::CopyToCompat(std::vector<OHOS::NeuralNetworkRuntime::QuantParam>& compatQuantParams) const
{
    if ((m_scales.size() != m_zeroPoints.size()) || (m_zeroPoints.size() != m_numBits.size())) {
        LOGE("CopyToCompat failed, the size of scales(%zu), zeroPoints(%zu) and numBits(%zu) are not equal.",
            m_scales.size(), m_zeroPoints.size(), m_numBits.size());
        return OH_NN_INVALID_PARAMETER;
    }

    size_t quantCount = m_scales.size();
    for (size_t i = 0; i < quantCount; i++) {
        compatQuantParams.push_back({
            .numBits = m_numBits[i],
            .scale = m_scales[i],
            .zeroPoint = m_zeroPoints[i],
        });
    }

    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS