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

#include "ops_builder.h"
#include "mindir.h"
#include "mindir_types.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
void DestroyLiteGraphPrimitive(void* primitive)
{
    mindspore::lite::MindIR_Primitive_Destroy(&primitive);
}

void OpsBuilder::GetInputIndex(std::vector<uint32_t>& inputsIndex,
                               const std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID) const
{
    for (auto index : m_inputsIndex) {
        // index has been prevented from taking value out of modelIDToGraphID, no need to check.
        inputsIndex.emplace_back(modelIDToGraphID.at(index));
    }
}

void OpsBuilder::GetOutputIndex(std::vector<uint32_t>& outputsIndex,
                                const std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID) const
{
    for (auto index : m_outputsIndex) {
        // index has been prevented from taking value out of modelIDToGraphID, no need to check.
        outputsIndex.emplace_back(modelIDToGraphID.at(index));
    }
}

std::string OpsBuilder::GetName() const
{
    return m_name;
}

OpsQuantType OpsBuilder::GetQuantType() const
{
    return m_quantType;
}

OH_NN_ReturnCode OpsBuilder::CheckIOIndex(const std::vector<uint32_t>& inputsIndex,
                                          const std::vector<uint32_t>& outputsIndex,
                                          const std::vector<std::shared_ptr<NNTensor>>& allTensors,
                                          const size_t inputNum,
                                          const size_t outputNum) const
{
    size_t inputsIndexSize = inputsIndex.size();
    size_t outputIndexSize = outputsIndex.size();
    if (inputsIndexSize != inputNum) {
        LOGE("The number of index of inputs is %zu don't equal to %zu.", inputsIndexSize, inputNum);
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputIndexSize != outputNum) {
        LOGE("The number of index of outputs is %zu don't equal to %zu.", outputIndexSize, outputNum);
        return OH_NN_INVALID_PARAMETER;
    }

    for (auto index : inputsIndex) {
        if (index >= allTensors.size()) {
            LOGE("The index of inputs is out of range.");
            return OH_NN_INVALID_PARAMETER;
        }
    }

    for (auto index : outputsIndex) {
        if (index >= allTensors.size()) {
            LOGE("The index of outputs is out of range.");
            return OH_NN_INVALID_PARAMETER;
        }
    }

    return OH_NN_SUCCESS;
}

void OpsBuilder::SetQuantType(const std::vector<uint32_t>& outputsIndex,
                              const std::vector<std::shared_ptr<NNTensor>>& allTensors)
{
    if (allTensors[outputsIndex.front()]->IsQuantTensor()) {
        m_quantType = OpsQuantType::QUANT_ALL;
    }
}
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS