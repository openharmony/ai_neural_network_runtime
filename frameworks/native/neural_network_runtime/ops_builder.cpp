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
    // index has been prevented from taking value out of modelIDToGraphID, no need to check.
    std::transform(m_inputsIndex.begin(), m_inputsIndex.end(), std::back_inserter(inputsIndex),
        [modelIDToGraphID](uint32_t index) {
            return modelIDToGraphID.at(index);
        });
}

void OpsBuilder::GetOutputIndex(std::vector<uint32_t>& outputsIndex,
                                const std::unordered_map<uint32_t, uint32_t>& modelIDToGraphID) const
{
    // index has been prevented from taking value out of modelIDToGraphID, no need to check.
    std::transform(m_outputsIndex.begin(), m_outputsIndex.end(), std::back_inserter(outputsIndex),
        [modelIDToGraphID](uint32_t index) {
            return modelIDToGraphID.at(index);
        });
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
        LOGE("The number of index of inputs is %{public}zu don't equal to %{public}zu.", inputsIndexSize, inputNum);
        return OH_NN_INVALID_PARAMETER;
    }
    if (outputIndexSize != outputNum) {
        LOGE("The number of index of outputs is %{public}zu don't equal to %zu.", outputIndexSize, outputNum);
        return OH_NN_INVALID_PARAMETER;
    }

    size_t allTensorsSize = allTensors.size();
    bool isOverTensorSize = std::any_of(inputsIndex.begin(), inputsIndex.end(), [allTensorsSize](uint32_t index) {
        return index >= allTensorsSize;
    });
    if (isOverTensorSize) {
        LOGE("The index of inputs is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    isOverTensorSize = std::any_of(outputsIndex.begin(), outputsIndex.end(), [allTensorsSize](uint32_t index) {
        return index >= allTensorsSize;
    });
    if (isOverTensorSize) {
        LOGE("The index of outputs is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode OpsBuilder::CheckParamIndex(const std::vector<uint32_t>& paramsIndex,
                                             const std::vector<std::shared_ptr<NNTensor>>& allTensors,
                                             const size_t paramNum) const
{
    size_t paramsIndexSize = paramsIndex.size();
    if (paramsIndexSize > paramNum) {
        LOGE("The number of index of params is %{public}zu larger than %{public}zu.", paramsIndexSize, paramNum);
        return OH_NN_INVALID_PARAMETER;
    }

    size_t allTensorsSize = allTensors.size();
    bool isParamsOutOfRange = std::any_of(paramsIndex.begin(), paramsIndex.end(), [allTensorsSize](uint32_t index) {
        return index >= allTensorsSize;
    });
    if (isParamsOutOfRange) {
        LOGE("The index of params is out of range.");
        return OH_NN_INVALID_PARAMETER;
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