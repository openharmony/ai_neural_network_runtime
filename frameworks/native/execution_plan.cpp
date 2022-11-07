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

#include "execution_plan.h"

#include <vector>

#include "common/log.h"
#include "cpp_type.h"


namespace OHOS {
namespace NeuralNetworkRuntime {
OH_NN_ReturnCode ExecutionPlan::Run(const std::vector<std::shared_ptr<NNTensor>>& inputTensors,
                                    std::vector<std::shared_ptr<NNTensor>>& outputTensors)
{
    OH_NN_ReturnCode ret {OH_NN_FAILED};
    IOTensor tensor;
    std::vector<IOTensor> inputIOTensors;
    size_t inputSize = inputTensors.size();
    size_t outputSize = outputTensors.size();
    for (size_t i = 0; i < inputSize; ++i) {
        inputTensors[i]->ConvertToIOTensor(tensor);
        inputIOTensors.emplace_back(std::move(tensor));
    }

    std::vector<IOTensor> outputIOTensors;
    for (size_t i = 0; i < outputSize; ++i) {
        outputTensors[i]->ConvertToIOTensor(tensor);
        outputIOTensors.emplace_back(std::move(tensor));
    }

    std::vector<std::vector<int32_t>> outputsDims;
    std::vector<bool> isSufficientDataBuffer;
    ret = m_preparedModel->Run(inputIOTensors, outputIOTensors, outputsDims, isSufficientDataBuffer);
    if (ret != OH_NN_SUCCESS) {
        LOGE("PrepardModel Run() failed.");
        return ret;
    }

    // Check if the output buffer is sufficient
    bool bufferFailed {false};
    for (size_t i = 0; i < outputSize; ++i) {
        if (!isSufficientDataBuffer[i]) {
            // Print all output indices with insufficient buffer, don't return until traversing all outputs.
            LOGE("Run failed, Output %zu does not have enough buffer to store the data.", i);
            bufferFailed = true;
        }
    }
    if (bufferFailed) {
        return OH_NN_FAILED;
    }

    // Set the output NNTensor's dimensions from output IOTensor if it is dynamic.
    // NNTensor::SetDimensions will check if the tensor buffer is enough for the new dimensions.
    for (size_t i = 0; i < outputSize; ++i) {
        ret = outputTensors[i]->SetDimensions(outputsDims[i]);
        if (ret != OH_NN_SUCCESS) {
            LOGE("Run failed, error happened when setting output tensor's dimensions, output id: %zu.", i);
            return ret;
        }
    }

    return OH_NN_SUCCESS;
}


std::shared_ptr<Device> ExecutionPlan::GetInputDevice() const
{
    return m_device;
}


std::shared_ptr<Device> ExecutionPlan::GetOutputDevice() const
{
    return m_device;
}
} // NeuralNetworkRuntime
} // OHOS