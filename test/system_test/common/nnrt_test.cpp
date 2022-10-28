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

#include "nnrt_test.h"

#include "securec.h"

#include "common/log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace SystemTest {
namespace {
std::unique_ptr<OH_NN_QuantParam> TransformQuantParam(const CppQuantParam& cppQuantParam)
{
    // cppQuantParam.numBits empty means no quantization is applied, return nullptr directly.
    if (cppQuantParam.numBits.empty()) {
        return nullptr;
    }

    std::unique_ptr<OH_NN_QuantParam> quantParam = std::make_unique<OH_NN_QuantParam>();
    quantParam->numBits = cppQuantParam.numBits.data();
    quantParam->quantCount = cppQuantParam.numBits.size();
    quantParam->scale = cppQuantParam.scale.data();
    quantParam->zeroPoint = cppQuantParam.zeroPoint.data();
    return quantParam;
}

OH_NN_UInt32Array TransformUInt32Array(const std::vector<uint32_t>& vector)
{
    uint32_t* data = (vector.empty()) ? nullptr : const_cast<uint32_t*>(vector.data());
    return {data, vector.size()};
}
} // Anonymous namespace

// AddTensors() expects tensors do not destruct and free before the test case end.
OH_NN_ReturnCode NNRtTest::AddTensors(const std::vector<CppTensor>& cppTensors)
{
    OH_NN_Tensor tensor;
    OH_NN_ReturnCode status{OH_NN_SUCCESS};
    for (const CppTensor& cppTensor : cppTensors) {
        tensor = {
            .dataType = cppTensor.dataType,
            .dimensionCount = static_cast<uint32_t>(cppTensor.dimensions.size()),
            .dimensions = cppTensor.dimensions.empty() ? nullptr : cppTensor.dimensions.data(),
            .type = cppTensor.type
        };

        const CppQuantParam& cppQuantParam = cppTensor.quantParam;
        if ((cppQuantParam.numBits.size() != cppQuantParam.scale.size())
            || (cppQuantParam.scale.size() != cppQuantParam.zeroPoint.size())) {
                LOGE("NNRtTest::AddTensors failed, get different number of numBits, scales and zeroPoints.");
                return OH_NN_INVALID_PARAMETER;
        }
        // If no quantization is applied, quantParam == nullptr and no need to check.
        std::unique_ptr<OH_NN_QuantParam> quantParam = TransformQuantParam(cppQuantParam);
        tensor.quantParam = quantParam.get();

        m_tensors.emplace_back(tensor);
        m_quantParams.emplace_back(std::move(quantParam));

        status = OH_NNModel_AddTensor(m_model, &tensor);
        if (status != OH_NN_SUCCESS) {
            LOGE("NNRtTest::AddTensors failed, error happens when adding tensor.");
            m_tensors.clear();
            m_quantParams.clear();
            return status;
        }

        if (cppTensor.data != nullptr) {
            uint32_t index = m_tensors.size() - 1;
            status = OH_NNModel_SetTensorData(m_model, index, cppTensor.data, cppTensor.dataLength);
            if (status != OH_NN_SUCCESS) {
                LOGE("NNRtTest::AddTensors failed, error happens when setting value.");
                m_tensors.clear();
                m_quantParams.clear();
                return status;
            }
        }
    }

    return status;
}

OH_NN_ReturnCode NNRtTest::AddOperation(OH_NN_OperationType opType,
                                        const std::vector<uint32_t>& paramIndices,
                                        const std::vector<uint32_t>& inputIndices,
                                        const std::vector<uint32_t>& outputIndices)
{
    const OH_NN_UInt32Array params = TransformUInt32Array(paramIndices);
    const OH_NN_UInt32Array inputs = TransformUInt32Array(inputIndices);
    const OH_NN_UInt32Array outputs = TransformUInt32Array(outputIndices);

    OH_NN_ReturnCode status = OH_NNModel_AddOperation(m_model, opType, &params, &inputs, &outputs);
    if (status == OH_NN_SUCCESS) {
        Node node = {
            .opType = opType,
            .inputs = inputIndices,
            .outputs = outputIndices,
            .params = paramIndices
        };
        m_nodes.emplace_back(node);
    }

    return status;
}

OH_NN_ReturnCode NNRtTest::SpecifyInputAndOutput(const std::vector<uint32_t>& inputIndices,
                                                 const std::vector<uint32_t>& outputIndices)
{
    const OH_NN_UInt32Array inputs = TransformUInt32Array(inputIndices);
    const OH_NN_UInt32Array outputs = TransformUInt32Array(outputIndices);

    OH_NN_ReturnCode status = OH_NNModel_SpecifyInputsAndOutputs(m_model, &inputs, &outputs);
    if (status == OH_NN_SUCCESS) {
        m_inputs = inputIndices;
        m_outputs = outputIndices;
    }

    return status;
}

OH_NN_ReturnCode NNRtTest::SetInput(uint32_t index,
                                    const std::vector<int32_t>& dimensions,
                                    const void* buffer,
                                    size_t length)
{
    OH_NN_Tensor tensor = m_tensors[m_inputs[index]];
    tensor.dimensions = dimensions.data();

    return OH_NNExecutor_SetInput(m_executor, index, &tensor, buffer, length);
}

OH_NN_ReturnCode NNRtTest::SetOutput(uint32_t index, void* buffer, size_t length)
{
    return OH_NNExecutor_SetOutput(m_executor, index, buffer, length);
}

OH_NN_ReturnCode NNRtTest::SetInputFromMemory(uint32_t index,
                                              const std::vector<int32_t>& dimensions,
                                              const void* buffer,
                                              size_t length,
                                              OH_NN_Memory** pMemory)
{
    if (buffer == nullptr) {
        LOGE("NNRtTest::SetInputFromMemory failed, passed nullptr to buffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (pMemory == nullptr) {
        LOGE("NNRtTest::SetInputFromMemory failed, passed nullptr to pMemory.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_Memory* memory = OH_NNExecutor_AllocateInputMemory(m_executor, index, length);
    if (memory == nullptr) {
        LOGE("NNRtTest::SetInputFromMemory failed, error happened when creating input memory.");
        return OH_NN_MEMORY_ERROR;
    }

    OH_NN_Tensor tensor = m_tensors[m_inputs[index]];
    tensor.dimensions = dimensions.data();

    OH_NN_ReturnCode status = OH_NNExecutor_SetInputWithMemory(m_executor, index, &tensor, memory);
    if (status != OH_NN_SUCCESS) {
        LOGE("NNRtTest::SetInputFromMemory failed, error happened when setting input.");
        OH_NNExecutor_DestroyInputMemory(m_executor, index, &memory);
    }

    errno_t error_code = memcpy_s(const_cast<void*>(memory->data), memory->length, buffer, length);
    if (error_code != EOK) {
        LOGE("NNRtTest::SetInputFromMemory failed, error happens when copying data to OH_NN_Memory. Error code: %d.",
             error_code);
        OH_NNExecutor_DestroyInputMemory(m_executor, index, &memory);
        return OH_NN_MEMORY_ERROR;
    }

    *pMemory = memory;
    return status;
}

OH_NN_ReturnCode NNRtTest::SetOutputFromMemory(uint32_t index, size_t length, OH_NN_Memory** pMemory)
{
    if (pMemory == nullptr) {
        LOGE("NNRtTest::SetOutputFromMemory failed, passed nullptr to pMemory.");
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_Memory* memory = OH_NNExecutor_AllocateOutputMemory(m_executor, index, length);
    if (memory == nullptr) {
        LOGE("NNRtTest::SetOutputFromMemory failed, error happened when creating output memory.");
        return OH_NN_MEMORY_ERROR;
    }

    OH_NN_ReturnCode status = OH_NNExecutor_SetOutputWithMemory(m_executor, index, memory);
    if (status != OH_NN_SUCCESS) {
        LOGE("NNRtTest::SetOutputFromMemory failed, error happened when setting output.");
        OH_NNExecutor_DestroyOutputMemory(m_executor, index, &memory);
    }

    *pMemory = memory;
    return status;
}

OH_NN_ReturnCode NNRtTest::GetDevices()
{
    const size_t* devicesID{nullptr};
    uint32_t count{0};
    OH_NN_ReturnCode status = OH_NNDevice_GetAllDevicesID(&devicesID, &count);
    if (status != OH_NN_SUCCESS) {
        LOGE("NNRtTest::GetDevices failed, get all devices ID failed.");
        return status;
    }

    for (uint32_t i = 0; i < count; i++) {
        m_devices.emplace_back(devicesID[i]);
    }
    return OH_NN_SUCCESS;
}
} // namespace SystemTest
} // NeuralNetworkRuntime
} // OHOS