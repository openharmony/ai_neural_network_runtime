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

#include "executor.h"

#include "securec.h"

#include "common/utils.h"
#include "common/scoped_trace.h"


namespace OHOS {
namespace NeuralNetworkRuntime {
Executor::Executor(const Compilation* compilation)
    : m_modelInputs(compilation->GetInputTensors()),
      m_modelOutputs(compilation->GetOutputTensors()),
      m_executionPlan(compilation->GetExecutionPlan()) {}

OH_NN_ReturnCode Executor::BuildInputTensor(uint32_t index, const OH_NN_Tensor& nnTensor,
                                            std::shared_ptr<NNTensor> inputTensor) const
{
    // Note: inputs have only shapes info.
    if (index >= m_modelInputs.size()) {
        LOGE("BuildInputTensor failed, input index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    // Build a tensor from nnTensor.
    auto ret = inputTensor->BuildFromOHNNTensor(nnTensor);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildInputTensor failed, please check input nnTensor.");
        return ret;
    }

    if (inputTensor->IsDynamicShape()) {
        LOGE("BuildInputTensor failed, input nnTensor should has certain dimensions which cannot contain -1.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (!m_modelInputs[index]->CompareAttribute(*inputTensor)) {
        LOGE("BuildInputTensor failed, input has different attributes from the one in the constructed model.");
        return OH_NN_INVALID_PARAMETER;
    }

    inputTensor->SetName(m_modelInputs[index]->GetName());
    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::SetInputTensorWithCurrentBuffer(uint32_t index,
                                                           std::shared_ptr<NNTensor> inputTensor,
                                                           const void* buffer,
                                                           size_t dataLength,
                                                           size_t curBufferLength)
{
    void* curBuffer = m_inputTensors[index].tensor->GetBuffer();
    errno_t status = memcpy_s(curBuffer, dataLength, buffer, dataLength);
    // Current buffer inside m_inputTensors is managed by executor, no need to release if memcpy failed.
    if (status != EOK) {
        LOGE("SetInputTensorWithCurrentBuffe failed, copy data from user buffer to device buffer failed. "
             "Error code: %d.", status);
        return OH_NN_MEMORY_ERROR;
    }

    // Set the new tensor with the buffer of current tensor
    inputTensor->SetBuffer(curBuffer, curBufferLength);

    // The memory is reused here. Thus, current tensor's buffer must set to nullptr, in case the memory is released
    // twice.
    m_inputTensors[index].tensor->SetBuffer(nullptr, 0);

    // Set to the new tensor, and release current one.
    m_inputTensors[index].tensor = inputTensor;
    return OH_NN_SUCCESS;
}


void Executor::SetInputTensorWithNewBuffer(uint32_t index,
                                           std::shared_ptr<NNTensor> inputTensor,
                                           const void* inputBuffer,
                                           size_t length,
                                           bool isInnerMem)
{
    // Release the memory inside the tensor first, if it is allocated by Executor during SetInput().
    if (m_inputTensors.find(index) != m_inputTensors.end()) {
        if (m_inputTensors[index].isInnerMem) {
            void* curBuffer = m_inputTensors[index].tensor->GetBuffer();
            std::shared_ptr<Device> inputDevice = m_executionPlan->GetInputDevice();
            inputDevice->ReleaseBuffer(curBuffer);
        }
        // Set current tensor's buffer to nullptr in case the NNTensor release the driver memory in destruction.
        m_inputTensors[index].tensor->SetBuffer(nullptr, 0);
    }

    // Set new input tensor data buffer
    inputTensor->SetBuffer(inputBuffer, length);

    // Create or update the input tensor
    ExeTensor exeTensor{inputTensor, nullptr, 0, isInnerMem};
    m_inputTensors[index] = exeTensor;
}


OH_NN_ReturnCode Executor::SetInput(uint32_t index, const OH_NN_Tensor& nnTensor, const void* buffer, size_t length)
{
    std::shared_ptr<NNTensor> inputTensor = CreateSharedPtr<NNTensor>();
    if (inputTensor == nullptr) {
        LOGE("SetInput failed, error happened when creating NNTensor.");
        return OH_NN_MEMORY_ERROR;
    }

    auto ret = BuildInputTensor(index, nnTensor, inputTensor);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetInput failed, please check input index or nnTensor.");
        return ret;
    }

    // dataLength will be larger than 0 after BuildInputTensor()
    size_t dataLength = inputTensor->GetDataLength();
    if (length == 0 || length < dataLength) {
        LOGE("SetInput failed, the given buffer length is too small to store the input nnTensor data.");
        return OH_NN_INVALID_PARAMETER;
    }

    // Get length of current buffer if it is allocate by SetInput() before.
    size_t curBufferLength = 0;
    if ((m_inputTensors.find(index) != m_inputTensors.end()) && (m_inputTensors[index].isInnerMem)) {
        curBufferLength = m_inputTensors[index].tensor->GetBufferLength();
    }

    // (dataLength <= curBufferLength) returns true if and only if current buffer is allocated by SetInput() before
    // and is larger than user buffer.
    if (dataLength <= curBufferLength) {
        ret = SetInputTensorWithCurrentBuffer(index, inputTensor, buffer, dataLength, curBufferLength);
        if (ret != OH_NN_SUCCESS) {
            LOGE("SetInput failed, error happened when setting input with current buffer.");
            return ret;
        }
        m_isRun = false;
        return OH_NN_SUCCESS;
    }

    /**
     * Buffer needs to allocated or reallocated if:
     *
     * - Current buffer is not enough.
     * - SetInput() has not been called for the input before.
     * - The buffer held in m_inputTensors is allocated and set by CreateInputMemory() and SetInputFromMemory().
     */
    std::shared_ptr<Device> inputDevice = m_executionPlan->GetInputDevice();
    void* inputBuffer = inputDevice->AllocateBuffer(length);
    if (inputBuffer == nullptr) {
        LOGE("SetInput failed, error happened when allocating input device buffer.");
        return OH_NN_MEMORY_ERROR;
    }

    errno_t status = memcpy_s(inputBuffer, dataLength, buffer, dataLength);
    if (status != EOK) {
        LOGE("SetInput failed, copy data from user buffer failed. Error code: %d.", status);
        inputDevice->ReleaseBuffer(inputBuffer);
        return OH_NN_MEMORY_ERROR;
    }

    SetInputTensorWithNewBuffer(index, inputTensor, inputBuffer, length, true);
    m_isRun = false;
    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::SetInputFromMemory(uint32_t index, const OH_NN_Tensor& nnTensor, const OH_NN_Memory& memory)
{
    // Build a input tensor
    std::shared_ptr<NNTensor> inputTensor = CreateSharedPtr<NNTensor>();
    if (inputTensor == nullptr) {
        LOGE("SetInputFromMemory failed, error happened when creating NNTensor.");
        return OH_NN_MEMORY_ERROR;
    }

    auto ret = BuildInputTensor(index, nnTensor, inputTensor);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetInputFromMemory failed, please check input index or nnTensor");
        return ret;
    }

    // check data length
    size_t dataLength = inputTensor->GetDataLength();
    if (memory.length == 0 || memory.length < dataLength) {
        LOGE("SetInputFromMemory failed,"
             " the length in the given memory is too small to store the input nnTensor data.");
        return OH_NN_INVALID_PARAMETER;
    }

    SetInputTensorWithNewBuffer(index, inputTensor, const_cast<const void*>(memory.data), memory.length, false);
    m_isRun = false;
    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::SetOutput(uint32_t index, void* buffer, size_t length)
{
    if (index >= m_modelOutputs.size()) {
        LOGE("SetOutput failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    size_t dataLength = m_modelOutputs[index]->GetDataLength();
    if (length == 0 || length < dataLength) {
        LOGE("SetOutput failed, the given buffer length is too small to store the output tensor data.");
        return OH_NN_INVALID_PARAMETER;
    }

    // If output tensor does not exist, or inner device buffer size is not enough,
    // or device buffer is set by SetOutputFromMemory() before,
    // allocate a new device buffer and set it to output tensor, and update the user buffer.
    std::shared_ptr<Device> outputDevice = m_executionPlan->GetOutputDevice();
    if (m_outputTensors.find(index) != m_outputTensors.end()) {
        if (m_outputTensors[index].isInnerMem) {
            size_t curBufferLength =  m_outputTensors[index].tensor->GetBufferLength();
            if (length <= curBufferLength) {
                // If current device buffer size is enough, only update the user buffer.
                m_outputTensors[index].userBuffer = buffer;
                m_outputTensors[index].userBufferLength = length;
                m_isRun = false;
                return OH_NN_SUCCESS;
            } else {
                // If current device buffer size is not enough,
                // release current device buffer and then allocate a new one below.
                void* curBuffer = m_outputTensors[index].tensor->GetBuffer();
                outputDevice->ReleaseBuffer(curBuffer);
            }
        }
    } else {
        // If output tensor does not exist, create a new null output tensor.
        ExeTensor exeTensor;
        m_outputTensors[index] = exeTensor;
        m_outputTensors[index].tensor = m_modelOutputs[index];
    }

    void* deviceOutputBuffer = outputDevice->AllocateBuffer(length);
    if (deviceOutputBuffer == nullptr) {
        LOGE("SetOutput failed, allocating output device buffer failed.");
        return OH_NN_MEMORY_ERROR;
    }

    m_outputTensors[index].tensor->SetBuffer(deviceOutputBuffer, length);
    m_outputTensors[index].userBuffer = buffer;
    m_outputTensors[index].userBufferLength = length;
    m_outputTensors[index].isInnerMem = true;
    m_isRun = false;
    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::SetOutputFromMemory(uint32_t index, const OH_NN_Memory& memory)
{
    if (index >= m_modelOutputs.size()) {
        LOGE("SetOutputFromMemory failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    size_t dataLength = m_modelOutputs[index]->GetDataLength();
    if (memory.length == 0 || memory.length < dataLength) {
        LOGE("SetOutputFromMemory failed, the memory is too small to store the output tensor data.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_outputTensors.find(index) != m_outputTensors.end()) {
        if (m_outputTensors[index].isInnerMem) {
            // If it is inner buffer, releate it
            void* curBuffer = m_outputTensors[index].tensor->GetBuffer();
            std::shared_ptr<Device> outputDevice = m_executionPlan->GetOutputDevice();
            outputDevice->ReleaseBuffer(curBuffer);
        }
    } else {
        // If output tensor does not exist, create a new null output tensor.
        ExeTensor exeTensor;
        m_outputTensors[index] = exeTensor;
        m_outputTensors[index].tensor = m_modelOutputs[index];
    }

    // Set the output tensor with memory
    m_outputTensors[index].tensor->SetBuffer(const_cast<const void*>(memory.data), memory.length);
    m_outputTensors[index].userBuffer = nullptr;
    m_outputTensors[index].userBufferLength = 0;
    m_outputTensors[index].isInnerMem = false;
    m_isRun = false;
    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::GetOutputShape(uint32_t index, int32_t** dimensions, uint32_t& dimensionCount)
{
    if (!m_isRun) {
        LOGE("GetOutputShape failed, cannot get output dimensions before Run.");
        return OH_NN_OPERATION_FORBIDDEN;
    }

    if (index >= m_modelOutputs.size()) {
        LOGE("GetOutputShape failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_outputTensors.find(index) == m_outputTensors.end()) {
        LOGE("GetOutputShape failed, output has not been set. Output index: %u.", index);
        return OH_NN_INVALID_PARAMETER;
    }

    m_outputDimensions[index] = m_outputTensors[index].tensor->GetDimensions();
    *dimensions = m_outputDimensions[index].data();
    dimensionCount = m_outputDimensions[index].size();

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::CreateInputMemory(uint32_t index, size_t length, OH_NN_Memory** memory)
{
    if (index >= m_modelInputs.size()) {
        LOGE("CreateInputMemory failed, input index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    // Allocate device buffer
    std::shared_ptr<Device> inputDevice = m_executionPlan->GetInputDevice();
    void* deviceInputBuffer = inputDevice->AllocateBuffer(length);
    if (deviceInputBuffer == nullptr) {
        LOGE("CreateInputMemory failed, allocating intput device buffer failed.");
        return OH_NN_MEMORY_ERROR;
    }

    *memory = new(std::nothrow) OH_NN_Memory{deviceInputBuffer, length};
    if (*memory == nullptr) {
        LOGE("CreateInputMemory failed, constructing OH_NN_Memory failed.");
        inputDevice->ReleaseBuffer(deviceInputBuffer);
        return OH_NN_MEMORY_ERROR;
    }

    // Save the buffer address for check when destroying it.
    m_inputCreatedMem[index].emplace_back(deviceInputBuffer);

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::DestroyInputMemory(uint32_t index, OH_NN_Memory** memory)
{
    if (index >= m_modelInputs.size()) {
        LOGE("DestroyInputMemory failed, input index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_inputCreatedMem.find(index) == m_inputCreatedMem.end()) {
        LOGE("DestroyInputMemory failed, the memory has not been created with the index.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<void*>& inputCreatedMem = m_inputCreatedMem[index];
    auto pos = std::find(inputCreatedMem.begin(), inputCreatedMem.end(), (*memory)->data);
    if (pos == inputCreatedMem.end()) {
        LOGE("DestroyInputMemory failed, the index does not match the memory.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::shared_ptr<Device> inputDevice = m_executionPlan->GetInputDevice();
    auto ret = inputDevice->ReleaseBuffer((*memory)->data);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Release input buffer failed.");
        return ret;
    }

    inputCreatedMem.erase(pos);
    delete *memory;
    *memory = nullptr;

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::CreateOutputMemory(uint32_t index, size_t length, OH_NN_Memory** memory)
{
    if (index >= m_modelOutputs.size()) {
        LOGE("CreateOutputMemory failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    // Allocate device buffer
    std::shared_ptr<Device> outputDevice = m_executionPlan->GetOutputDevice();
    void* deviceOutputBuffer = outputDevice->AllocateBuffer(length);
    if (deviceOutputBuffer == nullptr) {
        LOGE("CreateOutputMemory failed, allocating output device buffer failed.");
        return OH_NN_MEMORY_ERROR;
    }

    *memory = new(std::nothrow) OH_NN_Memory{deviceOutputBuffer, length};
    if (*memory == nullptr) {
        LOGE("CreateOutputMemory failed, constructing OH_NN_Memory failed.");
        outputDevice->ReleaseBuffer(deviceOutputBuffer);
        return OH_NN_MEMORY_ERROR;
    }

    // Save the buffer address for check when destroying it.
    m_outputCreatedMem[index].emplace_back(deviceOutputBuffer);

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::DestroyOutputMemory(uint32_t index, OH_NN_Memory** memory)
{
    if (index >= m_modelOutputs.size()) {
        LOGE("DestroyOutputMemory failed, output index is out of range.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_outputCreatedMem.find(index) == m_outputCreatedMem.end()) {
        LOGE("DestroyOutputMemory failed, the memory has not been created with the index.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<void*>& outputCreatedMem = m_outputCreatedMem[index];
    auto pos = std::find(outputCreatedMem.begin(), outputCreatedMem.end(), (*memory)->data);
    if (pos == outputCreatedMem.end()) {
        LOGE("DestroyOutputMemory failed, the index does not match the memory.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::shared_ptr<Device> outputDevice = m_executionPlan->GetOutputDevice();
    auto ret = outputDevice->ReleaseBuffer((*memory)->data);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Release output buffer failed.");
        return ret;
    }

    outputCreatedMem.erase(pos);
    delete *memory;
    *memory = nullptr;

    return OH_NN_SUCCESS;
}


OH_NN_ReturnCode Executor::Run()
{
    NNRT_TRACE_NAME("Execution");
    if (m_modelInputs.size() != m_inputTensors.size()) {
        LOGE("Run failed, some input tensors have not been set.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_modelOutputs.size() != m_outputTensors.size()) {
        LOGE("Run failed, some output tensors have not been set.");
        return OH_NN_INVALID_PARAMETER;
    }

    // Build the NNTensor pointer vector: inputTensors and outputTensors
    std::vector<std::shared_ptr<NNTensor>> inputTensors;
    std::vector<std::shared_ptr<NNTensor>> outputTensors;
    size_t inputSize = m_inputTensors.size();
    size_t outputSize = m_outputTensors.size();
    for (size_t i = 0; i < inputSize; ++i) {
        inputTensors.emplace_back(m_inputTensors[i].tensor);
    }
    for (size_t i = 0; i < outputSize; ++i) {
        outputTensors.emplace_back(m_outputTensors[i].tensor);
    }

    // Predict
    auto ret = m_executionPlan->Run(inputTensors, outputTensors);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Run failed, error happened when executing the inference.");
        return ret;
    }

    errno_t status{EOK};
    // Copy inner device buffer to user buffer if using SetOutput()
    for (size_t i = 0; i < outputSize; ++i) {
        if (m_outputTensors[i].isInnerMem) {
            auto size = outputTensors[i]->GetDataLength();
            if (size > m_outputTensors[i].userBufferLength) {
                LOGE("Output buffer size is not enough. Your size=%zu, but actual output size=%zu.",
                    m_outputTensors[i].userBufferLength, size);
                return OH_NN_INVALID_PARAMETER;
            }

            void* deviceBuffer = outputTensors[i]->GetBuffer();
            if (deviceBuffer == nullptr) {
                LOGE("Output buffer is nullptr.");
                return OH_NN_FAILED;
            }

            status = memcpy_s(m_outputTensors[i].userBuffer, m_outputTensors[i].userBufferLength, deviceBuffer, size);
            if (status != EOK) {
                LOGE("Run failed, memory copy from device buffer to user buffer failed. Error code: %d.", status);
                return OH_NN_MEMORY_ERROR;
            }
        }
    }

    m_isRun = true;
    return OH_NN_SUCCESS;
}

Executor::~Executor()
{
    std::shared_ptr<Device> inputDevice;
    for (auto& it : m_inputTensors) {
        inputDevice = m_executionPlan->GetInputDevice();
        if ((it.second).isInnerMem) {
            inputDevice->ReleaseBuffer((it.second).tensor->GetBuffer());
        }
        (it.second).tensor->SetBuffer(nullptr, 0);
        (it.second).tensor.reset();
        (it.second).userBuffer = nullptr;
    }
    m_inputTensors.clear();

    std::shared_ptr<Device> outputDevice;
    for (auto& it : m_outputTensors) {
        outputDevice = m_executionPlan->GetOutputDevice();
        if ((it.second).isInnerMem) {
            outputDevice->ReleaseBuffer((it.second).tensor->GetBuffer());
        }
        (it.second).tensor->SetBuffer(nullptr, 0);
        (it.second).tensor.reset();
        (it.second).userBuffer = nullptr;
    }
    m_outputTensors.clear();

    for (auto& it : m_inputCreatedMem) {
        it.second.clear();
    }
    m_inputCreatedMem.clear();

    for (auto& it : m_outputCreatedMem) {
        it.second.clear();
    }
    m_outputCreatedMem.clear();

    m_outputDimensions.clear();
    m_modelInputs.clear();
    m_modelOutputs.clear();
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS
