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

#include "prepared_model_service.h"

#include <hdf_base.h>
#include "securec.h"
#include "utils/hdf_log.h"

#include "shared_buffer_parser.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
PreparedModelService::PreparedModelService(std::shared_ptr<mindspore::Context> context)
    : m_context(context) {}

PreparedModelService::~PreparedModelService()
{
    if (m_cacheBuffer != nullptr) {
        m_cacheBuffer->CloseAshmem();
    }

    for (auto& inputAsh : m_inputAshmems) {
        inputAsh->UnmapAshmem();
        inputAsh->CloseAshmem();
    }

    for (auto& outputAsh : m_outputAshmems) {
        outputAsh->UnmapAshmem();
        outputAsh->CloseAshmem();
    }
}

int32_t PreparedModelService::ExportModelCache(std::vector<SharedBuffer>& modelCache)
{
    if (!modelCache.empty()) {
        HDF_LOGE("The parameters of ExportModelCache should be an empty vector.");
        return HDF_ERR_INVALID_PARAM;
    }

    if (m_cacheBuffer != nullptr) {
        auto fd = m_cacheBuffer->GetAshmemFd();
        auto size = m_cacheBuffer->GetAshmemSize();

        // SharedBuffer: fd, bufferSize, offset, dataSize
        modelCache.emplace_back(SharedBuffer{fd, size, 0, size});
        return HDF_SUCCESS;
    }

    auto size = m_builder.GetSize();
    auto buffer = m_builder.GetBufferPointer();
    const char* name = m_graph != nullptr ? m_graph->name.c_str() : "CacheModel";
    sptr<Ashmem> cache = Ashmem::CreateAshmem(name, size);
    if (cache == nullptr) {
        HDF_LOGE("Create shared memory failed.");
        return HDF_ERR_MALLOC_FAIL;
    }

    bool ret = cache->MapReadAndWriteAshmem();
    if (!ret) {
        HDF_LOGE("Map fd to write cache failed.");
        return HDF_FAILURE;
    }

    ret = cache->WriteToAshmem(buffer, size, 0);
    cache->UnmapAshmem();
    if (!ret) {
        HDF_LOGE("Write cache failed.");
        return HDF_FAILURE;
    }

    m_cacheBuffer = cache;

    // SharedBuffer: fd, bufferSize, offset, dataSize
    modelCache.emplace_back(SharedBuffer {cache->GetAshmemFd(), cache->GetAshmemSize(), 0, cache->GetAshmemSize()});
    return HDF_SUCCESS;
}

int32_t PreparedModelService::Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims, std::vector<bool>& isOutputBufferEnough)
{
    auto ret = SetInputs(inputs);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Inputs tensor is invalid.");
        return ret;
    }

    if (!m_isDynamicShape) {
        ret = SetOutputs(outputs);
        if (ret != HDF_SUCCESS) {
            HDF_LOGE("Output tensor is invalid.");
            ResetInputAndOutput();
            return ret;
        }
    }

    auto msRet = m_model->Predict(m_inputs, &m_outputs);
    if (msRet != mindspore::kSuccess) {
        HDF_LOGE("Run model failed.");
        ResetInputAndOutput();
        return HDF_FAILURE;
    }

    ret = UpdateOutput(outputs, outputsDims, isOutputBufferEnough);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Update output dimension or data failed.");
        ResetInputAndOutput();
        return ret;
    }

    ResetInputAndOutput();

    return HDF_SUCCESS;
}

int32_t PreparedModelService::UpdateOutput(const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims, std::vector<bool>& isOutputBufferEnough)
{
    bool isEnough {true};
    size_t outputSize = m_outputs.size();
    isOutputBufferEnough.resize(outputSize, true);
    for (size_t i = 0; i < outputSize; i++) {
        auto& msOutput = m_outputs[i];
        auto& output = outputs[i];

        auto msShape = msOutput.Shape();
        outputsDims.emplace_back(msShape.begin(), msShape.end());

        auto dataSize = msOutput.DataSize();
        if (dataSize > output.data.bufferSize) {
            HDF_LOGE("Output buffer is not enough. actual size %{public}zu, buffer size %{public}u",
                dataSize, output.data.bufferSize);
            isOutputBufferEnough[i] = false;
            isEnough= false;
        }

        if (isEnough && m_isDynamicShape) {
            auto msData = msOutput.MutableData();
            SharedBufferParser parser;
            auto ret = parser.Init(output.data);
            if (ret != HDF_SUCCESS) {
                HDF_LOGE("Parse %zu th output data failed.", i);
                return HDF_ERR_INVALID_PARAM;
            }

            auto data = parser.GetBufferPtr();
            auto memRet = memcpy_s(data, dataSize, msData, dataSize);
            if (memRet != EOK) {
                HDF_LOGE("Copy output memory failed.");
                return HDF_FAILURE;
            }
        }
    }

    return HDF_SUCCESS;
}

void PreparedModelService::ResetInputAndOutput()
{
    for (auto& msInput : m_inputs) {
        msInput.SetData(nullptr);
    }

    if (!m_isDynamicShape) {
        for (auto& msOutput : m_outputs) {
            msOutput.SetData(nullptr);
        }
    }
}

int32_t PreparedModelService::Compile(std::shared_ptr<mindspore::schema::MetaGraphT> graph)
{
    if (graph == nullptr) {
        HDF_LOGE("Graph cannot be nullptr");
        return HDF_ERR_INVALID_PARAM;
    }
    for (auto i : graph->inputIndex) {
        auto inputShape = graph->allTensors[i]->dims;
        auto iter = std::find(inputShape.begin(), inputShape.end(), DYNAMIC_SHAPE_FLAG);
        if (iter != inputShape.end()) {
            m_isDynamicShape = true;
            break;
        }
    }
    auto offset = mindspore::schema::MetaGraph::Pack(m_builder, graph.get());
    m_builder.Finish(offset);
    mindspore::schema::FinishMetaGraphBuffer(m_builder, offset);
    auto modelSize = m_builder.GetSize();
    uint8_t* modelBuffer = m_builder.GetBufferPointer();
    if (modelBuffer == nullptr) {
        HDF_LOGE("Model is invalid.");
        return HDF_FAILURE;
    }

    m_model = std::make_shared<mindspore::Model>();
    mindspore::Status msRet = m_model->Build(modelBuffer, modelSize, mindspore::kMindIR, m_context);
    if (msRet != mindspore::kSuccess) {
        HDF_LOGE("Prepare model failed, please make sure model is validate.");
        return HDF_FAILURE;
    }

    auto ret = GetMSInputsAndOutputs();
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Model without inputs or outputs is invalid.");
        return ret;
    }
    return HDF_SUCCESS;
}

int32_t PreparedModelService::Compile(const void* modelBuffer, size_t length)
{
    if (modelBuffer == nullptr || length == 0) {
        HDF_LOGE("ModelBuffer cannot be nullptr and length cannot be zero.");
        return HDF_ERR_INVALID_PARAM;
    }

    m_model = std::make_shared<mindspore::Model>();
    mindspore::Status msRet = m_model->Build(modelBuffer, length, mindspore::kMindIR, m_context);
    if (msRet != mindspore::kSuccess) {
        HDF_LOGE("Prepare model from cache failed, please make sure model cache is valid.");
        return HDF_FAILURE;
    }

    auto ret = GetMSInputsAndOutputs();
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Model without inputs or outputs is invalid.");
        return ret;
    }

    for (auto input : m_inputs) {
        auto shapes = input.Shape();
        if (std::find(shapes.begin(), shapes.end(), DYNAMIC_SHAPE_FLAG) != shapes.end()) {
            m_isDynamicShape = true;
            break;
        }
    }
    return HDF_SUCCESS;
}

int32_t PreparedModelService::SetInputs(const std::vector<IOTensor>& inputs)
{
    if (inputs.size() != m_inputs.size()) {
        HDF_LOGE("inputs size is invalid. expect: %zu, actual: %zu", m_inputs.size(), inputs.size());
        return HDF_ERR_INVALID_PARAM;
    }
    for (auto& ash : m_inputAshmems) {
        ash->UnmapAshmem();
        ash->CloseAshmem();
    }
    m_inputAshmems.clear();

    int32_t ret {0};
    size_t inputSize = m_inputs.size();
    std::vector<std::vector<int64_t>> tmpAllDims;
    for (size_t i = 0; i < inputSize; i++) {
        auto& input = inputs[i];
        auto& msInput = m_inputs[i];
        ret = CompareTensor(input, msInput);
        if (ret != HDF_SUCCESS) {
            HDF_LOGE("Inputs tensor is not match that of model. Please check input tensor.");
            return ret;
        }
        tmpAllDims.emplace_back(input.dimensions.begin(), input.dimensions.end());
    }

    if (m_isDynamicShape) {
        auto msRet = m_model->Resize(m_inputs, tmpAllDims);
        if (msRet != mindspore::kSuccess) {
            HDF_LOGE("Resize for dynamic inputs failed.");
            return HDF_FAILURE;
        }
        ret = GetMSInputsAndOutputs();
        if (ret != HDF_SUCCESS) {
            HDF_LOGE("Get ms inputs or outputs failed after resize.");
            return ret;
        }
    }

    for (size_t i = 0; i < inputSize; i++) {
        auto& input = inputs[i];
        auto& msInput = m_inputs[i];
        sptr<Ashmem> ashptr = ParseBuffer(input.data);
        if (ashptr == nullptr) {
            HDF_LOGE("Parse %zuth input data failed.", i);
            return HDF_ERR_INVALID_PARAM;
        }

        auto data = const_cast<void*>(ashptr->ReadFromAshmem(input.data.dataSize, 0));
        msInput.SetData(data);
        m_inputAshmems.emplace_back(ashptr);
    }
    return HDF_SUCCESS;
}

int32_t PreparedModelService::SetOutputs(const std::vector<IOTensor>& outputs)
{
    HDF_LOGI("Start Set outputs, m_outputs size=%zu", m_outputs.size());
    if (outputs.size() != m_outputs.size()) {
        HDF_LOGE("outputs size is invalid. expect: %{public}zu, actual: %{public}zu", m_outputs.size(), outputs.size());
        return HDF_ERR_INVALID_PARAM;
    }
    for (auto ash : m_outputAshmems) {
        ash->UnmapAshmem();
        ash->CloseAshmem();
    }
    m_outputAshmems.clear();

    for (size_t i = 0; i < m_outputs.size(); i++) {
        auto& output = outputs[i];
        auto& msOutput = m_outputs[i];

        sptr<Ashmem> ashptr = ParseBuffer(output.data);
        if (ashptr == nullptr) {
            HDF_LOGE("Parse %{public}zu th output data failed.", i);
            return HDF_ERR_INVALID_PARAM;
        }

        auto data = const_cast<void*>(ashptr->ReadFromAshmem(output.data.dataSize, 0));
        msOutput.SetAllocator(nullptr);
        msOutput.SetData(data);
        m_outputAshmems.emplace_back(ashptr);
    }
    return HDF_SUCCESS;
}

int32_t PreparedModelService::GetMSInputsAndOutputs()
{
    m_inputs = m_model->GetInputs();
    if (m_inputs.empty()) {
        HDF_LOGE("Get inputs failed.");
        return HDF_FAILURE;
    }

    m_outputs = m_model->GetOutputs();
    if (m_outputs.empty()) {
        HDF_LOGE("Get outputs failed.");
        return HDF_FAILURE;
    }
    return HDF_SUCCESS;
}

int32_t PreparedModelService::CompareTensor(const IOTensor& tensor, const mindspore::MSTensor& msTensor)
{
    auto dataType = static_cast<DataType>(msTensor.DataType());
    if (tensor.dataType != dataType) {
        HDF_LOGE("Data type of tensor is not match that of model.");
        return HDF_ERR_INVALID_PARAM;
    }

    auto format = static_cast<Format>(msTensor.format());
    if (tensor.format != format) {
        HDF_LOGE("Format of tensor is not match that of model.");
        return HDF_ERR_INVALID_PARAM;
    }

    for (size_t i = 0; i < tensor.dimensions.size(); i++) {
        if (msTensor.Shape()[i] != DYNAMIC_SHAPE_FLAG && tensor.dimensions[i] != msTensor.Shape()[i]) {
            HDF_LOGE("The Shape of tensor is not match that of model.");
            return HDF_ERR_INVALID_PARAM;
        }
    }

    return HDF_SUCCESS;
}

sptr<Ashmem> PreparedModelService::ParseBuffer(const SharedBuffer& buffer)
{
    if (buffer.fd == -1) {
        HDF_LOGE("Invalid buffer fd, it cannot be -1.");
        return nullptr;
    }

    HDF_LOGW("NNRT buffer fd=%{public}d, length=%{public}u", buffer.fd, buffer.dataSize);

    sptr<Ashmem> ashptr = new (std::nothrow) Ashmem(buffer.fd, buffer.bufferSize);
    if (ashptr == nullptr) {
        HDF_LOGE("Create shared memory failed.");
        return nullptr;
    }

    if (!ashptr->MapReadAndWriteAshmem()) {
        HDF_LOGE("Map buffer fd to address failed.");
        return nullptr;
    }

    const void* data = ashptr->ReadFromAshmem(buffer.dataSize, buffer.offset);
    if (data == nullptr) {
        HDF_LOGE("Get data address failed.");
        ashptr->UnmapAshmem();
        ashptr->CloseAshmem();
        return nullptr;
    }
    return ashptr;
}
} // V1_0
} // Nnrt
} // HDI
} // OHOS
