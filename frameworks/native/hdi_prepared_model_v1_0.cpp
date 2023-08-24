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

#include "hdi_prepared_model_v1_0.h"

#include "common/log.h"
#include "memory_manager.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace {
V1_0::DataType TransDataType(const OH_NN_DataType& dataType)
{
    switch (dataType) {
        case OH_NN_BOOL:
            return V1_0::DataType::DATA_TYPE_BOOL;
        case OH_NN_INT8:
            return V1_0::DataType::DATA_TYPE_INT8;
        case OH_NN_INT16:
            return V1_0::DataType::DATA_TYPE_INT16;
        case OH_NN_INT32:
            return V1_0::DataType::DATA_TYPE_INT32;
        case OH_NN_INT64:
            return V1_0::DataType::DATA_TYPE_INT64;
        case OH_NN_UINT8:
            return V1_0::DataType::DATA_TYPE_UINT8;
        case OH_NN_UINT16:
            return V1_0::DataType::DATA_TYPE_UINT16;
        case OH_NN_UINT32:
            return V1_0::DataType::DATA_TYPE_UINT32;
        case OH_NN_UINT64:
            return V1_0::DataType::DATA_TYPE_UINT64;
        case OH_NN_FLOAT16:
            return V1_0::DataType::DATA_TYPE_FLOAT16;
        case OH_NN_FLOAT32:
            return V1_0::DataType::DATA_TYPE_FLOAT32;
        case OH_NN_FLOAT64:
            return V1_0::DataType::DATA_TYPE_FLOAT64;
        default:
            return V1_0::DataType::DATA_TYPE_UNKNOWN;
    }
}

V1_0::Format TransFormat(const OH_NN_Format& format)
{
    switch (format) {
        case OH_NN_FORMAT_NCHW:
            return V1_0::Format::FORMAT_NCHW;
        case OH_NN_FORMAT_NHWC:
            return V1_0::Format::FORMAT_NHWC;
        default:
            return V1_0::Format::FORMAT_NONE;
    }
}

V1_0::IOTensor TransIOTensor(const IOTensor& tensor)
{
    V1_0::IOTensor iTensor;
    iTensor.name = tensor.name;
    iTensor.dataType = TransDataType(tensor.dataType);
    iTensor.dimensions = tensor.dimensions;
    iTensor.format = TransFormat(tensor.format);

    V1_0::SharedBuffer iBuffer {INVALID_FD, 0, 0, 0};
    if (tensor.data != nullptr) {
        auto memManager = MemoryManager::GetInstance();
        Memory memory;
        auto ret = memManager->GetMemory(tensor.data, memory);
        if (ret != OH_NN_SUCCESS) {
            LOGE("Invalid Tensor buffer, cannot transform to fd.");
        } else {
            iBuffer.fd = memory.fd;
            iBuffer.bufferSize = memory.length;
            iBuffer.offset = 0;
            iBuffer.dataSize = memory.length;
        }
    }
    iTensor.data = iBuffer;

    return iTensor;
}
} // unamed namespace

HDIPreparedModelV1_0::HDIPreparedModelV1_0(OHOS::sptr<V1_0::IPreparedModel> hdiPreparedModel)
    : m_hdiPreparedModel(hdiPreparedModel)
{
    hdiPreparedModel->GetVersion(m_hdiVersion.first, m_hdiVersion.second);
}

OH_NN_ReturnCode HDIPreparedModelV1_0::ExportModelCache(std::vector<Buffer>& modelCache)
{
    if (!modelCache.empty()) {
        LOGE("The vector of modelCache should be empty. size=%zu", modelCache.size());
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<V1_0::SharedBuffer> iBuffers;
    auto ret = m_hdiPreparedModel->ExportModelCache(iBuffers);
    if (ret != HDF_SUCCESS) {
        LOGE("Export model cache failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }

    auto memManager = MemoryManager::GetInstance();
    size_t iBuffersSize = iBuffers.size();
    for (size_t i = 0; i < iBuffersSize; i++) {
        auto addr = memManager->MapMemory(iBuffers[i].fd, iBuffers[i].bufferSize);
        if (addr == nullptr) {
            LOGE("Export the %zuth model cache failed, cannot not map fd to address.", i + 1);
            return OH_NN_MEMORY_ERROR;
        }
        Buffer modelbuffer {addr, iBuffers[i].bufferSize};
        modelCache.emplace_back(modelbuffer);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIPreparedModelV1_0::Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims, std::vector<bool>& isOutputBufferEnough)
{
    V1_0::IOTensor iTensor;
    std::vector<V1_0::IOTensor> iInputTensors;
    for (const auto& input: inputs) {
        iTensor = TransIOTensor(input);
        if (iTensor.data.fd == INVALID_FD) {
            LOGE("Transform inputs tensor failed, cannot find data file descriptor.");
            return OH_NN_INVALID_PARAMETER;
        }
        iInputTensors.emplace_back(iTensor);
    }

    std::vector<V1_0::IOTensor> iOutputTensors;
    for (const auto& output: outputs) {
        iTensor = TransIOTensor(output);
        if (iTensor.data.fd == INVALID_FD) {
            LOGE("Transform outputs tensor failed, cannot find data file descriptor.");
            return OH_NN_INVALID_PARAMETER;
        }
        iOutputTensors.emplace_back(iTensor);
    }

    auto ret = m_hdiPreparedModel->Run(iInputTensors, iOutputTensors, outputsDims, isOutputBufferEnough);
    if (ret != HDF_SUCCESS || outputsDims.empty()) {
        LOGE("Run model failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }

    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // OHOS