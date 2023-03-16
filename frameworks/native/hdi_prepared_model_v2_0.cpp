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

#include "hdi_prepared_model_v2_0.h"

#include "common/log.h"
#include "memory_manager.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace {
V2_0::DataType TransDataType(const OH_NN_DataType& dataType)
{
    switch (dataType) {
        case OH_NN_BOOL:
            return V2_0::DataType::DATA_TYPE_BOOL;
        case OH_NN_INT8:
            return V2_0::DataType::DATA_TYPE_INT8;
        case OH_NN_INT16:
            return V2_0::DataType::DATA_TYPE_INT16;
        case OH_NN_INT32:
            return V2_0::DataType::DATA_TYPE_INT32;
        case OH_NN_INT64:
            return V2_0::DataType::DATA_TYPE_INT64;
        case OH_NN_UINT8:
            return V2_0::DataType::DATA_TYPE_UINT8;
        case OH_NN_UINT16:
            return V2_0::DataType::DATA_TYPE_UINT16;
        case OH_NN_UINT32:
            return V2_0::DataType::DATA_TYPE_UINT32;
        case OH_NN_UINT64:
            return V2_0::DataType::DATA_TYPE_UINT64;
        case OH_NN_FLOAT16:
            return V2_0::DataType::DATA_TYPE_FLOAT16;
        case OH_NN_FLOAT32:
            return V2_0::DataType::DATA_TYPE_FLOAT32;
        case OH_NN_FLOAT64:
            return V2_0::DataType::DATA_TYPE_FLOAT64;
        default:
            return V2_0::DataType::DATA_TYPE_UNKNOWN;
    }
}

V2_0::Format TransFormat(const OH_NN_Format& format)
{
    switch (format) {
        case OH_NN_FORMAT_NCHW:
            return V2_0::Format::FORMAT_NCHW;
        case OH_NN_FORMAT_NHWC:
            return V2_0::Format::FORMAT_NHWC;
        default:
            return V2_0::Format::FORMAT_NONE;
    }
}

V2_0::IOTensor TransIOTensor(const IOTensor& tensor)
{
    V2_0::IOTensor iTensor;
    iTensor.name = tensor.name;
    iTensor.dataType = TransDataType(tensor.dataType);
    iTensor.dimensions = tensor.dimensions;
    iTensor.format = TransFormat(tensor.format);

    V2_0::SharedBuffer iBuffer {INVALID_FD, 0, 0, 0};
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

HDIPreparedModelV2_0::HDIPreparedModelV2_0(OHOS::sptr<V2_0::IPreparedModel> hdiPreparedModel)
    : m_hdiPreparedModel(hdiPreparedModel)
{
    hdiPreparedModel->GetVersion(m_hdiVersion.first, m_hdiVersion.second);
}

OH_NN_ReturnCode HDIPreparedModelV2_0::ExportModelCache(std::vector<ModelBuffer>& modelCache)
{
    if (!modelCache.empty()) {
        LOGE("The vector of modelCache should be empty. size=%zu", modelCache.size());
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<V2_0::SharedBuffer> iBuffers;
    auto ret = m_hdiPreparedModel->ExportModelCache(iBuffers);
    if (ret != HDF_SUCCESS) {
        LOGE("Export model cache failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }

    auto memManager = MemoryManager::GetInstance();
    for (size_t i = 0; i < iBuffers.size(); i++) {
        auto addr = memManager->MapMemory(iBuffers[i].fd, iBuffers[i].bufferSize);
        if (addr == nullptr) {
            LOGE("Export the %zuth model cache failed, cannot not map fd to address.", i + 1);
            return OH_NN_MEMORY_ERROR;
        }
        ModelBuffer modelbuffer {addr, iBuffers[i].bufferSize};
        modelCache.emplace_back(modelbuffer);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIPreparedModelV2_0::Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims, std::vector<bool>& isOutputBufferEnough)
{
    V2_0::IOTensor iTensor;
    std::vector<V2_0::IOTensor> iInputTensors;
    for (auto& input: inputs) {
        iTensor = TransIOTensor(input);
        if (iTensor.data.fd == INVALID_FD) {
            LOGE("Transform inputs tensor failed, cannot find data file descriptor.");
            return OH_NN_INVALID_PARAMETER;
        }
        iInputTensors.emplace_back(iTensor);
    }

    std::vector<V2_0::IOTensor> iOutputTensors;
    for (auto& output: outputs) {
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

OH_NN_ReturnCode HDIPreparedModelV2_0::GetInputDimRanges(std::vector<std::vector<uint32_t>>& minInputDims,
                                                         std::vector<std::vector<uint32_t>>& maxInputDims)
{
    auto ret = m_hdiPreparedModel->GetInputDimRanges(minInputDims, maxInputDims);
    if (ret != HDF_SUCCESS) {
        LOGE("GetInputDimRanges failed. ErrorCode=%d", ret);
        return OH_NN_UNAVALIDABLE_DEVICE;
    }

    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // OHOS