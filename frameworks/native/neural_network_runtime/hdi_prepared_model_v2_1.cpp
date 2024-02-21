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

#include "hdi_prepared_model_v2_1.h"

#include "common/log.h"
#include "hdi_returncode_utils_v2_1.h"
#include "memory_manager.h"
#include "nntensor.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace {
V2_1::DataType TransDataType(const OH_NN_DataType& dataType)
{
    switch (dataType) {
        case OH_NN_BOOL:
            return V2_1::DataType::DATA_TYPE_BOOL;
        case OH_NN_INT8:
            return V2_1::DataType::DATA_TYPE_INT8;
        case OH_NN_INT16:
            return V2_1::DataType::DATA_TYPE_INT16;
        case OH_NN_INT32:
            return V2_1::DataType::DATA_TYPE_INT32;
        case OH_NN_INT64:
            return V2_1::DataType::DATA_TYPE_INT64;
        case OH_NN_UINT8:
            return V2_1::DataType::DATA_TYPE_UINT8;
        case OH_NN_UINT16:
            return V2_1::DataType::DATA_TYPE_UINT16;
        case OH_NN_UINT32:
            return V2_1::DataType::DATA_TYPE_UINT32;
        case OH_NN_UINT64:
            return V2_1::DataType::DATA_TYPE_UINT64;
        case OH_NN_FLOAT16:
            return V2_1::DataType::DATA_TYPE_FLOAT16;
        case OH_NN_FLOAT32:
            return V2_1::DataType::DATA_TYPE_FLOAT32;
        case OH_NN_FLOAT64:
            return V2_1::DataType::DATA_TYPE_FLOAT64;
        default:
            return V2_1::DataType::DATA_TYPE_UNKNOWN;
    }
}

V2_1::Format TransFormat(const OH_NN_Format& format)
{
    switch (format) {
        case OH_NN_FORMAT_NCHW:
            return V2_1::Format::FORMAT_NCHW;
        case OH_NN_FORMAT_NHWC:
            return V2_1::Format::FORMAT_NHWC;
        default:
            return V2_1::Format::FORMAT_NONE;
    }
}

V2_1::IOTensor TransIOTensor(const IOTensor& tensor)
{
    V2_1::IOTensor iTensor;
    iTensor.name = tensor.name;
    iTensor.dataType = TransDataType(tensor.dataType);
    iTensor.dimensions = tensor.dimensions;
    iTensor.format = TransFormat(tensor.format);

    V2_1::SharedBuffer iBuffer {INVALID_FD, 0, 0, 0};
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

OH_NN_ReturnCode TransIOTensor(const NN_Tensor* tensor, V2_1::IOTensor& ioTensor)
{
    if (tensor == nullptr) {
        LOGE("TransIOTensor failed, failed to transform to V2_1 IOTensor.");
        return OH_NN_NULL_PTR;
    }

    const NNTensor2_0* nnTensor = reinterpret_cast<const NNTensor2_0*>(tensor);
    TensorDesc* nnTensorDesc = nnTensor->GetTensorDesc();
    if (nnTensorDesc == nullptr) {
        LOGE("TransIOTensor failed, failed to get desc from tensor.");
        return OH_NN_NULL_PTR;
    }

    // convert name
    const char* tensorName = nullptr;
    OH_NN_ReturnCode ret = nnTensorDesc->GetName(&tensorName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("TransIOTensor failed, failed to get name from desc.");
        return ret;
    }
    ioTensor.name = tensorName;
    
    // convert data type
    OH_NN_DataType dataType;
    ret = nnTensorDesc->GetDataType(&dataType);
    if (ret != OH_NN_SUCCESS) {
        LOGE("TransIOTensor failed, failed to get data type from desc.");
        return ret;
    }
    ioTensor.dataType = TransDataType(dataType);

    // convert format
    OH_NN_Format format;
    ret = nnTensorDesc->GetFormat(&format);
    if (ret != OH_NN_SUCCESS) {
        LOGE("TransIOTensor failed, failed to get format from desc.");
        return ret;
    }
    ioTensor.format = TransFormat(format);

    // convert shape
    int32_t* shape = nullptr;
    size_t shapeNum = 0;
    ret = nnTensorDesc->GetShape(&shape, &shapeNum);
    if (ret != OH_NN_SUCCESS) {
        LOGE("TransIOTensor failed, failed to get shape from desc.");
        return ret;
    }
    ioTensor.dimensions.clear();
    for (size_t i = 0; i < shapeNum; ++i) {
        ioTensor.dimensions.emplace_back(shape[i]);
    }

    // convert data
    if (!nnTensor->CheckTensorData()) {
        LOGE("TransIOTensor failed, failed to check tensor data.");
        return OH_NN_INVALID_PARAMETER;
    }
    V2_1::SharedBuffer iBuffer {nnTensor->GetFd(), nnTensor->GetSize(), nnTensor->GetOffset(), nnTensor->GetSize()};
    ioTensor.data = iBuffer;

    return OH_NN_SUCCESS;
}
} // unamed namespace

HDIPreparedModelV2_1::HDIPreparedModelV2_1(OHOS::sptr<V2_1::IPreparedModel> hdiPreparedModel)
    : m_hdiPreparedModel(hdiPreparedModel)
{
    hdiPreparedModel->GetVersion(m_hdiVersion.first, m_hdiVersion.second);
}

OH_NN_ReturnCode HDIPreparedModelV2_1::ExportModelCache(std::vector<Buffer>& modelCache)
{
    if (!modelCache.empty()) {
        LOGE("The vector of modelCache should be empty. size=%{public}zu", modelCache.size());
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<V2_1::SharedBuffer> iBuffers;
    auto ret = m_hdiPreparedModel->ExportModelCache(iBuffers);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_SAVE_CACHE_EXCEPTION, "Export model cache failed");
    }

    auto memManager = MemoryManager::GetInstance();
    size_t iBuffersSize = iBuffers.size();
    for (size_t i = 0; i < iBuffersSize; i++) {
        auto addr = memManager->MapMemory(iBuffers[i].fd, iBuffers[i].bufferSize);
        if (addr == nullptr) {
            LOGE("Export the %{public}zuth model cache failed, cannot not map fd to address.", i + 1);
            return OH_NN_MEMORY_ERROR;
        }
        Buffer modelbuffer {addr, iBuffers[i].bufferSize};
        modelCache.emplace_back(modelbuffer);
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIPreparedModelV2_1::Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims, std::vector<bool>& isOutputBufferEnough)
{
    V2_1::IOTensor iTensor;
    std::vector<V2_1::IOTensor> iInputTensors;
    for (const auto& input: inputs) {
        iTensor = TransIOTensor(input);
        if (iTensor.data.fd == INVALID_FD) {
            LOGE("Transform inputs tensor failed, cannot find data file descriptor.");
            return OH_NN_INVALID_PARAMETER;
        }
        iInputTensors.emplace_back(iTensor);
    }

    std::vector<V2_1::IOTensor> iOutputTensors;
    for (const auto& output: outputs) {
        iTensor = TransIOTensor(output);
        if (iTensor.data.fd == INVALID_FD) {
            LOGE("Transform outputs tensor failed, cannot find data file descriptor.");
            return OH_NN_INVALID_PARAMETER;
        }
        iOutputTensors.emplace_back(iTensor);
    }

    auto ret = m_hdiPreparedModel->Run(iInputTensors, iOutputTensors, outputsDims);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Run model failed");
    }
    if (outputsDims.empty()) {
        LOGE("Run failed, outputsDims is empty.");
        return OH_NN_UNAVAILABLE_DEVICE;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIPreparedModelV2_1::Run(const std::vector<NN_Tensor*>& inputs,
    const std::vector<NN_Tensor*>& outputs, std::vector<std::vector<int32_t>>& outputsDims,
    std::vector<bool>& isOutputBufferEnough)
{
    V2_1::IOTensor iTensor;
    std::vector<V2_1::IOTensor> iInputTensors;
    for (const auto& input: inputs) {
        auto returnCode = TransIOTensor(input, iTensor);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("Run failed, failed to transform to ioTensor.");
            return OH_NN_FAILED;
        }
        if (iTensor.data.fd == INVALID_FD) {
            LOGE("Transform inputs tensor failed, cannot find data file descriptor.");
            return OH_NN_INVALID_PARAMETER;
        }
        iInputTensors.emplace_back(iTensor);
    }

    std::vector<V2_1::IOTensor> iOutputTensors;
    for (const auto& output: outputs) {
        auto returnCode = TransIOTensor(output, iTensor);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("Run failed, failed to transform to ioTensor.");
            return OH_NN_FAILED;
        }
        if (iTensor.data.fd == INVALID_FD) {
            LOGE("Transform outputs tensor failed, cannot find data file descriptor.");
            return OH_NN_INVALID_PARAMETER;
        }
        iOutputTensors.emplace_back(iTensor);
    }

    auto ret = m_hdiPreparedModel->Run(iInputTensors, iOutputTensors, outputsDims);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Run model failed");
    }
    if (outputsDims.empty()) {
        LOGE("Run failed, outputsDims is empty.");
        return OH_NN_UNAVAILABLE_DEVICE;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIPreparedModelV2_1::GetInputDimRanges(std::vector<std::vector<uint32_t>>& minInputDims,
                                                         std::vector<std::vector<uint32_t>>& maxInputDims)
{
    auto ret = m_hdiPreparedModel->GetInputDimRanges(minInputDims, maxInputDims);
    if (ret != V2_1::NNRT_ReturnCode::NNRT_SUCCESS) {
        return CheckReturnCode_V2_1(ret, OH_NN_UNAVAILABLE_DEVICE, "Get input dim ranges failed");
    }

    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // OHOS
