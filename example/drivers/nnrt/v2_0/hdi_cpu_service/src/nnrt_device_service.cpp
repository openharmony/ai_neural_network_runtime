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

#include "nnrt_device_service.h"

#include <hdf_base.h>
#include "hdf_log.h"
#include "ashmem.h"
#include "securec.h"

#include "node_registry.h"
#include "prepared_model_service.h"
#include "shared_buffer_parser.h"
#include "validation.h"
#include "utils.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V2_0 {
extern "C" INnrtDevice *NnrtDeviceImplGetInstance(void)
{
    return new (std::nothrow) NnrtDeviceService();
}

NnrtDeviceService::~NnrtDeviceService()
{
    for (auto ash : m_ashmems) {
        ash.second->UnmapAshmem();
        ash.second->CloseAshmem();
    }
}

int32_t NnrtDeviceService::GetDeviceName(std::string& name, NNRT_ReturnCode& returnCode)
{
    name = "RK3568-CPU";
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::GetVendorName(std::string& name, NNRT_ReturnCode& returnCode)
{
    name = "Rockchip";
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::GetDeviceType(DeviceType& deviceType, NNRT_ReturnCode& returnCode)
{
    deviceType = DeviceType::CPU;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::GetDeviceStatus(DeviceStatus& status, NNRT_ReturnCode& returnCode)
{
    status = DeviceStatus::AVAILABLE;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::GetSupportedOperation(const Model& model, std::vector<bool>& ops,
    NNRT_ReturnCode& returnCode)
{
    size_t nodeSize = model.nodes.size();
    auto nodes = model.nodes;
    ops.resize(nodeSize, false);
    auto& regInstance = NodeRegistry::GetSingleton();
    for (size_t i = 0; i < nodeSize; i++) {
        ops[i] = regInstance.IsNodeTypeExist(nodes[i].nodeType);
    }

    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsFloat16PrecisionSupported(bool& isSupported, NNRT_ReturnCode& returnCode)
{
    isSupported = true;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsPerformanceModeSupported(bool& isSupported, NNRT_ReturnCode& returnCode)
{
    isSupported = true;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsPrioritySupported(bool& isSupported, NNRT_ReturnCode& returnCode)
{
    isSupported = false;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsDynamicInputSupported(bool& isSupported, NNRT_ReturnCode& returnCode)
{
    isSupported = true;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::PrepareModel(const Model& model, const ModelConfig& config,
    sptr<IPreparedModel>& preparedModel, NNRT_ReturnCode& returnCode)
{
    auto ret = ValidateModel(model);
    if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
        HDF_LOGE("Model is invalid.");
        returnCode = ret;
        return GetHDFReturnCode(returnCode);
    }

    auto graph = TransModelToGraph(model, returnCode);
    if (graph == nullptr) {
        HDF_LOGE("Transfrom model to graph failed.");
        return HDF_ERR_INVALID_PARAM;
    }

    ret = ValidateModelConfig(config);
    if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
        HDF_LOGE("ModelConfig is invalid.");
        returnCode = ret;
        return GetHDFReturnCode(returnCode);
    }

    ret = ShowCustomAttributes(config.extensions);
    if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
        HDF_LOGE("Showing custom attributes failed.");
        returnCode = ret;
        return GetHDFReturnCode(returnCode);
    }

    auto context = TransModelConfig(config);
    sptr<PreparedModelService> service = new (std::nothrow) PreparedModelService(context);
    if (service == nullptr) {
        HDF_LOGE("Create new PreparedModelService instance failed.");
        returnCode = NNRT_ReturnCode::NNRT_OUT_OF_MEMORY;
        return HDF_ERR_MALLOC_FAIL;
    }

    ret = service->Compile(graph);
    if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
        HDF_LOGE("Prepared model failed.");
        returnCode = ret;
        return GetHDFReturnCode(returnCode);
    }

    preparedModel = service;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::PrepareOfflineModel(const std::vector<SharedBuffer>& offlineModels,
    const ModelConfig& config, sptr<IPreparedModel>& preparedModel, NNRT_ReturnCode& returnCode)
{
    auto ret = PrepareModelFromModelCache(offlineModels, config, preparedModel, returnCode);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Prepare offline model failed.");
        return ret;
    }

    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsModelCacheSupported(bool& isSupported, NNRT_ReturnCode& returnCode)
{
    isSupported = true;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::PrepareModelFromModelCache(const std::vector<SharedBuffer>& modelCache,
    const ModelConfig& config, sptr<IPreparedModel>& preparedModel, NNRT_ReturnCode& returnCode)
{
    HDF_LOGD("Using cache to prepare model.");

    // modelCache must be 1, because PreparedModel only export one cache file.
    if (modelCache.size() != 1) {
        HDF_LOGE("The size of modelCache vector is not valid, it should be one elememt in that vector.");
        returnCode = NNRT_ReturnCode::NNRT_INVALID_MODEL_CACHE;
        return HDF_ERR_INVALID_PARAM;
    }

    SharedBufferParser parser;
    auto result = parser.Init(modelCache[0]);
    if (result != HDF_SUCCESS) {
        HDF_LOGE("Parse model buffer failed.");
        returnCode = NNRT_ReturnCode::NNRT_INVALID_BUFFER;
        return HDF_ERR_INVALID_PARAM;
    }

    auto ret = ValidateModelConfig(config);
    if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
        HDF_LOGE("ModelConfig is invalid.");
        returnCode = ret;
        return GetHDFReturnCode(returnCode);
    }

    ret = ShowCustomAttributes(config.extensions);
    if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
        HDF_LOGE("Showing custom attributes failed.");
        returnCode = ret;
        return GetHDFReturnCode(returnCode);
    }

    auto context = TransModelConfig(config);
    sptr<PreparedModelService> service = new (std::nothrow) PreparedModelService(context);
    if (service == nullptr) {
        HDF_LOGE("Create new instance PreparedModelService failed.");
        returnCode = NNRT_ReturnCode::NNRT_OUT_OF_MEMORY;
        return HDF_ERR_MALLOC_FAIL;
    }

    void* modelBuffer = parser.GetBufferPtr();
    ret = service->Compile(modelBuffer, modelCache[0].dataSize);
    if (result != NNRT_ReturnCode::NNRT_SUCCESS) {
        HDF_LOGE("Prepared model failed.");
        returnCode = ret;
        return GetHDFReturnCode(returnCode);
    }

    preparedModel = service;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::AllocateBuffer(uint32_t length, SharedBuffer& buffer, NNRT_ReturnCode& returnCode)
{
    sptr<Ashmem> ashptr = Ashmem::CreateAshmem("allocateBuffer", length);
    if (ashptr == nullptr) {
        HDF_LOGE("Create shared memory failed.");
        returnCode = NNRT_ReturnCode::NNRT_OUT_OF_MEMORY;
        return HDF_ERR_MALLOC_FAIL;
    }

    if (!ashptr->MapReadAndWriteAshmem()) {
        HDF_LOGE("Map allocate buffer failed.");
        returnCode = NNRT_ReturnCode::NNRT_MEMORY_ERROR;
        return HDF_FAILURE;
    }

    buffer.fd = ashptr->GetAshmemFd();
    buffer.bufferSize = ashptr->GetAshmemSize();
    buffer.offset = 0;
    buffer.dataSize = length;

    m_ashmems[buffer.fd] = ashptr;
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::ReleaseBuffer(const SharedBuffer& buffer, NNRT_ReturnCode& returnCode)
{
    // parser will close current fd.
    SharedBufferParser parser;
    auto ret = parser.Init(buffer);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse buffer failed.");
        returnCode = NNRT_ReturnCode::NNRT_INVALID_BUFFER;
        return HDF_ERR_INVALID_PARAM;
    }

    for (auto& ash : m_ashmems) {
        ash.second->UnmapAshmem();
        ash.second->CloseAshmem();
    }
    m_ashmems.clear();
    
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return HDF_SUCCESS;
}

NNRT_ReturnCode NnrtDeviceService::ValidateModelConfig(const ModelConfig& config) const
{
    if (!ValidatePerformanceMode(config.mode)) {
        HDF_LOGE("PerformanceMode is invalid. mode=%d", config.mode);
        return NNRT_ReturnCode::NNRT_INVALID_PERFORMANCE_MODE;
    }

    if (!ValidatePriority(config.priority)) {
        HDF_LOGE("Priority is invalid. priority=%d", config.priority);
        return NNRT_ReturnCode::NNRT_INVALID_PRIORITY;
    }

    return NNRT_ReturnCode::NNRT_SUCCESS;
}

NNRT_ReturnCode NnrtDeviceService::ValidateModel(const Model& model) const
{
    if (model.allTensors.empty()) {
        HDF_LOGE("Model has no tensors.");
        return NNRT_ReturnCode::NNRT_INVALID_TENSOR;
    }

    if (model.subGraph.empty()) {
        HDF_LOGE("Model has no subGraphs.");
        return NNRT_ReturnCode::NNRT_INVALID_MODEL;
    }

    if (model.nodes.empty()) {
        HDF_LOGE("Model has no nodes.");
        return NNRT_ReturnCode::NNRT_INVALID_NODE;
    }

    if (model.inputIndex.empty()) {
        HDF_LOGE("Model has no input.");
        return NNRT_ReturnCode::NNRT_INVALID_INPUT;
    }

    if (model.outputIndex.empty()) {
        HDF_LOGE("Model has no output.");
        return NNRT_ReturnCode::NNRT_INVALID_OUTPUT;
    }

    size_t tensorSize = model.allTensors.size();
    for (auto index : model.inputIndex) {
        if (index > tensorSize) {
            HDF_LOGE("Input index is invalid, index=%u", index);
            return NNRT_ReturnCode::NNRT_INVALID_INPUT;
        }
    }

    for (auto index : model.outputIndex) {
        if (index > tensorSize) {
            HDF_LOGE("Output index is invalid, index=%u", index);
            return NNRT_ReturnCode::NNRT_INVALID_OUTPUT;
        }
    }

    return NNRT_ReturnCode::NNRT_SUCCESS;
}

std::shared_ptr<mindspore::schema::MetaGraphT> NnrtDeviceService::TransModelToGraph(const Model& model,
    NNRT_ReturnCode& returnCode) const
{
    auto metaGraph = std::make_shared<mindspore::schema::MetaGraphT>();
    metaGraph->name = model.name;
    metaGraph->version = mindspore::Version();

    std::unique_ptr<mindspore::schema::TensorT> transTensor{nullptr};
    for (auto tensor : model.allTensors) {
        transTensor = TransTensor(tensor, returnCode);
        if (transTensor == nullptr) {
            HDF_LOGE("Transform tensor failed.");
            return nullptr;
        }
        metaGraph->allTensors.emplace_back(std::move(transTensor));
    }
    metaGraph->inputIndex = model.inputIndex;
    metaGraph->outputIndex = model.outputIndex;

    // Transform node
    std::unique_ptr<mindspore::schema::CNodeT> transNode {nullptr};
    for (auto& node : model.nodes) {
        transNode = TransNode(node, returnCode);
        if (transNode == nullptr) {
            HDF_LOGE("Transform node failed, node name=%{public}s", node.name.c_str());
            return nullptr;
        }
        metaGraph->nodes.emplace_back(std::move(transNode));
    }

    // Transform subgraph
    const size_t numTensor = model.allTensors.size();
    for (auto graph : model.subGraph) {
        metaGraph->subGraph.emplace_back(TransSubGraph(graph, numTensor));
    }

    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return metaGraph;
}

std::unique_ptr<mindspore::schema::TensorT> NnrtDeviceService::TransTensor(const Tensor& tensor,
    NNRT_ReturnCode& returnCode) const
{
    if (!ValidateDataType(tensor.dataType)) {
        HDF_LOGE("DataType of tensor is invalid. dataType=%d", tensor.dataType);
        returnCode = NNRT_ReturnCode::NNRT_INVALID_DATATYPE;
        return nullptr;
    }

    if (!ValidateFormat(tensor.format)) {
        HDF_LOGE("Format of tensor is invalid. format=%d", tensor.format);
        returnCode = NNRT_ReturnCode::NNRT_INVALID_FORMAT;
        return nullptr;
    }

    auto schemaTensor = std::make_unique<mindspore::schema::TensorT>();
    schemaTensor->name = tensor.name;
    schemaTensor->dataType = static_cast<int32_t>(tensor.dataType);
    schemaTensor->format = static_cast<mindspore::schema::Format>(tensor.format);
    schemaTensor->dims = tensor.dims;
    for (auto param : tensor.quantParams) {
        auto quantParam = std::make_unique<mindspore::schema::QuantParamT>();
        quantParam->scale = param.scale;
        quantParam->zeroPoint = param.zeroPoint;
        quantParam->numBits = param.numBits;
        quantParam->inited = true;
        schemaTensor->quantParams.emplace_back(std::move(quantParam));
    }

    if (tensor.data.fd != INVALID_FD) {
        SharedBufferParser parser;
        auto ret = parser.Init(tensor.data);
        if (ret != HDF_SUCCESS) {
            HDF_LOGE("Parse tensor data failed.");
            returnCode = NNRT_ReturnCode::NNRT_MEMORY_ERROR;
            return nullptr;
        }

        auto data = parser.GetBufferPtr();
        schemaTensor->data.resize(tensor.data.dataSize);
        auto memRet = memcpy_s(const_cast<uint8_t*>(schemaTensor->data.data()),
                               tensor.data.dataSize, data, tensor.data.dataSize);
        if (memRet != EOK) {
            HDF_LOGW("Copy tensor data failed.");
            returnCode = NNRT_ReturnCode::NNRT_INVALID_BUFFER;
            return nullptr;
        }
    }

    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return schemaTensor;
}

std::unique_ptr<mindspore::schema::CNodeT> NnrtDeviceService::TransNode(const Node& node,
    NNRT_ReturnCode& returnCode) const
{
    auto cnode = std::make_unique<mindspore::schema::CNodeT>();
    cnode->name = node.name;
    cnode->inputIndex = node.inputIndex;
    cnode->outputIndex = node.outputIndex;
    cnode->quantType = static_cast<mindspore::schema::QuantType>(node.quantType);

    auto& regInstance = NodeRegistry::GetSingleton();
    auto parseFunc = regInstance.GetNodeFunc(node.nodeType);
    auto primitive = parseFunc(node.nodeAttr);
    if (primitive == nullptr) {
        HDF_LOGE("Parse primitve data failed. node name=%{public}s", node.name.c_str());
        returnCode = NNRT_ReturnCode::NNRT_INVALID_NODE;
        return nullptr;
    }

    cnode->primitive = std::move(primitive);
    returnCode = NNRT_ReturnCode::NNRT_SUCCESS;
    return cnode;
}

std::unique_ptr<mindspore::schema::SubGraphT> NnrtDeviceService::TransSubGraph(const SubGraph& graph,
    const size_t numTensor) const
{
    auto subGraph = std::make_unique<mindspore::schema::SubGraphT>();
    subGraph->name = graph.name;
    subGraph->inputIndices = graph.inputIndices;
    subGraph->outputIndices = graph.outputIndices;
    subGraph->nodeIndices = graph.nodeIndices;
    subGraph->tensorIndices.reserve(numTensor);
    for (size_t i = 0; i < numTensor; i++) {
        subGraph->tensorIndices.emplace_back(static_cast<uint32_t>(i));
    }
    return subGraph;
}

std::shared_ptr<mindspore::Context> NnrtDeviceService::TransModelConfig(const ModelConfig& config) const
{
    auto context = std::make_shared<mindspore::Context>();
    const int cpuThreadNum = 2;
    const int cpuNoAffinities = 0;
    const int cpuBigCore = 1;
    const int cpuLittleCore = 2;
    context->SetThreadNum(cpuThreadNum);

    int mode = cpuNoAffinities;
    switch (config.mode) {
        case PerformanceMode::PERFORMANCE_LOW:
        case PerformanceMode::PERFORMANCE_MEDIUM:
            mode = cpuLittleCore;
            break;
        case PerformanceMode::PERFORMANCE_HIGH:
        case PerformanceMode::PERFORMANCE_EXTREME:
            mode = cpuBigCore;
            break;
        default:
            mode = cpuNoAffinities;
    }
    context->SetThreadAffinity(mode);

    auto cpuInfo = std::make_shared<mindspore::CPUDeviceInfo>();
    cpuInfo->SetEnableFP16(config.enableFloat16);
    auto& deviceInfos = context->MutableDeviceInfo();
    deviceInfos.emplace_back(cpuInfo);
    return context;
}

NNRT_ReturnCode NnrtDeviceService::ShowCustomAttributes(const std::map<std::string,
    std::vector<int8_t>>& extensions) const
{
    float attr1{0.0};
    std::string attr2;

    auto ret = ParseCustomAttributes(extensions, attr1, attr2);
    if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
        HDF_LOGE("Parsing custom attributes failed.");
        return ret;
    }

    if (attr1 != 0.0f) {
        HDF_LOGI("Set attr1: %f", attr1);
    }

    if (!attr2.empty()) {
        HDF_LOGI("Set attr2: %s", attr2.c_str());
    }

    return NNRT_ReturnCode::NNRT_SUCCESS;
}

NNRT_ReturnCode NnrtDeviceService::ConvertVecToFloat(std::vector<int8_t> vecFloat, float& result) const
{
    if (vecFloat.size() != sizeof(float)) {
        HDF_LOGE("Size of the int8_t vector dose not match a float value.");
        return NNRT_ReturnCode::NNRT_INVALID_PARAMETER;
    }

    result = *(reinterpret_cast<float*>(vecFloat.data()));
    return NNRT_ReturnCode::NNRT_SUCCESS;
}

NNRT_ReturnCode NnrtDeviceService::ConvertVecToString(std::vector<int8_t> vecFloat, std::string& result) const
{
    if (vecFloat.empty()) {
        HDF_LOGE("int8_t vector is empty.");
        return NNRT_ReturnCode::NNRT_INVALID_PARAMETER;
    }

    result = reinterpret_cast<char*>(vecFloat.data());
    return NNRT_ReturnCode::NNRT_SUCCESS;
}

NNRT_ReturnCode NnrtDeviceService::ParseCustomAttributes(const std::map<std::string, std::vector<int8_t>>& extensions,
    float& attr1, std::string& attr2) const
{
    NNRT_ReturnCode ret;
    for (auto extension : extensions) {
        if (extension.first == "attr1") {
            ret = ConvertVecToFloat(extension.second, attr1);
            if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
                HDF_LOGE("ConvertVecToFloat failed.");
                return ret;
            }
            if (attr1 <= 0.0f || attr1 > 1.0f) {
                HDF_LOGE("attr1 is out of range (0,1].");
                return NNRT_ReturnCode::NNRT_INVALID_PARAMETER;
            }
        } else if (extension.first == "attr2") {
            ret = ConvertVecToString(extension.second, attr2);
            if (ret != NNRT_ReturnCode::NNRT_SUCCESS) {
                HDF_LOGE("ConvertVecToString failed.");
                return ret;
            }
            if (attr2 != "LOW" || attr2 != "HIGH") {
                HDF_LOGE("attr2 is neither LOW nor HIGH.");
                return NNRT_ReturnCode::NNRT_INVALID_PARAMETER;
            }
        }
    }

    return NNRT_ReturnCode::NNRT_SUCCESS;
}
} // V2_0
} // Nnrt
} // HDI
} // OHOS
