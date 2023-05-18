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
#include "utils/hdf_log.h"
#include "ashmem.h"
#include "securec.h"

#include "node_registry.h"
#include "prepared_model_service.h"
#include "shared_buffer_parser.h"
#include "validation.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
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

int32_t NnrtDeviceService::GetDeviceName(std::string& name)
{
    name = "RK3568-CPU";
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::GetVendorName(std::string& name)
{
    name = "Rockchip";
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::GetDeviceType(DeviceType& deviceType)
{
    deviceType = DeviceType::CPU;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::GetDeviceStatus(DeviceStatus& status)
{
    status = DeviceStatus::AVAILABLE;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::GetSupportedOperation(const Model& model, std::vector<bool>& ops)
{
    size_t nodeSize = model.nodes.size();
    auto nodes = model.nodes;
    ops.resize(nodeSize, false);
    auto& regInstance = NodeRegistry::GetSingleton();
    for (size_t i = 0; i < nodeSize; i++) {
        ops[i] = regInstance.IsNodeTypeExist(nodes[i].nodeType);
    }
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsFloat16PrecisionSupported(bool& isSupported)
{
    isSupported = true;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsPerformanceModeSupported(bool& isSupported)
{
    isSupported = true;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsPrioritySupported(bool& isSupported)
{
    isSupported = false;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsDynamicInputSupported(bool& isSupported)
{
    isSupported = true;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::PrepareModel(const Model& model, const ModelConfig& config,
    sptr<IPreparedModel>& preparedModel)
{
    auto ret = ValidateModel(model);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Model is invalid.");
        return ret;
    }

    auto graph = TransModelToGraph(model);
    if (graph == nullptr) {
        HDF_LOGE("Transfrom model to graph failed.");
        return HDF_ERR_INVALID_PARAM;
    }

    ret = ValidateModelConfig(config);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("ModelConfig is invalid.");
        return ret;
    }

    auto context = TransModelConfig(config);
    sptr<PreparedModelService> service = new (std::nothrow) PreparedModelService(context);
    if (service == nullptr) {
        HDF_LOGE("Create new PreparedModelService instance failed.");
        return HDF_ERR_MALLOC_FAIL;
    }

    ret = service->Compile(graph);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Prepared model failed.");
        return ret;
    }

    preparedModel = service;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::IsModelCacheSupported(bool& isSupported)
{
    isSupported = true;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::PrepareModelFromModelCache(const std::vector<SharedBuffer>& modelCache,
    const ModelConfig& config, sptr<IPreparedModel>& preparedModel)
{
    HDF_LOGD("Using cache to prepare model.");

    // modelCache must be 1, because PreparedModel only export one cache file.
    if (modelCache.size() != 1) {
        HDF_LOGE("The size of modelCache vector is not valid, it should be one elememt in that vector.");
        return HDF_ERR_INVALID_PARAM;
    }

    SharedBufferParser parser;
    auto ret = parser.Init(modelCache[0]);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse modle buffer failed.");
        return HDF_ERR_INVALID_PARAM;
    }

    void* modelBuffer = parser.GetBufferPtr();
    auto context = TransModelConfig(config);
    sptr<PreparedModelService> service = new (std::nothrow) PreparedModelService(context);
    if (service == nullptr) {
        HDF_LOGE("Create new instance PreparedModelService failed.");
        return HDF_ERR_MALLOC_FAIL;
    }

    ret = service->Compile(modelBuffer, modelCache[0].dataSize);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Prepared model failed.");
        return ret;
    }

    preparedModel = service;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::AllocateBuffer(uint32_t length, SharedBuffer& buffer)
{
    sptr<Ashmem> ashptr = Ashmem::CreateAshmem("allocateBuffer", length);
    if (ashptr == nullptr) {
        HDF_LOGE("Create shared memory failed.");
        return HDF_FAILURE;
    }

    if (!ashptr->MapReadAndWriteAshmem()) {
        HDF_LOGE("Map allocate buffer failed.");
        return HDF_FAILURE;
    }

    buffer.fd = ashptr->GetAshmemFd();
    buffer.bufferSize = ashptr->GetAshmemSize();
    buffer.offset = 0;
    buffer.dataSize = length;

    m_ashmems[buffer.fd] = ashptr;
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::ReleaseBuffer(const SharedBuffer& buffer)
{
    // parser will close current fd.
    SharedBufferParser parser;
    auto ret = parser.Init(buffer);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Parse buffer failed.");
        return HDF_ERR_INVALID_PARAM;
    }

    for (auto& ash : m_ashmems) {
        ash.second->UnmapAshmem();
        ash.second->CloseAshmem();
    }
    m_ashmems.clear();
    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::ValidateModelConfig(const ModelConfig& config) const
{
    if (!ValidatePerformanceMode(config.mode)) {
        HDF_LOGE("PerformanceMode is invalid. mode=%d", config.mode);
        return HDF_ERR_INVALID_PARAM;
    }

    if (!ValidatePriority(config.priority)) {
        HDF_LOGE("Priority is invalid. priority=%d", config.priority);
        return HDF_ERR_INVALID_PARAM;
    }

    return HDF_SUCCESS;
}

int32_t NnrtDeviceService::ValidateModel(const Model& model) const
{
    if (model.allTensors.empty()) {
        HDF_LOGE("Model has no tensors.");
        return HDF_ERR_INVALID_PARAM;
    }

    if (model.subGraph.empty()) {
        HDF_LOGE("Model has no subGraphs.");
        return HDF_ERR_INVALID_PARAM;
    }

    if (model.nodes.empty()) {
        HDF_LOGE("Model has no nodes.");
        return HDF_ERR_INVALID_PARAM;
    }

    if (model.inputIndex.empty()) {
        HDF_LOGE("Model has no input.");
        return HDF_ERR_INVALID_PARAM;
    }

    if (model.outputIndex.empty()) {
        HDF_LOGE("Model has no output.");
        return HDF_ERR_INVALID_PARAM;
    }

    size_t tensorSize = model.allTensors.size();
    for (auto index : model.inputIndex) {
        if (index > tensorSize) {
            HDF_LOGE("Input index is invalid, index=%u", index);
            return HDF_ERR_INVALID_PARAM;
        }
    }

    for (auto index : model.outputIndex) {
        if (index > tensorSize) {
            HDF_LOGE("Output index is invalid, index=%u", index);
            return HDF_ERR_INVALID_PARAM;
        }
    }

    return HDF_SUCCESS;
}

std::shared_ptr<mindspore::schema::MetaGraphT> NnrtDeviceService::TransModelToGraph(const Model& model) const
{
    auto metaGraph = std::make_shared<mindspore::schema::MetaGraphT>();
    metaGraph->name = model.name;
    metaGraph->version = mindspore::Version();

    std::unique_ptr<mindspore::schema::TensorT> transTensor{nullptr};
    for (auto tensor : model.allTensors) {
        transTensor = TransTensor(tensor);
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
        transNode = TransNode(node);
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
    return metaGraph;
}

std::unique_ptr<mindspore::schema::TensorT> NnrtDeviceService::TransTensor(const Tensor& tensor) const
{
    if (!ValidateDataType(tensor.dataType)) {
        HDF_LOGE("DataType of tensor is invalid. dataType=%d", tensor.dataType);
        return nullptr;
    }

    if (!ValidateFormat(tensor.format)) {
        HDF_LOGE("Format of tensor is invalid. format=%d", tensor.format);
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
            return nullptr;
        }

        auto data = parser.GetBufferPtr();
        schemaTensor->data.resize(tensor.data.dataSize);
        auto memRet = memcpy_s(const_cast<uint8_t*>(schemaTensor->data.data()),
                               tensor.data.dataSize, data, tensor.data.dataSize);
        if (memRet != EOK) {
            HDF_LOGW("Copy tensor data failed.");
            return nullptr;
        }
    }
    return schemaTensor;
}

std::unique_ptr<mindspore::schema::CNodeT> NnrtDeviceService::TransNode(const Node& node) const
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
        return nullptr;
    }

    cnode->primitive = std::move(primitive);
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
    for (auto i = 0; i < numTensor; i++) {
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
} // V1_0
} // Nnrt
} // HDI
} // OHOS
