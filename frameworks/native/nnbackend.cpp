/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#include "nnbackend.h"

#include <new>
#include "common/log.h"
#include "common/utils.h"
#include "nncompiler.h"
#include "nnexecutor.h"
#include "nntensor.h"
#include "tensor_desc.h"
#include "device.h"


namespace OHOS {
namespace NeuralNetworkRuntime {
NNBackend::NNBackend(const std::shared_ptr<Device>& device, size_t backendID)
    : m_device(device),
    m_backendID(backendID) {}

NNBackend::~NNBackend()
{
    m_device = nullptr;
}

size_t NNBackend::GetBackendID() const
{
    return m_backendID;
}

OH_NN_ReturnCode NNBackend::GetBackendName(std::string& backendName) const
{
    if (m_device == nullptr) {
        LOGE("[NNBackend] GetBackendName failed, m_device is nullptr");
        return OH_NN_FAILED;
    }

    std::string deviceName;
    OH_NN_ReturnCode ret = m_device->GetDeviceName(deviceName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNBackend] GetBackendName failed, get device name failed.");
        return ret;
    }

    std::string vendorName;
    ret = m_device->GetVendorName(vendorName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNBackend] GetBackendName failed, get vendor name failed.");
        return ret;
    }

    std::string version;
    ret = m_device->GetVersion(version);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNBackend] GetBackendName failed, get version failed.");
        return ret;
    }

    backendName = GenUniqueName(deviceName, vendorName, version);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNBackend::GetBackendType(OH_NN_DeviceType& backendType) const
{
    if (m_device == nullptr) {
        LOGE("[NNBackend] GetBackendType failed, m_device is nullptr");
        return OH_NN_FAILED;
    }

    OH_NN_ReturnCode ret = m_device->GetDeviceType(backendType);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNBackend] GetBackendType failed, fail to get device type");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNBackend::GetBackendStatus(DeviceStatus& status) const
{
    if (m_device == nullptr) {
        LOGE("[NNBackend] GetBackendStatus failed, m_device is nullptr");
        return OH_NN_FAILED;
    }

    OH_NN_ReturnCode ret = m_device->GetDeviceStatus(status);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNBackend] GetBackendStatus failed, fail to get device status");
        return ret;
    }
    return OH_NN_SUCCESS;
}

Compiler* NNBackend::CreateCompiler(Compilation* compilation)
{
    if (compilation == nullptr) {
        LOGE("[NNBackend] CreateCompiler failed, compilation is nullptr");
        return nullptr;
    }

    // 仅支持从nnmodel 和 nnmodel-cache构建编译器
    if ((compilation->offlineModelPath != nullptr) ||
        ((compilation->offlineModelBuffer.first != nullptr) ||
         (compilation->offlineModelBuffer.second != static_cast<size_t>(0)))) {
        LOGE("[NNBackend] CreateCompiler failed, only support build NN model and NN model cache.");
        return nullptr;
    }

    // 如果nnmodel是空值，构建空的编译器，后续从cache编译模型,
    // 如果nnmodel不为空，则从对应模型构建编译器
    NNCompiler* nnCompiler = nullptr;
    if (compilation->nnModel == nullptr) {
        nnCompiler = new (std::nothrow) NNCompiler(m_device, m_backendID);
    } else {
        nnCompiler = new (std::nothrow) NNCompiler(compilation->nnModel, m_device, m_backendID);
    }

    if (nnCompiler == nullptr) {
        LOGE("[NNBackend] CreateCompiler failed, error happend when allocating NN Compiler.");
        return nullptr;
    }

    return reinterpret_cast<Compiler*>(nnCompiler);
}

OH_NN_ReturnCode NNBackend::DestroyCompiler(Compiler* compiler)
{
    if (compiler == nullptr) {
        LOGE("[NNBackend] DestroyCompiler failed, compiler is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    delete compiler;
    return OH_NN_SUCCESS;
}

Executor* NNBackend::CreateExecutor(Compilation* compilation)
{
    if (compilation == nullptr) {
        LOGE("[NNBackend] CreateExecutor failed, compilation is nullptr.");
        return nullptr;
    }

    if (compilation->compiler == nullptr) {
        LOGE("[NNBackend] CreateExecutor failed, the compiler in compilation is nullptr, create complier first.");
        return nullptr;
    }

    NNCompiler* nnCompiler = reinterpret_cast<NNCompiler*>(compilation->compiler);
    NNExecutor* nnExecutor = nnCompiler->CreateExecutor();
    if (nnExecutor == nullptr) {
        LOGE("[NNBackend] CreateExecutor failed, fail to create NN Executor.");
        return nullptr;
    }

    return reinterpret_cast<Executor*>(nnExecutor);
}

OH_NN_ReturnCode NNBackend::DestroyExecutor(Executor* executor)
{
    if (executor == nullptr) {
        LOGE("[NNBackend] DestroyExecutor failed, executor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    delete executor;
    return OH_NN_SUCCESS;
}

Tensor* NNBackend::CreateTensor(TensorDesc* desc)
{
    if (desc == nullptr) {
        LOGE("[NNBackend] CreateTensor failed, tensor desc is nullptr.");
        return nullptr;
    }

    NNTensor2_0* tensorImpl = new (std::nothrow) NNTensor2_0(m_backendID);
    if (tensorImpl == nullptr) {
        LOGE("[NNBackend] CreateTensor failed,  error happend when allocating NN Tensor.");
        return nullptr;
    }

    auto ret = tensorImpl->SetTensorDesc(desc);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNBackend] CreateTensor failed,  error happend when setting tensor desc.");
        delete tensorImpl;
        return nullptr;
    }

    return reinterpret_cast<Tensor*>(tensorImpl);
}

OH_NN_ReturnCode NNBackend::DestroyTensor(Tensor* tensor)
{
    if (tensor == nullptr) {
        LOGE("[NNBackend] DestroyTensor failed, tensor is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    delete tensor;
    return OH_NN_SUCCESS;
}

std::shared_ptr<Device> NNBackend::GetDevice() const
{
    if (m_device == nullptr) {
        LOGE("[NNBackend] GetDevice failed, m_device is nullptr.");
    }
    return m_device;
}

OH_NN_ReturnCode NNBackend::GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                                  std::vector<bool>& ops)
{
    if (model == nullptr) {
        LOGE("[NNBackend] GetSupportedOperation failed, model is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (m_device == nullptr) {
        LOGE("[NNBackend] GetSupportedOperation failed, device is nullptr, some error happend.");
        return OH_NN_FAILED;
    }

    OH_NN_ReturnCode ret = m_device->GetSupportedOperation(model, ops);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNBackend] GetSupportedOperation failed, fail to get supported ops from device.");
        return OH_NN_FAILED;
    }

    return OH_NN_SUCCESS;
}
} // NeuralNetworkRuntime
} // OHOS