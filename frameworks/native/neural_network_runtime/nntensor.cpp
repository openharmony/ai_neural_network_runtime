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

#include <sys/mman.h>
#include <unistd.h>

#include "common/log.h"
#include "backend_manager.h"
#include "nnbackend.h"
#include "nntensor.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
NNTensor2_0::~NNTensor2_0()
{
    ReleaseMemory();

    delete m_tensorDesc;
    m_tensorDesc = nullptr;

    m_data = nullptr;
    m_fd = 0;
    m_offset = 0;
    m_size = 0;
    m_isUserData = false;
}

OH_NN_ReturnCode NNTensor2_0::SetTensorDesc(const TensorDesc* tensorDesc)
{
    if (m_tensorDesc != nullptr) {
        delete m_tensorDesc;
        m_tensorDesc = nullptr;
    }
    m_tensorDesc = new (std::nothrow) TensorDesc();
    if (m_tensorDesc == nullptr) {
        LOGE("[NNTensor2_0] SetTensorDesc failed, failed to create desc for tensor.");
        return OH_NN_NULL_PTR;
    }

    // Copy the member attributes to new tensor description
    *m_tensorDesc = *tensorDesc;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNTensor2_0::CreateData()
{
    if (m_data != nullptr) {
        LOGE("NNTensor2_0::CreateData failed, m_data has been created before.");
        return OH_NN_FAILED;
    }
    if (m_tensorDesc == nullptr) {
        LOGE("NNTensor2_0::CreateData failed, m_tensorDesc is nullptr.");
        return OH_NN_NULL_PTR;
    }

    size_t byteSize = 0;
    auto ret = m_tensorDesc->GetByteSize(&byteSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::CreateData failed, failed to get byte size from tensorDesc.");
        return ret;
    }
    if (byteSize > ALLOCATE_BUFFER_LIMIT) {
        LOGE("NNTensor2_0::CreateData failed, Invalid buffer size, "
             "it must greater than 0 and less than 1Gb. length=%{public}zu", byteSize);
        return OH_NN_INVALID_PARAMETER;
    }

    ret = AllocateMemory(byteSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::CreateData failed, failed to allocate memory.");
        return ret;
    }
    m_isUserData = false;
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode NNTensor2_0::CreateData(size_t size)
{
    if (m_data != nullptr) {
        LOGE("NNTensor2_0::CreateData failed, m_data has been created before.");
        return OH_NN_FAILED;
    }
    if (m_tensorDesc == nullptr) {
        LOGE("NNTensor2_0::CreateData failed, m_tensorDesc is nullptr.");
        return OH_NN_NULL_PTR;
    }
    if (size > ALLOCATE_BUFFER_LIMIT) {
        LOGE("NNTensor2_0::CreateData failed, Invalid buffer size, "
             "it must greater than 0 and less than 1Gb. length=%{public}zu", size);
        return OH_NN_INVALID_PARAMETER;
    }
    size_t byteSize = 0;
    auto ret = m_tensorDesc->GetByteSize(&byteSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::CreateData failed, failed to get byte size from tensorDesc.");
        return ret;
    }
    if (size < byteSize) {
        LOGE("NNTensor2_0::CreateData failed, size:%{public}zu must be larger than "
             "or equal to byte size:%{public}zu.", size, byteSize);
        return OH_NN_INVALID_PARAMETER;
    }

    ret = AllocateMemory(size);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::CreateData failed, failed to allocate memory.");
        return ret;
    }
    m_isUserData = false;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNTensor2_0::CreateData(int fd, size_t size, size_t offset)
{
    if (m_data != nullptr) {
        LOGE("NNTensor2_0::CreateData failed, m_data has been created before.");
        return OH_NN_FAILED;
    }
    if (m_tensorDesc == nullptr) {
        LOGE("NNTensor2_0::CreateData failed, m_tensorDesc is nullptr.");
        return OH_NN_NULL_PTR;
    }

    size_t byteSize = 0;
    auto ret = m_tensorDesc->GetByteSize(&byteSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::CreateData failed, failed to get byte size from tensorDesc.");
        return ret;
    }
    if (fd < 0) {
        LOGE("NNTensor2_0::CreateData failed, fd is less than 0.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (size == 0) {
        LOGE("NNTensor2_0::CreateData failed, size is zero.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (size < offset) {
        LOGE("NNTensor2_0::CreateData failed, size is smaller than offset.");
        return OH_NN_INVALID_PARAMETER;
    }
    if ((size - offset) < byteSize) {
        LOGE("NNTensor2_0::CreateData failed, size of fd is insufficient.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_data = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, offset);
    if (m_data == MAP_FAILED) {
        LOGE("NNTensor2_0::AllocateMemory failed, Map fd to address failed: %{public}s.", strerror(errno));
        m_data = nullptr;
        return OH_NN_MEMORY_ERROR;
    }

    m_fd = fd;
    m_size = size;
    m_offset = offset;
    m_isUserData = true;
    return OH_NN_SUCCESS;
}

TensorDesc* NNTensor2_0::GetTensorDesc() const
{
    return m_tensorDesc;
}

void* NNTensor2_0::GetData() const
{
    return m_data;
}

int NNTensor2_0::GetFd() const
{
    return m_fd;
}

size_t NNTensor2_0::GetSize() const
{
    return m_size;
}

size_t NNTensor2_0::GetOffset() const
{
    return m_offset;
}

OH_NN_ReturnCode NNTensor2_0::AllocateMemory(size_t length)
{
    BackendManager& backendManager = BackendManager::GetInstance();
    std::shared_ptr<Backend> backend = backendManager.GetBackend(m_backendID);
    if (backend == nullptr) {
        LOGE("NNTensor2_0::AllocateMemory failed, failed to get backend of %{public}zu.", m_backendID);
        return OH_NN_NULL_PTR;
    }

    auto* nnBackend = reinterpret_cast<NNBackend*>(backend.get());
    auto device = nnBackend->GetDevice();
    if (device == nullptr) {
        LOGE("NNTensor2_0::AllocateMemory failed, device of nnbackend is nullptr.");
        return OH_NN_NULL_PTR;
    }
    int fd = 0;
    auto oldRet = device->AllocateBuffer(length, fd);
    if (oldRet != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::AllocateMemory failed, failed to allocate buffer.");
        return OH_NN_MEMORY_ERROR;
    }
    if (fd < 0) {
        LOGE("NNTensor2_0::AllocateMemory failed, fd must greater than 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    m_data = mmap(nullptr, length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (m_data == MAP_FAILED) {
        LOGE("NNTensor2_0::AllocateMemory failed, Map fd to address failed: %{public}s.", strerror(errno));
        m_data = nullptr;
        return OH_NN_MEMORY_ERROR;
    }
    m_fd = fd;
    m_offset = 0;
    m_size = length;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNTensor2_0::ReleaseMemory()
{
    if (m_size == 0 || m_data == nullptr) {
        return OH_NN_SUCCESS;
    }
    if (m_fd < 0) {
        LOGE("NNTensor2_0::ReleaseMemory failed, m_fd must greater than 0.");
        return OH_NN_INVALID_PARAMETER;
    }

    auto unmapResult = munmap(m_data, m_size);
    if (unmapResult != 0) {
        LOGE("NNTensor2_0::ReleaseMemory failed. Please try again.");
        return OH_NN_MEMORY_ERROR;
        }

    if (!m_isUserData) {
        if (close(m_fd) != 0) {
            LOGE("NNTensor2_0::ReleaseMemory failed. fd=%{public}d", m_fd);
            return OH_NN_MEMORY_ERROR;
        }
        BackendManager& backendManager = BackendManager::GetInstance();
        std::shared_ptr<Backend> backend = backendManager.GetBackend(m_backendID);
        if (backend == nullptr) {
            LOGE("NNTensor2_0::ReleaseMemory failed, failed to get backend of %{public}zu.", m_backendID);
            return OH_NN_NULL_PTR;
        }

        auto* nnrtBackend = reinterpret_cast<NNBackend*>(backend.get());
        auto device = nnrtBackend->GetDevice();
        if (device == nullptr) {
            LOGE("");
            return OH_NN_NULL_PTR;
        }
        auto oldRet = device->ReleaseBuffer(m_fd, m_size);
        if (oldRet != OH_NN_SUCCESS) {
            LOGE("NNTensor2_0::ReleaseMemory failed, failed to release buffer.");
            return OH_NN_MEMORY_ERROR;
        }
    }

    m_data = nullptr;
    m_size = 0;
    m_fd = 0;

    return OH_NN_SUCCESS;
}

size_t NNTensor2_0::GetBackendID() const
{
    return m_backendID;
}

bool NNTensor2_0::CheckTensorData() const
{
    if (m_tensorDesc == nullptr) {
        LOGE("NNTensor2_0::CheckTensorData failed, m_tensorDesc is nullptr.");
        return false;
    }

    size_t byteSize = 0;
    auto ret = m_tensorDesc->GetByteSize(&byteSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::CheckTensorData failed, failed to get byte size from tensorDesc.");
        return false;
    }
    if ((m_size - m_offset) < byteSize) {
        LOGE("NNTensor2_0::CheckTensorData failed, m_size is less than byte size.");
        return false;
    }

    if (m_data == nullptr) {
        LOGE("NNTensor2_0::CheckTensorData failed, m_data is nullptr.");
        return false;
    }

    if (m_fd < 0) {
        LOGE("NNTensor2_0::CheckTensorData failed, m_fd is less than zero.");
        return false;
    }

    return true;
}

OH_NN_ReturnCode NNTensor2_0::CheckDimRanges(
    const std::vector<uint32_t>& minDimRanges, const std::vector<uint32_t>& maxDimRanges) const
{
    if (m_tensorDesc == nullptr) {
        LOGE("NNTensor2_0::CheckInputDimRanges failed, m_tensorDesc is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    int32_t* shape = nullptr;
    size_t shapeSize = 0;
    auto ret = m_tensorDesc->GetShape(&shape, &shapeSize);
    if (ret != OH_NN_SUCCESS) {
        LOGE("NNTensor2_0::CheckInputDimRanges failed, failed to get shape from desc.");
        return ret;
    }
    for (size_t j = 0; j < shapeSize; ++j) {
        // Dimensions cannot be negative
        if (shape[j] < 0) {
            LOGE("Dimension %{public}zu is %{public}d.", j, shape[j]);
            return OH_NN_INVALID_PARAMETER;
        }
        uint32_t dim = static_cast<uint32_t>(shape[j]);
        if (dim < minDimRanges[j] || dim > maxDimRanges[j]) {
            LOGE("Dimension %{public}zu is %{public}u, which is out of range "
                "[%{public}u, %{public}u]", j, dim, minDimRanges[j], maxDimRanges[j]);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    return OH_NN_SUCCESS;
}
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS