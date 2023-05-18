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

#ifndef OHOS_HDI_NNR_V1_0_UTILS_H
#define OHOS_HDI_NNR_V1_0_UTILS_H

#include "shared_buffer_parser.h"

#include <hdf_base.h>
#include "ashmem.h"
#include "v1_0/nnrt_types.h"
#include "utils/hdf_log.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
SharedBufferParser::~SharedBufferParser()
{
    if (m_ashptr != nullptr) {
        m_ashptr->UnmapAshmem();
        m_ashptr->CloseAshmem();
        m_bufferAddr = nullptr;
    }
}

int32_t SharedBufferParser::Init(const std::string& name, int32_t size)
{
    HDF_LOGI("Init SharedBufferParser from name and size.");
    sptr<Ashmem> ashptr = Ashmem::CreateAshmem(name.c_str(), size);
    if (ashptr == nullptr) {
        HDF_LOGE("Create ashmen from size failed.");
        return HDF_FAILURE;
    }

    SharedBuffer buffer;
    buffer.fd = ashptr->GetAshmemFd();
    buffer.bufferSize = ashptr->GetAshmemSize();
    buffer.offset = 0;
    buffer.dataSize = size;

    auto ret = Init(buffer);
    if (ret != HDF_SUCCESS) {
        HDF_LOGE("Init SharedBufferParser failed.");
        return ret;
    }
    return HDF_SUCCESS;
}

int32_t SharedBufferParser::Init(const SharedBuffer& buffer)
{
    if (buffer.fd == INVALID_FD) {
        HDF_LOGE("Invalid buffer fd, it cannot be %{public}d.", INVALID_FD);
        return HDF_ERR_INVALID_PARAM;
    }

    m_ashptr = new (std::nothrow) Ashmem(buffer.fd, buffer.bufferSize);
    if (m_ashptr == nullptr) {
        HDF_LOGE("Create ashmem failed.");
        return HDF_FAILURE;
    }

    if (!m_ashptr->MapReadAndWriteAshmem()) {
        HDF_LOGE("Map buffer fd to address failed.");
        return HDF_FAILURE;
    }

    auto bufferAddr = m_ashptr->ReadFromAshmem(buffer.dataSize, buffer.offset);
    if (bufferAddr == nullptr) {
        HDF_LOGE("Invalid dataSize or offset of SharedBuffer.");
        return HDF_ERR_INVALID_PARAM;
    }
    m_bufferAddr = const_cast<void*>(bufferAddr);

    m_buffer = buffer;
    return HDF_SUCCESS;
}

void* SharedBufferParser::GetBufferPtr()
{
    return m_bufferAddr;
}

SharedBuffer SharedBufferParser::GetBuffer()
{
    return m_buffer;
}
} // V1_0
} // Nnrt
} // HDI
} // OHOS
#endif // OHOS_HDI_NNR_V1_0_UTILS_H