/*
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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

#ifndef OHOS_HDI_NNRT_V2_0_NNRTPREPAREMODELIMPL_H
#define OHOS_HDI_NNRT_V2_0_NNRTPREPAREMODELIMPL_H
#include"../../common/log.h"

#include "v2_0/iprepared_model.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V2_0 {
class NnrtPrepareModelImpl : public IPreparedModel {
public:
    NnrtPrepareModelImpl() = default;
    virtual ~NnrtPrepareModelImpl() = default;

    int32_t ExportModelCache(std::vector<SharedBuffer>& modelCache)
    {
        LOGI("Export model cache.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t GetInputDimRanges(std::vector<std::vector<uint32_t>>& minInputDims,
         std::vector<std::vector<uint32_t>>& maxInputDims)
    {
        LOGI("Get input dim ranges.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
         std::vector<std::vector<int32_t>>& outputDims)
    {
        LOGI("Run prepare model.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }

    int32_t GetVersion(uint32_t& majorVer, uint32_t& minorVer)
    {
        LOGI("Get version.");
        return NNRT_ReturnCode::NNRT_FAILED;
    }
};
} // V2_0
} // Nnrt
} // HDI
} // OHOS

#endif // OHOS_HDI_NNRT_V2_0_NNRTPREPAREMODELIMPL_H