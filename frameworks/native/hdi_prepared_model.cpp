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

#include "hdi_prepared_model.h"

#include "common/log.h"
#include "memory_manager.h"
#include "transform.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
HDIPreparedModel::HDIPreparedModel(OHOS::sptr<V1_0::IPreparedModel> hdiPreparedModel)
    : m_hdiPreparedModel(hdiPreparedModel)
{
    hdiPreparedModel->GetVersion(m_hdiVersion.first, m_hdiVersion.second);
}

OH_NN_ReturnCode HDIPreparedModel::ExportModelCache(std::vector<ModelBuffer>& modelCache)
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

OH_NN_ReturnCode HDIPreparedModel::Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
    std::vector<std::vector<int32_t>>& outputsDims, std::vector<bool>& isOutputBufferEnough)
{
    V1_0::IOTensor iTensor;
    std::vector<V1_0::IOTensor> iInputTensors;
    for (auto& input: inputs) {
        iTensor = NNToHDI::TransIOTensor(input);
        if (iTensor.data.fd == INVALID_FD) {
            LOGE("Transform inputs tensor failed, cannot find data file descriptor.");
            return OH_NN_INVALID_PARAMETER;
        }
        iInputTensors.emplace_back(iTensor);
    }

    std::vector<V1_0::IOTensor> iOutputTensors;
    for (auto& output: outputs) {
        iTensor = NNToHDI::TransIOTensor(output);
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