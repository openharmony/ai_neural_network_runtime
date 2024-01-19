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

#include "backend_manager.h"

#include <algorithm>
#include "cpp_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
BackendManager::~BackendManager()
{
    m_backends.clear();
    m_backendIDs.clear();
    m_tmpBackendIds.clear();
}

const std::vector<size_t>& BackendManager::GetAllBackendsID()
{
    const std::lock_guard<std::mutex> lock(m_mtx);
    m_tmpBackendIds.clear();
    std::shared_ptr<Backend> backend {nullptr};
    for (auto iter = m_backends.begin(); iter != m_backends.end(); ++iter) {
        backend = iter->second;
        if (!IsValidBackend(backend)) {
            continue;
        }
        m_tmpBackendIds.emplace_back(iter->first);
    }
    return m_tmpBackendIds;
}

std::shared_ptr<Backend> BackendManager::GetBackend(size_t backendID) const
{
    if (m_backends.empty()) {
        LOGE("[BackendManager] GetBackend failed, there is no registered backend can be used.");
        return nullptr;
    }

    auto iter = m_backends.begin();
    if (backendID == static_cast<size_t>(0)) {
        LOGI("[BackendManager] the backendID is 0, default return 1st backend.");
        return iter->second;
    }

    iter = m_backends.find(backendID);
    if (iter == m_backends.end()) {
        LOGE("[BackendManager] GetBackend failed, not find backendId=%{public}zu", backendID);
        return nullptr;
    }

    return iter->second;
}

const std::string& BackendManager::GetBackendName(size_t backendID)
{
    const std::lock_guard<std::mutex> lock(m_mtx);
    m_tmpBackendName.clear();
    if (m_backends.empty()) {
        LOGE("[BackendManager] GetBackendName failed, there is no registered backend can be used.");
        return m_tmpBackendName;
    }

    auto iter = m_backends.begin();
    if (backendID == static_cast<size_t>(0)) {
        LOGI("[BackendManager] the backendID is 0, default return 1st backend.");
    } else {
        iter = m_backends.find(backendID);
    }

    if (iter == m_backends.end()) {
        LOGE("[BackendManager] GetBackendName failed, backendID %{public}zu is not registered.", backendID);
        return m_tmpBackendName;
    }

    OH_NN_ReturnCode ret = iter->second->GetBackendName(m_tmpBackendName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BackendManager] GetBackendName failed, fail to get backendName from backend.");
    }

    return m_tmpBackendName;
}

OH_NN_ReturnCode BackendManager::RegisterBackend(std::function<std::shared_ptr<Backend>()> creator)
{
    auto regBackend = creator();
    if (regBackend == nullptr) {
        LOGE("[BackendManager] RegisterBackend failed, fail to create backend.");
        return OH_NN_FAILED;
    }

    if (!IsValidBackend(regBackend)) {
        LOGE("[BackendManager] RegisterBackend failed, backend is not available.");
        return OH_NN_UNAVAILABLE_DEVICE;
    }

    size_t backendID = regBackend->GetBackendID();

    const std::lock_guard<std::mutex> lock(m_mtx);
    auto setResult = m_backendIDs.emplace(backendID);
    if (!setResult.second) {
        LOGE("[BackendManager] RegisterBackend failed, backend already exists, cannot register again. "
             "backendID=%{public}zu", backendID);
        return OH_NN_FAILED;
    }

    m_backends.emplace(backendID, regBackend);
    return OH_NN_SUCCESS;
}

bool BackendManager::IsValidBackend(std::shared_ptr<Backend> backend) const
{
    DeviceStatus status = UNKNOWN;

    OH_NN_ReturnCode ret = backend->GetBackendStatus(status);
    if (ret != OH_NN_SUCCESS || status == UNKNOWN || status == OFFLINE) {
        return false;
    }

    return true;
}
} // NeuralNetworkCore
} // OHOS
