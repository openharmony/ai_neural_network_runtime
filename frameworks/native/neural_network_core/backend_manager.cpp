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
    m_backendNames.clear();
    m_backendIDs.clear();
    m_backendIDGroup.clear();
}

BackendManager& BackendManager::GetInstance()
{
    // if libneural_network_runtime.so loaded
    if (dlopen("libneural_network_runtime.so", RTLD_NOLOAD) != nullptr) {
        // if libneural_network_runtime_ext.so not loaded, try to dlopen it
        if (dlopen("libneural_network_runtime_ext.so", RTLD_NOLOAD) == nullptr) {
            void* libHandle = dlopen("libneural_network_runtime_ext.so", RTLD_NOW | RTLD_GLOBAL);
            if (libHandle == nullptr) {
                LOGW("Failed to dlopen libneural_network_runtime_ext.so.");
            }
        }
    }

    static BackendManager instance;
    return instance;
}

const std::vector<size_t>& BackendManager::GetAllBackendsID()
{
    const std::lock_guard<std::mutex> lock(m_mtx);
    return m_backendIDs;
}

std::shared_ptr<Backend> BackendManager::GetBackend(size_t backendID)
{
    const std::lock_guard<std::mutex> lock(m_mtx);
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
    if (m_backendNames.empty()) {
        LOGE("[BackendManager] GetBackendName failed, there is no registered backend can be used.");
        return m_emptyBackendName;
    }

    auto iter = m_backendNames.begin();
    if (backendID == static_cast<size_t>(0)) {
        LOGI("[BackendManager] the backendID is 0, default return 1st backend.");
    } else {
        iter = m_backendNames.find(backendID);
    }

    if (iter == m_backendNames.end()) {
        LOGE("[BackendManager] GetBackendName failed, backendID %{public}zu is not registered.", backendID);
        return m_emptyBackendName;
    }

    return iter->second;
}

OH_NN_ReturnCode BackendManager::RegisterBackend(
    const std::string& backendName, std::function<std::shared_ptr<Backend>()> creator)
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
    auto iter = std::find(m_backendIDs.begin(), m_backendIDs.end(), backendID);
    if (iter != m_backendIDs.end()) {
        LOGE("[BackendManager] RegisterBackend failed, backend already exists, cannot register again. "
             "backendID=%{public}zu", backendID);
        return OH_NN_FAILED;
    }

    std::string tmpBackendName;
    auto ret = regBackend->GetBackendName(tmpBackendName);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BackendManager] RegisterBackend failed, fail to get backend name.");
        return OH_NN_FAILED;
    }
    m_backends.emplace(backendID, regBackend);
    m_backendIDs.emplace_back(backendID);
    m_backendNames.emplace(backendID, tmpBackendName);
    if (m_backendIDGroup.find(backendName) == m_backendIDGroup.end()) {
        std::vector<size_t> backendIDsTmp {backendID};
        m_backendIDGroup.emplace(backendName, backendIDsTmp);
    } else {
        m_backendIDGroup[backendName].emplace_back(backendID);
    }
    return OH_NN_SUCCESS;
}

void BackendManager::RemoveBackend(const std::string& backendName)
{
    const std::lock_guard<std::mutex> lock(m_mtx);
    if (m_backendIDGroup.find(backendName) == m_backendIDGroup.end()) {
        LOGI("[RemoveBackend] No need to remove backend for %{public}s.", backendName.c_str());
        return;
    }

    auto backendIDs = m_backendIDGroup[backendName];
    for (auto backendID : backendIDs) {
        if (m_backends.find(backendID) != m_backends.end()) {
            m_backends.erase(backendID);
        }
        auto iter = std::find(m_backendIDs.begin(), m_backendIDs.end(), backendID);
        if (iter != m_backendIDs.end()) {
            m_backendIDs.erase(iter);
        }
        if (m_backendNames.find(backendID) != m_backendNames.end()) {
            m_backendNames.erase(backendID);
        }
    }
    m_backendIDGroup.erase(backendName);
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
