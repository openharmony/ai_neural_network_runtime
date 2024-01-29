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

#include "backend_registrar.h"

#include "common/log.h"
#include "backend_manager.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
BackendRegistrar::BackendRegistrar(const CreateBackend creator)
{
    auto& backendManager = BackendManager::GetInstance();
    OH_NN_ReturnCode ret = backendManager.RegisterBackend(creator);
    if (ret != OH_NN_SUCCESS) {
        LOGW("[BackendRegistrar] Register backend failed. ErrorCode=%{public}d", ret);
    }
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS