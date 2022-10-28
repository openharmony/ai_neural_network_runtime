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

#ifndef NEURAL_NETWORK_RUNTIME_UTILS_H
#define NEURAL_NETWORK_RUNTIME_UTILS_H

#include <memory>

#include "log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
template<typename T, typename... Args>
std::shared_ptr<T> CreateSharedPtr(Args&&... args)
{
    std::shared_ptr<T> tPtr = nullptr;
    try {
        tPtr = std::make_shared<T>(args...);
    } catch (const std::bad_alloc& except) {
        LOGW("Create a new shared pointer failed. Error: %s", except.what());
        return nullptr;
    }
    return tPtr;
}

} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_UTILS_H
