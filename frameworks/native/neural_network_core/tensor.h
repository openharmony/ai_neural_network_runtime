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

#ifndef NEURAL_NETWORK_RUNTIME_TENSOR_H
#define NEURAL_NETWORK_RUNTIME_TENSOR_H

#include <memory>
#include "tensor_desc.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class Tensor {
public:
    Tensor() = default;
    Tensor(size_t backendID);
    virtual ~Tensor() = default;

    virtual OH_NN_ReturnCode SetTensorDesc(const TensorDesc* tensorDesc) = 0;
    virtual OH_NN_ReturnCode CreateData() = 0;
    virtual OH_NN_ReturnCode CreateData(size_t size) = 0;
    virtual OH_NN_ReturnCode CreateData(int fd, size_t size, size_t offset) = 0;

    virtual TensorDesc* GetTensorDesc() const = 0;
    virtual void* GetData() const = 0;
    virtual int GetFd() const = 0;
    virtual size_t GetSize() const = 0;
    virtual size_t GetOffset() const = 0;
    virtual size_t GetBackendID() const = 0;
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif  // NEURAL_NETWORK_RUNTIME_TENSOR_H