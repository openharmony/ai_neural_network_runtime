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

#ifndef NEURAL_NETWORK_RUNTIME_NNTENSOR_H
#define NEURAL_NETWORK_RUNTIME_NNTENSOR_H

#include <memory>
#include "tensor.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class NNTensor2_0 : public Tensor {
public:
    explicit NNTensor2_0(size_t backendID) : m_backendID(backendID) {}
    ~NNTensor2_0() override;

    OH_NN_ReturnCode SetTensorDesc(const TensorDesc* tensorDesc) override;
    OH_NN_ReturnCode CreateData() override;
    OH_NN_ReturnCode CreateData(size_t size) override;
    OH_NN_ReturnCode CreateData(int fd, size_t size, size_t offset) override;

    TensorDesc* GetTensorDesc() const override;
    void* GetData() const override;
    int GetFd() const override;
    size_t GetSize() const override;
    size_t GetOffset() const override;
    size_t GetBackendID() const override;

    bool CheckTensorData() const;

    OH_NN_ReturnCode CheckDimRanges(const std::vector<uint32_t>& minDimRanges,
                                    const std::vector<uint32_t>& maxDimRanges) const;

private:
    OH_NN_ReturnCode AllocateMemory(size_t length);
    OH_NN_ReturnCode ReleaseMemory();

private:
    size_t m_backendID {0};
    TensorDesc* m_tensorDesc {nullptr};
    void* m_data {nullptr};
    int m_fd {0};
    size_t m_size {0};
    size_t m_offset {0};
    bool m_isUserData {false};
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif  // NEURAL_NETWORK_RUNTIME_NNTENSOR_H