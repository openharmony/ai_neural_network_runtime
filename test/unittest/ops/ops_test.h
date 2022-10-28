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

#ifndef NEURAL_NETWORK_RUNTIME_OPS_TEST_H
#define NEURAL_NETWORK_RUNTIME_OPS_TEST_H

#include <memory>

#include "mindir.h"

#include "frameworks/native/nn_tensor.h"
#include "test/unittest/common/base_test.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class OpsTest : public BaseTest {
public:
    OpsTest() = default;
    virtual void SaveInputTensor(const std::vector<uint32_t>& inputsIndex, OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam);
    virtual void SaveOutputTensor(const std::vector<uint32_t>& outputsIndex, OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam);
    virtual void InitTensor(const std::vector<uint32_t>& inputsIndex, const std::vector<uint32_t>& outputsIndex) {};

    void SetKernelSize(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetStride(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetActivation(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetDilation(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SetGroup(OH_NN_DataType dataType,
        const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

public:
    std::vector<uint32_t> m_inputsIndex {};
    std::vector<uint32_t> m_outputsIndex {};
    std::vector<uint32_t> m_paramsIndex {};
    std::vector<std::shared_ptr<NNTensor>> m_allTensors;
};
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_OPS_TEST_H
