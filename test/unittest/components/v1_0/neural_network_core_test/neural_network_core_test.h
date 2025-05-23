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

#ifndef NEURAL_NETWORK_RUNTIME_UNITTEST_H
#define NEURAL_NETWORK_RUNTIME_UNITTEST_H

#include <gtest/gtest.h>

#include "neural_network_runtime/neural_network_runtime.h"
#include "inner_model.h"
#include "executor.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Unittest {
class NeuralNetworkCoreTest : public testing::Test {
public:
    OH_NN_ReturnCode BuildModel(InnerModel& model);
    void InitIndices();
    void AddModelTensor(InnerModel& innerModel);
    void SetInnerBuild(InnerModel& innerModel);
    void SetExecutor(Executor& executor);
    void SetInputAndOutput(Executor& executor);
    void SetTensor();

public:
    OH_NN_UInt32Array m_inputIndices;
    OH_NN_UInt32Array m_outputIndices;
    OH_NN_UInt32Array m_paramIndices;
    OH_NN_Tensor m_tensor;
    TensorDesc desc;
    TensorDesc* tensorDesc = &desc;
    Compilation ation;
    Compilation* compilation = &ation;

    size_t backendID = 1;
    size_t sizetensor = 3;
    uint32_t m_inputIndexs[2] {0, 1};
    uint32_t m_outputIndexs[1] {2};
    uint32_t m_paramIndexs[1] {3};
};
} // namespace Unittest
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_UNITTEST_H
