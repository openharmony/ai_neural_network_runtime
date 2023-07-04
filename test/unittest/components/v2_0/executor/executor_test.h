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

#ifndef NEURAL_NETWORK_RUNTIME_EXECUTOR_UNITTEST_H
#define NEURAL_NETWORK_RUNTIME_EXECUTOR_UNITTEST_H

#include <gtest/gtest.h>

#include "mindir.h"

#include "frameworks/native/executor.h"

namespace MSLITE = mindspore::lite;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class ExecutorTest : public testing::Test {
public:
    MSLITE::LiteGraph* BuildLiteGraph(const std::vector<int32_t> dim, const std::vector<int32_t> dimOut);
    OH_NN_Tensor SetTensor(OH_NN_DataType dataType, uint32_t dimensionCount, const int32_t *dimensions,
        const OH_NN_QuantParam *quantParam, OH_NN_TensorType type);
    void SetMermory(OH_NN_Memory** &memory);

public:
    uint32_t m_index {0};
    const std::vector<int32_t> m_dim {3, 3};
    const std::vector<int32_t> m_dimOut {3, 3};
    const int32_t m_dimArry[2] {3, 3};
    uint32_t m_dimensionCount {2};
    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
};
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_EXECUTOR_UNITTEST_H