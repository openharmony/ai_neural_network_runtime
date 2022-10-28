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

#ifndef NEURAL_NETWORK_RUNTIME_BASE_TEST_H
#define NEURAL_NETWORK_RUNTIME_BASE_TEST_H

#include <gtest/gtest.h>
#include <memory>
#include "frameworks/native/ops_builder.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class BaseTest : public testing::Test {
public:
    virtual void SetUp();
    virtual void TearDown();
    virtual std::shared_ptr<OHOS::NeuralNetworkRuntime::NNTensor> TransToNNTensor(
        OH_NN_DataType dataType, const std::vector<int32_t> &dim, const OH_NN_QuantParam* quantParam,
        OH_NN_TensorType type);
};
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
#endif