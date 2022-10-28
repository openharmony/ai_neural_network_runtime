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

#include "base_test.h"

using namespace OHOS::NeuralNetworkRuntime::Ops;
using namespace std;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
void BaseTest::SetUp() {}

void BaseTest::TearDown() {}

std::shared_ptr<OHOS::NeuralNetworkRuntime::NNTensor> BaseTest::TransToNNTensor(
    OH_NN_DataType dataType, const std::vector<int32_t>& dim, const OH_NN_QuantParam* quantParam,
    OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> nnTensor = std::make_shared<NNTensor>();
    OH_NN_Tensor tensor;
    tensor.dataType = dataType;
    tensor.dimensionCount = dim.size();
    tensor.dimensions = (dim.empty() ? nullptr : dim.data());
    tensor.quantParam = quantParam;
    tensor.type = type;
    nnTensor->BuildFromOHNNTensor(tensor);
    return nnTensor;
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
