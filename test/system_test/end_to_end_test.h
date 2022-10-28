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

#ifndef SYSTEM_TEST_END_TO_END_TEST
#define SYSTEM_TEST_END_TO_END_TEST

#include <cmath>
#include <cstdio>
#include <vector>

#include "interfaces/kits/c/neural_network_runtime.h"
#include "test/system_test/common/nnrt_test.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace SystemTest {
class End2EndTest : public NNRtTest {
public:
    End2EndTest() = default;

    OH_NN_ReturnCode BuildModel(const std::vector<CppTensor>& tensors);
    OH_NN_ReturnCode IsExpectedOutput(const float* outputBuffer);
    OH_NN_ReturnCode IsExpectedOutput(const OH_NN_Memory* outputMemory);
};
} // namespace SystemTest
} // NeuralNetworkRuntime
} // OHOS

#endif // SYSTEM_TEST_END_TO_END_TEST