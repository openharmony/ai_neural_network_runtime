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

#include <gtest/gtest.h>

#include "frameworks/native/validation.h"
#include "frameworks/native/ops_registry.h"
#include "frameworks/native/ops/add_builder.h"
#include "frameworks/native/ops/div_builder.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Validation;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace NNRT {
namespace UnitTest {
class OpsRegistryTest : public testing::Test {
};

/**
 * @tc.name: registry_001
 * @tc.desc: Verify the registry success the registar function
 * @tc.type: FUNC
 */
HWTEST_F(OpsRegistryTest, registry_001, TestSize.Level1)
{
    const int newRegistryOperationType = 100;
    REGISTER_OPS(AddBuilder, OH_NN_OperationType(newRegistryOperationType));

    OpsRegistry& opsregistry = OpsRegistry::GetSingleton();
    EXPECT_NE(nullptr, opsregistry.GetOpsBuilder(OH_NN_OperationType(newRegistryOperationType)));
}

/**
 * @tc.name: registry_002
 * @tc.desc: Verify the registry twice the registar function
 * @tc.type: FUNC
 */
HWTEST_F(OpsRegistryTest, registry_002, TestSize.Level1)
{
    const int newRegistryOperationType = 1000;
    REGISTER_OPS(AddBuilder, OH_NN_OperationType(newRegistryOperationType));

    OpsRegistry& opsregistry = OpsRegistry::GetSingleton();
    EXPECT_NE(nullptr, opsregistry.GetOpsBuilder(OH_NN_OperationType(newRegistryOperationType)));

    REGISTER_OPS(DivBuilder, OH_NN_OperationType(newRegistryOperationType));
}
} // namespace UnitTest
} // namespace NNRT
