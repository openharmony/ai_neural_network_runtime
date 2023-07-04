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
#include "frameworks/native/nn_tensor.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Validation;

namespace NNRT {
namespace UnitTest {
class NnValidationTest : public testing::Test {
};

/**
 * @tc.name: nn_validation_validate_tensor_datatype_001
 * @tc.desc: Verify the success of the validate_tensor_datatype function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_tensor_datatype_001, TestSize.Level1)
{
    int dataTypeTest = 12;
    OH_NN_DataType dataType = (OH_NN_DataType)dataTypeTest;
    EXPECT_EQ(true, ValidateTensorDataType(dataType));
}

/**
 * @tc.name: nn_validation_validate_tensor_datatype_002
 * @tc.desc: Verify the gt bounds of the validate_tensor_datatype function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_tensor_datatype_002, TestSize.Level1)
{
    int dataTypeTest = 13;
    OH_NN_DataType dataType = (OH_NN_DataType)dataTypeTest;
    EXPECT_EQ(false, ValidateTensorDataType(dataType));
}

/**
 * @tc.name: nn_validation_validate_tensor_datatype_003
 * @tc.desc: Verify the lt bounds of the validate_tensor_datatype function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_tensor_datatype_003, TestSize.Level1)
{
    int dataTypeTest = -1;
    OH_NN_DataType dataType = (OH_NN_DataType)dataTypeTest;
    EXPECT_EQ(false, ValidateTensorDataType(dataType));
}

/**
 * @tc.name: nn_validation_validate_preformance_mode_001
 * @tc.desc: Verify the success of the validate_preformance_mode function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_preformance_mode_001, TestSize.Level1)
{
    int performanceModeTest = 4;
    OH_NN_PerformanceMode performanceMode = (OH_NN_PerformanceMode)performanceModeTest;
    EXPECT_EQ(true, ValidatePerformanceMode(performanceMode));
}

/**
 * @tc.name: nn_validation_validate_preformance_mode_002
 * @tc.desc: Verify the gt bounds of the validate_preformance_mode function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_preformance_mode_002, TestSize.Level1)
{
    int performanceModeTest = 5;
    OH_NN_PerformanceMode performanceMode = (OH_NN_PerformanceMode)performanceModeTest;
    EXPECT_EQ(false, ValidatePerformanceMode(performanceMode));
}

/**
 * @tc.name: nn_validation_validate_preformance_mode_003
 * @tc.desc: Verify the lt bounds of the validate_preformance_mode function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_preformance_mode_003, TestSize.Level1)
{
    int performanceModeTest = -1;
    OH_NN_PerformanceMode performanceMode = (OH_NN_PerformanceMode)performanceModeTest;
    EXPECT_EQ(false, ValidatePerformanceMode(performanceMode));
}

/**
 * @tc.name: nn_validation_validate_priority_001
 * @tc.desc: Verify the success of the validate_priority function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_priority_001, TestSize.Level1)
{
    int priorityTest = 2;
    OH_NN_Priority priority = (OH_NN_Priority)priorityTest;
    EXPECT_EQ(true, ValidatePriority(priority));
}

/**
 * @tc.name: nn_validation_validate_priority_002
 * @tc.desc: Verify the gt bounds of the validate_priority function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_priority_002, TestSize.Level1)
{
    int priorityTest = 4;
    OH_NN_Priority priority = (OH_NN_Priority)priorityTest;
    EXPECT_EQ(false, ValidatePriority(priority));
}

/**
 * @tc.name: nn_validation_validate_priority_003
 * @tc.desc: Verify the lt bounds of the validate_priority function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_validate_priority_003, TestSize.Level1)
{
    int priorityTest = -1;
    OH_NN_Priority priority = (OH_NN_Priority)priorityTest;
    EXPECT_EQ(false, ValidatePriority(priority));
}

/**
 * @tc.name: nn_validation_fusetype_001
 * @tc.desc: Verify the success of the validate_fusetype function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_fusetype_001, TestSize.Level1)
{
    int fuseTypeTest = 2;
    OH_NN_FuseType fuseType = (OH_NN_FuseType)fuseTypeTest;
    EXPECT_EQ(true, ValidateFuseType(fuseType));
}

/**
 * @tc.name: nn_validation_fusetype_002
 * @tc.desc: Verify the gt bounds of the validate_fusetype function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_fusetype_002, TestSize.Level1)
{
    int fuseTypeTest = 3;
    OH_NN_FuseType fuseType = (OH_NN_FuseType)fuseTypeTest;
    EXPECT_EQ(false, ValidateFuseType(fuseType));
}

/**
 * @tc.name: nn_validation_fusetype_003
 * @tc.desc: Verify the lt bounds of the validate_fusetype function
 * @tc.type: FUNC
 */
HWTEST_F(NnValidationTest, nn_validation_fusetype_003, TestSize.Level1)
{
    int fuseTypeTest = -1;
    OH_NN_FuseType fuseType = (OH_NN_FuseType)fuseTypeTest;
    EXPECT_EQ(false, ValidateFuseType(fuseType));
}
} // namespace UnitTest
} // namespace NNRT
