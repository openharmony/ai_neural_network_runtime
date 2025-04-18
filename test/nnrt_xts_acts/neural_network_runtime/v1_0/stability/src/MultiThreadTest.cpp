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
#include <cmath>
#include <cstdio>
#include <vector>
#include <thread>

#include "neural_network_runtime/neural_network_runtime.h"
#include "nnrt_utils.h"
#include "model.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Test;
using namespace OHOS::HDI::Nnrt::V1_0;

class MultiThreadTest : public testing::Test {
public:
    void SetUp()
    {
    }
    void TearDown()
    {
    }

protected:
    OHNNCompileParam compileParam;
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
};


/**
 * @tc.number : SUB_AI_NNR_Reliability_North_Stress_0100
 * @tc.name   : 模型编译并发长稳测试
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(MultiThreadTest, SUB_AI_NNR_Reliability_North_Stress_0100, Reliability | MediumTest | Level2)
{
    for (int i = 0; i < STRESS_COUNT; i++) {
        OH_NNModel *model1 = OH_NNModel_Construct();
        ASSERT_NE(nullptr, model1);
        ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model1, graphArgs));

        OH_NNCompilation *compilation1 = OH_NNCompilation_Construct(model1);
        ASSERT_NE(nullptr, compilation1);
        ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation1, compileParam));

        Free(model1, compilation1);
        if (i % PRINT_FREQ == 0) {
            printf("[NnrtTest] SUB_AI_NNR_Reliability_North_Stress_0100 times: %d/%d\n", i, STRESS_COUNT);
        }
    }
}

/**
 * @tc.number : SUB_AI_NNR_Reliability_North_Stress_0200
 * @tc.name   : 模型推理并发长稳测试
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(MultiThreadTest, SUB_AI_NNR_Reliability_North_Stress_0200, Reliability | MediumTest | Level2)
{
    OH_NNModel *model1 = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model1);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model1, graphArgs));

    OH_NNCompilation *compilation1 = OH_NNCompilation_Construct(model1);
    ASSERT_NE(nullptr, compilation1);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation1, compileParam));

    for (int i = 0; i < STRESS_COUNT; i++) {
        OH_NNExecutor *executor1 = OH_NNExecutor_Construct(compilation1);
        ASSERT_NE(nullptr, executor1);
        ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor1, graphArgs, nullptr));
        OH_NNExecutor_Destroy(&executor1);
        ASSERT_EQ(nullptr, executor1);
        if (i % PRINT_FREQ == 0) {
            printf("[NnrtTest] SUB_AI_NNR_Reliability_North_Stress_0200 times: %d/%d\n", i, STRESS_COUNT);
        }
    }
    Free(model1, compilation1);
}
