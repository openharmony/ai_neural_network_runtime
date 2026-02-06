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
#include <cstdlib>

#include "nnrt_utils.h"
#include "model.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Test;
using namespace OHOS::HDI::Nnrt::V1_0;

namespace {

class MemoryTest : public testing::Test {
protected:
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
    OHNNCompileParam compileParam;
};

void CheckCreateInputMemory(OH_NNExecutor *executor, uint32_t inputIndex, size_t length)
{
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, inputIndex, length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyInputMemory(executor, inputIndex, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
}

void CheckCreateOutputMemory(OH_NNExecutor *executor, uint32_t outputIndex, size_t length)
{
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, outputIndex, length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyOutputMemory(executor, outputIndex, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
}

} // namespace

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0100
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0100
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0100, Function | MediumTest | Level3)
{
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(nullptr, 0, 4);
    ASSERT_EQ(nullptr, OHNNMemory);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0200
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0200
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 2, graphArgs.operands[0].length);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0300
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0300
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, 0);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0400
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0400
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0400, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, 1024 * 1024 * 1024 + 1);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0500
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0500
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0500, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NN_Memory *OHNNMemory2 = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory2);
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory2);
    ASSERT_EQ(nullptr, OHNNMemory2);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0600
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0600
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateInputMemory_0600, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    std::thread th1(CheckCreateInputMemory, executor, 0, graphArgs.operands[0].length);
    std::thread th2(CheckCreateInputMemory, executor, 1, graphArgs.operands[1].length);
    th1.join();
    th2.join();
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0100
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0100
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0100, Function | MediumTest | Level3)
{
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(nullptr, 0, 4);
    ASSERT_EQ(nullptr, OHNNMemory);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0200
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0200
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 2, graphArgs.operands[0].length);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0300
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0300
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, 0);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0400
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0400
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0400, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, 1024 * 1024 * 1024 + 1);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0500
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0500
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0500, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NN_Memory *OHNNMemory2 = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory2);
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory2);
    ASSERT_EQ(nullptr, OHNNMemory2);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0600
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0600
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_CreateOutputMemory_0600, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    TopKModel topKModel;
    graphArgs = topKModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OHNNCompileParam compileParam;
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    std::thread th1(CheckCreateOutputMemory, executor, 0, graphArgs.operands[3].length);
    std::thread th2(CheckCreateOutputMemory, executor, 1, graphArgs.operands[4].length);
    th1.join();
    th2.join();
    Free(model, compilation, executor);
}
/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0100
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0100
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0100, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyInputMemory(nullptr, 0, &OHNNMemory);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0200
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0200
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyInputMemory(executor, 1, &OHNNMemory);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0300
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0300
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = nullptr;
    ASSERT_NO_THROW(OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory));
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0400
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0400
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0400, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NN_Memory *OHNNMemory2 = OH_NNExecutor_AllocateInputMemory(executor, 1, graphArgs.operands[1].length);
    ASSERT_NE(nullptr, OHNNMemory2);
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyInputMemory(executor, 1, &OHNNMemory2);
    ASSERT_EQ(nullptr, OHNNMemory2);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0500
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0500
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyInputMemory_0500, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NN_Memory *OHNNMemory2 = OH_NNExecutor_AllocateInputMemory(executor, 1, graphArgs.operands[1].length);
    ASSERT_NE(nullptr, OHNNMemory2);
    std::thread th1(OH_NNExecutor_DestroyInputMemory, executor, 0, &OHNNMemory);
    std::thread th2(OH_NNExecutor_DestroyInputMemory, executor, 1, &OHNNMemory2);
    th1.join();
    th2.join();
    ASSERT_EQ(nullptr, OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory2);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0100
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0100
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0100, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyOutputMemory(nullptr, 0, &OHNNMemory);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0200
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0200
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyOutputMemory(executor, 1, &OHNNMemory);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0300
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0300
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    ASSERT_NO_THROW(OH_NNExecutor_DestroyOutputMemory(executor, 0, nullptr));
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0400
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0400
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0400, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NN_Memory *OHNNMemory2 = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory2);
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory2);
    ASSERT_EQ(nullptr, OHNNMemory2);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0500
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0500
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_DestroyOutputMemory_0500, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    TopKModel topKModel;
    graphArgs = topKModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    OH_NN_Memory *OHNNMemory2 = OH_NNExecutor_AllocateOutputMemory(executor, 1, graphArgs.operands[1].length);
    ASSERT_NE(nullptr, OHNNMemory2);
    std::thread th1(OH_NNExecutor_DestroyOutputMemory, executor, 0, &OHNNMemory);
    std::thread th2(OH_NNExecutor_DestroyOutputMemory, executor, 1, &OHNNMemory2);
    th1.join();
    th2.join();
    ASSERT_EQ(nullptr, OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory2);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0100
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0100
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0100, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInputWithMemory(nullptr, 0, &operand, OHNNMemory));
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0200
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0200
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInputWithMemory(executor, 2, &operand, OHNNMemory));
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0300
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0300
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInputWithMemory(executor, 0, nullptr, OHNNMemory));
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0400
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0400
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0400, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory1 = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory1);
    const OHNNOperandTest &operandTem = graphArgs.operands[2];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInputWithMemory(executor, 0, &operand, OHNNMemory1));
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory1);
    ASSERT_EQ(nullptr, OHNNMemory1);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0500
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0500
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0500, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInputWithMemory(executor, 0, &operand, nullptr));
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0600
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0600
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetInputFromMemory_0600, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateInputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetInputWithMemory(executor, 0, &operand, OHNNMemory));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetInputWithMemory(executor, 0, &operand, OHNNMemory));
    OH_NNExecutor_DestroyInputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0100
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0100
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0100, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutputWithMemory(nullptr, 0, OHNNMemory));
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0200
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0200
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutputWithMemory(executor, 1, OHNNMemory));
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0300
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0300
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutputWithMemory(executor, 0, nullptr));
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0400
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0400
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_SetOutputFromMemory_0400, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory = OH_NNExecutor_AllocateOutputMemory(executor, 0, graphArgs.operands[0].length);
    ASSERT_NE(nullptr, OHNNMemory);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetOutputWithMemory(executor, 0, OHNNMemory));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetOutputWithMemory(executor, 0, OHNNMemory));
    OH_NNExecutor_DestroyOutputMemory(executor, 0, &OHNNMemory);
    ASSERT_EQ(nullptr, OHNNMemory);
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_Run_0100
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_Run_0100
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_Run_0100, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    uint32_t inputIndex = 0;
    uint32_t outputIndex = 0;
    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        auto quantParam = operandTem.quantParam;
        OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                                 quantParam, operandTem.type};
        if (i == 0) {
            OH_NN_Memory *inputMemory = OH_NNExecutor_AllocateInputMemory(executor, inputIndex, operandTem.length);
            ASSERT_NE(nullptr, inputMemory);
            ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetInputWithMemory(executor, inputIndex, &operand, inputMemory));

            ASSERT_EQ(EOK, memcpy_s(inputMemory->data, operandTem.length, static_cast<void*>(operandTem.data), operandTem.length));

            OH_NNExecutor_DestroyInputMemory(executor, inputIndex, &inputMemory);
            ASSERT_EQ(nullptr, inputMemory);
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            OH_NN_Memory *outputMemory = OH_NNExecutor_AllocateOutputMemory(executor, outputIndex, operandTem.length);
            ASSERT_NE(nullptr, outputMemory);
            ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetOutputWithMemory(executor, outputIndex, outputMemory));
            OH_NNExecutor_DestroyOutputMemory(executor, outputIndex, &outputMemory);
            ASSERT_EQ(nullptr, outputMemory);
            outputIndex += 1;
        }
    }
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_Run(executor));
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_Run_0200
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_Run_0200
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL3
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_Run_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    uint32_t inputIndex = 0;
    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        auto quantParam = operandTem.quantParam;
        OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                                 quantParam, operandTem.type};
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            OH_NN_Memory *inputMemory = OH_NNExecutor_AllocateInputMemory(executor, inputIndex, operandTem.length);
            ASSERT_NE(nullptr, inputMemory);
            ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetInputWithMemory(executor, inputIndex, &operand, inputMemory));
            ASSERT_EQ(EOK, memcpy_s(inputMemory->data, operandTem.length, static_cast<void*>(operandTem.data), operandTem.length));
            OH_NNExecutor_DestroyInputMemory(executor, inputIndex, &inputMemory);
            ASSERT_EQ(nullptr, inputMemory);
        }
    }
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_Run(executor));
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_Run_0300
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_Run_0300
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_Run_0300, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    size_t ioSize = graphArgs.inputIndices.size() + graphArgs.outputIndices.size();
    OH_NN_Memory *OHNNMemory[ioSize];
    ASSERT_EQ(OH_NN_SUCCESS, ExecutorWithMemory(executor, graphArgs, OHNNMemory, addModel.expectValue));
    for (size_t i = 0; i < graphArgs.inputIndices.size(); i++) {
        OH_NNExecutor_DestroyInputMemory(executor, i, &OHNNMemory[i]);
        ASSERT_EQ(OHNNMemory[i], nullptr);
    }
    for (size_t j = 0; j < graphArgs.outputIndices.size(); j++) {
        auto outputIndex = graphArgs.inputIndices.size() + j;
        // check memory output
        EXPECT_TRUE(CheckOutput(static_cast<float*>(const_cast<void*>(OHNNMemory[outputIndex]->data)),
        static_cast<float*>(addModel.expectValue)));
        OH_NNExecutor_DestroyOutputMemory(executor, j, &OHNNMemory[outputIndex]);
        ASSERT_EQ(OHNNMemory[outputIndex], nullptr);
    }
    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Executor_Memory_Run_0400
 * @tc.number SUB_AI_NNRt_Func_North_Executor_Memory_Run_0400
 * @tc.desc   [C- SOFTWARE -0200]
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(MemoryTest, SUB_AI_NNRt_Func_North_Executor_Memory_Run_0400, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    AvgPoolDynamicModel avgModel;
    graphArgs = avgModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);

    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    avgModel.dynamicInput.shape = {1, 3, 3, 1};
    avgModel.output.shape = {1, 2, 2, 1};
    graphArgs.operands = {avgModel.dynamicInput, avgModel.kernel,     avgModel.strides,
                          avgModel.padMode,      avgModel.activation, avgModel.output};
    size_t ioSize = graphArgs.inputIndices.size() + graphArgs.outputIndices.size();
    OH_NN_Memory *OHNNMemory[ioSize];

    ASSERT_EQ(OH_NN_SUCCESS, ExecutorWithMemory(executor, graphArgs, OHNNMemory, avgModel.expectValue));
    
    for (size_t i = 0; i < graphArgs.inputIndices.size(); i++) {
        OH_NNExecutor_DestroyInputMemory(executor, i, &OHNNMemory[i]);
        ASSERT_EQ(OHNNMemory[i], nullptr);
    }
    for (size_t j = 0; j < graphArgs.outputIndices.size(); j++) {
        auto outputIndex = graphArgs.inputIndices.size() + j;
        // check memory output
        EXPECT_TRUE(CheckOutput(static_cast<float*>(const_cast<void*>(OHNNMemory[outputIndex]->data)),
        static_cast<float*>(avgModel.expectValue)));
        OH_NNExecutor_DestroyOutputMemory(executor, j, &OHNNMemory[outputIndex]);
        ASSERT_EQ(OHNNMemory[outputIndex], nullptr);
    }
    Free(model, compilation, executor);
}