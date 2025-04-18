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

#include "nnrt_utils.h"
#include "model.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Test;
using namespace OHOS::HDI::Nnrt::V2_0;

namespace {

class ExecutorTest : public testing::Test {
protected:
    OHOS::sptr<V2_0::MockIDevice> device;
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
    OHNNCompileParam compileParam;
};

void ExecuteModel(OH_NNExecutor *executor, const OHNNGraphArgs &graphArgs, float* expect)
{
    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, expect));
}

} // namespace

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Create_0100
 * @tc.name   : 创建执行实例，compilation为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Create_0100, Function | MediumTest | Level3)
{
    OH_NNExecutor *executor = OH_NNExecutor_Construct(nullptr);
    ASSERT_EQ(nullptr, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Create_0200
 * @tc.name   : 创建执行实例，compilation未完成编译
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Create_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);

    const size_t *devicesID{nullptr};
    uint32_t devicesCount{0};
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNDevice_GetAllDevicesID(&devicesID, &devicesCount));
    size_t targetDevice = devicesID[0];

    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(compilation, targetDevice));
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_EQ(nullptr, executor);

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0100
 * @tc.name   : 设置输入，executor为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0100, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    uint32_t inputIndex = 0;
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_SetInput(nullptr, inputIndex, &operand, operandTem.data, operandTem.length));

    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0200
 * @tc.name   : 设置输入，inputIndex不存在
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    uint32_t inputIndex = 100000;
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0300
 * @tc.name   : 设置输入，operand参数不一致
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0300, Function | MediumTest | Level3)
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
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0400
 * @tc.name   : 设置输入，operand形状改变
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0400, Function | MediumTest | Level3)
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
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    int32_t dimensions[3]{3, 3, 3};
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), dimensions, quantParam,
                             operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0500
 * @tc.name   : 设置输入，buffer为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0500, Function | MediumTest | Level3)
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
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, nullptr, operandTem.length));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0600
 * @tc.name   : 设置输入，length小于输入长度
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0600, Function | MediumTest | Level3)
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
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, 0));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0700
 * @tc.name   : 设置输入，重复设置同一inputIndex
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0700, Function | MediumTest | Level2)
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
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0800
 * @tc.name   : 变长模型，设置超出动态范围的输入
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0800, Function | MediumTest | Level2)
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
    uint32_t inputIndex = 0;
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    int32_t shape[4] = {1, 100, 100, 11};
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), shape,
                             quantParam, operandTem.type};

    size_t length = shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float);
    void* data = malloc(length);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, data, length));

    free(data);
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_0900
 * @tc.name   : 变长模型，设置等于动态范围的最小输入
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_0900, Function | MediumTest | Level2)
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
    uint32_t inputIndex = 0;
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    int32_t shape[4] = {1, 1, 1, 1};       //minInputDims
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), shape,
                             quantParam, operandTem.type};

    size_t length = shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float);
    void* data = malloc(length);
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, data, length));

    free(data);
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetInput_1000
 * @tc.name   : 变长模型，设置等于动态范围的最大输入
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetInput_1000, Function | MediumTest | Level2)
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
    uint32_t inputIndex = 0;
    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    int32_t shape[4] = {1, 100, 100, 10};       //maxInputDims
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), shape,
                             quantParam, operandTem.type};

    size_t length = shape[0] * shape[1] * shape[2] * shape[3] * sizeof(float);
    void* data = malloc(length);
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, data, length));

    free(data);
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetOutput_0100
 * @tc.name   : 设置输出，executor为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetOutput_0100, Function | MediumTest | Level3)
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
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_INVALID_PARAMETER,
                      OH_NNExecutor_SetOutput(nullptr, outputIndex, operandTem.data, operandTem.length));
            outputIndex += 1;
        }
    }
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetOutput_0200
 * @tc.name   : 设置输出，outputIndex不存在
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetOutput_0200, Function | MediumTest | Level3)
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
    uint32_t outputIndex = 10000;

    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        auto quantParam = operandTem.quantParam;
        OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                                 quantParam, operandTem.type};
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_INVALID_PARAMETER,
                      OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, operandTem.length));
            outputIndex += 1;
        }
    }
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetOutput_0300
 * @tc.name   : 设置输出，buffer为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetOutput_0300, Function | MediumTest | Level3)
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
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_INVALID_PARAMETER,
                      OH_NNExecutor_SetOutput(executor, outputIndex, nullptr, operandTem.length));
            outputIndex += 1;
        }
    }
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetOutput_0400
 * @tc.name   : 设置输出，length小于输出长度
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetOutput_0400, Function | MediumTest | Level3)
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
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, 0));
            outputIndex += 1;
        }
    }
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_SetOutput_0500
 * @tc.name   : 设置输出，重复设置同一outputIndex
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_SetOutput_0500, Function | MediumTest | Level2)
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
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, operandTem.length));
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, operandTem.length));
            outputIndex += 1;
        }
    }
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Run_0100
 * @tc.name   : 模型推理，executor为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Run_0100, Function | MediumTest | Level3)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_Run(nullptr));
}
/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Run_0200
 * @tc.name   : 模型推理，executor未设置输入
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Run_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    uint32_t outputIndex = 0;

    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
            graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, operandTem.length));
        }
    }
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_Run(executor));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Run_0300
 * @tc.name   : 模型推理，executor未设置输出
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Run_0300, Function | MediumTest | Level3)
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
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            inputIndex += 1;
        }
    }
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_Run(executor));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Run_0400
 * @tc.name   : 模型推理，executor设置输入个数不足
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Run_0400, Function | MediumTest | Level3)
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

    const OHNNOperandTest &operandTem = graphArgs.operands[0];
    auto quantParam = operandTem.quantParam;
    OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                             quantParam, operandTem.type};
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
    inputIndex += 1;
    const OHNNOperandTest &operandOut = graphArgs.operands[3];
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetOutput(executor, outputIndex, operandOut.data, operandOut.length));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_Run(executor));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Run_0500
 * @tc.name   : 模型推理，executor设置输出个数不足
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Run_0500, Function | MediumTest | Level3)
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

    graphArgs.outputIndices = {3};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Run_0600
 * @tc.name   : 定长模型推理测试
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Run_0600, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Run_0700
 * @tc.name   : 变长模型推理测试
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Run_0700, Function | MediumTest | Level1)
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
    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, avgModel.expectValue));
    // check result
    EXPECT_TRUE(CheckOutput(avgModel.outputValue, avgModel.expectValue));
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0100
 * @tc.name   : 获取输出维度，executor为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0100, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));
    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount{0};
    uint32_t addOutputIndex = {0};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_GetOutputShape(nullptr, addOutputIndex, &outputDimensions, &outputDimensionCount));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0200
 * @tc.name   : 获取输出维度，outputIndex不存在
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));
    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount{0};
    uint32_t addOutputIndex = {10000};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_GetOutputShape(executor, addOutputIndex, &outputDimensions, &outputDimensionCount));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0300
 * @tc.name   : 获取输出维度，*dimensions为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));
    uint32_t outputDimensionCount{0};
    uint32_t addOutputIndex = {0};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_GetOutputShape(executor, addOutputIndex, nullptr, &outputDimensionCount));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0400
 * @tc.name   : 获取输出维度，**dimensions非nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0400, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));
    int32_t outputDimensions{2};
    int32_t *pOutputDimensions = &outputDimensions;
    uint32_t outputDimensionCount{0};
    uint32_t addOutputIndex = {0};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_GetOutputShape(executor, addOutputIndex, &pOutputDimensions, &outputDimensionCount));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0500
 * @tc.name   : 获取输出维度，*dimensionCount为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0500, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));
    int32_t *outputDimensions = nullptr;
    uint32_t addOutputIndex = {0};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER,
              OH_NNExecutor_GetOutputShape(executor, addOutputIndex, &outputDimensions, nullptr));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0600
 * @tc.name   : 未调用推理接口，获取输出维度
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0600, Function | MediumTest | Level3)
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
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, operandTem.length));
            outputIndex += 1;
        }
    }
    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount{0};
    uint32_t addOutputIndex = {0};
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNExecutor_GetOutputShape(executor, addOutputIndex, &outputDimensions, &outputDimensionCount));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0700
 * @tc.name   : 模型推理成功，获取输出维度
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0700, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));
    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount{0};
    uint32_t addOutputIndex = {0};
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNExecutor_GetOutputShape(executor, addOutputIndex, &outputDimensions, &outputDimensionCount));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0800
 * @tc.name   : 变长模型推理成功，获取输出维度
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_GetOutputDimensions_0800, Function | MediumTest | Level1)
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
    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, avgModel.expectValue));

    // check result
    EXPECT_TRUE(CheckOutput(avgModel.outputValue, avgModel.expectValue));
    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount{0};
    uint32_t addOutputIndex = {0};
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNExecutor_GetOutputShape(executor, addOutputIndex, &outputDimensions, &outputDimensionCount));

    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Destroy_0100
 * @tc.name   : 销毁执行器实例，*executor为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Destroy_0100, Function | MediumTest | Level3)
{
    OH_NNExecutor *executor = nullptr;
    ASSERT_NO_THROW(OH_NNExecutor_Destroy(&executor));
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Destroy_0200
 * @tc.name   : 销毁执行器实例，executor释放
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Destroy_0200, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    ASSERT_EQ(OH_NN_SUCCESS, ExecuteGraphMock(executor, graphArgs, addModel.expectValue));

    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));

    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(nullptr, executor);

    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNR_Func_North_Executor_Combine_0100
 * @tc.name   : 并发模型推理，推理成功
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNR_Func_North_Executor_Combine_0100, Function | MediumTest | Level2)
{
    OH_NNModel *model1 = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model1);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model1, graphArgs));

    OH_NNModel *model2 = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model2);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model2, graphArgs));

    OH_NNCompilation *compilation1 = OH_NNCompilation_Construct(model1);
    ASSERT_NE(nullptr, compilation1);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation1, compileParam));

    OH_NNCompilation *compilation2 = OH_NNCompilation_Construct(model2);
    ASSERT_NE(nullptr, compilation2);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation2, compileParam));

    OH_NNExecutor *executor1 = OH_NNExecutor_Construct(compilation1);
    ASSERT_NE(nullptr, executor1);

    OH_NNExecutor *executor2 = OH_NNExecutor_Construct(compilation2);
    ASSERT_NE(nullptr, executor2);

    std::thread th1(ExecuteModel, executor1, graphArgs, addModel.expectValue);
    std::thread th2(ExecuteModel, executor2, graphArgs, addModel.expectValue);
    th1.join();
    th2.join();
    Free(model1, compilation1, executor1);
    Free(model2, compilation2, executor2);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Combine_0200
 * @tc.name   : 多次设置输入，仅首次成功，模型推理
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Combine_0200, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);

    float valueX2[4] = {3, 2, 1, 0};
    uint32_t inputIndex = 0;
    uint32_t outputIndex = 0;
    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        auto quantParam = operandTem.quantParam;
        OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                                 quantParam, operandTem.type};
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            EXPECT_EQ(OH_NN_INVALID_PARAMETER,
                      OH_NNExecutor_SetInput(executor, 3, &operand, valueX2, operandTem.length));
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, operandTem.length));
            OHOS::sptr<V2_0::MockIDevice> device = V2_0::MockIDevice::GetInstance();
            ASSERT_EQ(OH_NN_SUCCESS, device->MemoryCopy(addModel.expectValue, operandTem.length));
            outputIndex += 1;
        }
    }
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(executor));
    // check result
    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Combine_0300
 * @tc.name   : 多次设置输出，仅首次生效，模型推理
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Combine_0300, Function | MediumTest | Level1)
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
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, operandTem.length));
            ASSERT_EQ(OH_NN_INVALID_PARAMETER,
            OH_NNExecutor_SetOutput(executor, outputIndex+10, operandTem.data, operandTem.length));
            OHOS::sptr<V2_0::MockIDevice> device = V2_0::MockIDevice::GetInstance();
            ASSERT_EQ(OH_NN_SUCCESS, device->MemoryCopy(addModel.expectValue, operandTem.length));
            outputIndex += 1;
        }
    }
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(executor));
     // check result
    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(executor));
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Combine_0400
 * @tc.name   : 模型推理，共享输入非共享输出
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Combine_0400, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, executor);
    OH_NN_Memory *OHNNMemory[graphArgs.inputIndices.size()];
    uint32_t inputIndex = 0;
    uint32_t outputIndex = 0;

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

            ASSERT_EQ(EOK, memcpy_s(inputMemory->data,
                operandTem.length, static_cast<void*>(operandTem.data), operandTem.length));
            OHNNMemory[inputIndex] = inputMemory;
            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetOutput(executor, outputIndex, operandTem.data, operandTem.length));
            OHOS::sptr<V2_0::MockIDevice> device = V2_0::MockIDevice::GetInstance();
            ASSERT_EQ(OH_NN_SUCCESS, device->MemoryCopy(addModel.expectValue, operandTem.length));
            outputIndex += 1;
        }
    }
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(executor));
    // check result
    EXPECT_TRUE(CheckOutput(addModel.outputValue, addModel.expectValue));

    for (size_t i = 0; i < graphArgs.inputIndices.size(); i++) {
        OH_NNExecutor_DestroyInputMemory(executor, i, &OHNNMemory[i]);
        ASSERT_EQ(OHNNMemory[i], nullptr);
    }
    Free(model, compilation, executor);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Executor_Combine_0500
 * @tc.name   : 模型推理，非共享输入共享输出
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Func_North_Executor_Combine_0500, Function | MediumTest | Level2)
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
    OH_NN_Memory *outputMemory;

    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        auto quantParam = operandTem.quantParam;
        OH_NN_Tensor operand = {operandTem.dataType, (uint32_t)operandTem.shape.size(), operandTem.shape.data(),
                                 quantParam, operandTem.type};
        if (std::find(graphArgs.inputIndices.begin(), graphArgs.inputIndices.end(), i) !=
            graphArgs.inputIndices.end()) {
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetInput(executor, inputIndex, &operand, operandTem.data, operandTem.length));

            inputIndex += 1;
        } else if (std::find(graphArgs.outputIndices.begin(), graphArgs.outputIndices.end(), i) !=
                   graphArgs.outputIndices.end()) {

            outputMemory = OH_NNExecutor_AllocateOutputMemory(executor, outputIndex, operandTem.length);
            ASSERT_NE(nullptr, outputMemory);
            ASSERT_EQ(OH_NN_SUCCESS,
                      OH_NNExecutor_SetOutputWithMemory(executor, outputIndex, outputMemory));
            OHOS::sptr<V2_0::MockIDevice> device = V2_0::MockIDevice::GetInstance();
            ASSERT_EQ(OH_NN_SUCCESS, device->MemoryCopy(addModel.expectValue, operandTem.length));
            outputIndex += 1;
        }
    }
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(executor));
    // check result
    EXPECT_TRUE(CheckOutput(static_cast<float*>(const_cast<void*>(outputMemory->data)),
        static_cast<float*>(addModel.expectValue)));

    OH_NNExecutor_DestroyOutputMemory(executor, 0, &outputMemory);
    ASSERT_EQ(outputMemory, nullptr);
    Free(model, compilation, executor);
}