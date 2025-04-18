/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
#include <vector>

#include "nncore_utils.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Test;

namespace OHOS::NeuralNetworkCore {
void RunDone(void *pointer, OH_NN_ReturnCode returnCode, void* pointerArray[], int32_t intNum)
{
    return;
}

void ServiceDied(void* point)
{
    return;
}
class ExecutorTest : public testing::Test {
protected:
    OHNNCompileParam m_compileParam;
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
};

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Construct_Executor_0100
 * @tc.desc: compilation为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Construct_Executor_0100, Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    ASSERT_EQ(nullptr, OH_NNExecutor_Construct(compilation));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Destroy_Executor_0100
 * @tc.desc: 重复释放executor，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Destroy_Executor_0100, Function | MediumTest | Level1)
{
    OH_NNExecutor *executor = nullptr;
    CreateExecutor(&executor);

    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(nullptr, executor);
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Destroy_Executor_0200
 * @tc.desc: 正常释放，检查executor为空
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Destroy_Executor_0200, Function | MediumTest | Level1)
{
    OH_NNExecutor *executor = nullptr;
    CreateExecutor(&executor);

    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(nullptr, executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Set_Executor_OnRunDone_0100
 * @tc.desc: 在推理完成时设置executor，executor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Set_Executor_OnRunDone_0100, Function | MediumTest | Level1)
{
    NN_OnRunDone onRunDone = RunDone;
    OH_NNExecutor *executor = nullptr;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOnRunDone(executor, onRunDone));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Set_Executor_OnRunDone_0200
 * @tc.desc: 在推理完成时设置executor，合法参数返回不支持
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Set_Executor_OnRunDone_0200, Function | MediumTest | Level1)
{
    NN_OnRunDone onRunDone= RunDone;
    OH_NNExecutor *executor = nullptr;
    CreateExecutor(&executor);
    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, OH_NNExecutor_SetOnRunDone(executor, onRunDone));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Set_Executor_Service_Died_0100
 * @tc.desc: executor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Set_Executor_Service_Died_0100, Function | MediumTest | Level1)
{
    NN_OnServiceDied onServiceDied = ServiceDied;
    OH_NNExecutor *executor = nullptr;

    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOnServiceDied(executor, onServiceDied));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Set_Executor_Service_Died_0200
 * @tc.desc: 合法参数，返回不支持
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Set_Executor_Service_Died_0200, Function | MediumTest | Level1)
{
    NN_OnServiceDied onServiceDied = ServiceDied;
    OH_NNExecutor *executor = nullptr;
    CreateExecutor(&executor);

    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, OH_NNExecutor_SetOnServiceDied(executor, onServiceDied));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0100
 * @tc.desc: executor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0100, Function | MediumTest | Level1)
{
    vector<NN_Tensor*>inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                             outputTensors.data(), outputCount));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0200
 * @tc.desc: executor sync推理，inputTensor数组为空,返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0200, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, nullptr, inputCount,
                                                             outputTensors.data(), outputCount));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0300
 * @tc.desc: executor sync推理，outputTensor数组为空,返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0300, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                             nullptr, outputCount));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0400
 * @tc.desc: executor sync推理，inputCount为0，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0400, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    inputCount = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                             outputTensors.data(), outputCount));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0500
 * @tc.desc: executor sync推理，outputCount为0，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0500, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    outputCount = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                             outputTensors.data(), outputCount));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0600
 * @tc.desc: executor sync推理，inputTensor个数不足，小于正确的输入数量，返回错误
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0600, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount - 1,
                                                             outputTensors.data(), outputCount));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0700
 * @tc.desc: executor sync推理，inputTensor数组个数超过inputNum，返回错误
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0700, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount + 1,
                                                             outputTensors.data(), outputCount));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0800
 * @tc.desc: executor sync推理，outputTensor个数不足，小于正确的输入数量，返回错误
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0800, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                             outputTensors.data(), outputCount - 1));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0900
 * @tc.desc: executor sync推理，outputTensor数组个数超过outputNum，返回错误
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunSync_0900, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                             outputTensors.data(), outputCount + 1));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0100
 * @tc.desc: executor async推理，executor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0100, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    int32_t timeout = 60;
    void* userData = (void*) executor;
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunAsync(executor, inputTensors.data(), inputCount,
                                                              outputTensors.data(), outputCount, timeout, userData));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0200
 * @tc.desc: executor async推理，inputCount为0，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0200, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    int32_t timeout = 60;
    void* userData = (void*) executor;
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    inputCount = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunAsync(executor, inputTensors.data(), inputCount,
                                                              outputTensors.data(), outputCount, timeout, userData));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0300
 * @tc.desc: executor async推理，outputCount为0，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0300, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    int32_t timeout = 60;
    void* userData = (void*) executor;
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    outputCount = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunAsync(executor, inputTensors.data(), inputCount,
                                                              outputTensors.data(), outputCount, timeout, userData));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0400
 * @tc.desc: executor async推理，inputTensor为空指针
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0400, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    int32_t timeout = 60;
    void* userData = (void*) executor;
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunAsync(executor, nullptr, inputCount,
                                                              outputTensors.data(), outputCount, timeout, userData));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0500
 * @tc.desc: executor async推理，outputTensor为空指针
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0500, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    int32_t timeout = 60;
    void* userData = (void*) executor;
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunAsync(executor, inputTensors.data(), inputCount,
                                                              nullptr, outputCount, timeout, userData));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0600
 * @tc.desc: executor async推理，定长模型返回不支持
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_RunASync_0600, Function | MediumTest | Level1)
{
    vector<NN_Tensor*> inputTensors, outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    int32_t timeout = 60;
    void* userData = (void*) executor;
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, OH_NNExecutor_RunAsync(executor, inputTensors.data(), inputCount,
                                                                outputTensors.data(), outputCount,
                                                                timeout, userData));
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Shape_0100
 * @tc.desc: executor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Shape_0100, Function | MediumTest | Level1)
{
    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount = 0;
    uint32_t outputIndex = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputShape(nullptr, outputIndex,
                                                                    &outputDimensions, &outputDimensionCount));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Shape_0200
 * @tc.desc: outputindex不存在，等于输出个数，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Shape_0200, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);

    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount = 0;
    uint32_t addOutputIndex = 4;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputShape(executor, addOutputIndex,
                                                                    &outputDimensions, &outputDimensionCount));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Shape_0300
 * @tc.desc: 定长模型推理成功，获取输出维度成功
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Shape_0300, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    
    vector<NN_Tensor*> inputTensors;
    vector<NN_Tensor*> outputTensors;
    size_t inputCount = 0;
    size_t outputCount = 0;
    GetExecutorInputOutputTensor(executor, inputTensors, inputCount, outputTensors, outputCount);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                   outputTensors.data(), outputCount));

    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount = 0;
    uint32_t addOutputIndex = 0;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetOutputShape(executor, addOutputIndex,
                                                          &outputDimensions, &outputDimensionCount));

    // 销毁Executor
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Shape_0400
 * @tc.desc: 变长模型推理成功，获取输出维度成功
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Shape_0400, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateDynamicExecutor(&executor);

    // 创建输入输出tensorDesc
    vector<NN_TensorDesc*> inputTensorDescs;
    vector<NN_TensorDesc*> outputTensorDescs;
    size_t inputCount = 0;
    size_t outputCount = 0;
    GetExecutorInputOutputTensorDesc(executor, inputTensorDescs, inputCount, outputTensorDescs, outputCount);

    // 设置输入维度合法
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputDimRange(executor, i, &minInputDims,
                                                                &maxInputDims, &shapeLength));
        std::vector<int32_t> minInputDimsT;
        for (size_t j = 0; j < shapeLength; ++j) {
            minInputDimsT.emplace_back(static_cast<int32_t>(minInputDims[j]));
        }
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(inputTensorDescs[i], minInputDimsT.data(), shapeLength));
    }
    std::vector<int32_t> outputShape{1, 2, 2, 1};
    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(outputTensorDescs[i],
                                                          outputShape.data(), outputShape.size()));
    }
    
    vector<NN_Tensor*> inputTensors;
    vector<NN_Tensor*> outputTensors;
    GetExecutorInputOutputTensorByDesc(executor, inputTensors, inputTensorDescs, outputTensors, outputTensorDescs);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                   outputTensors.data(), outputCount));

    int32_t *outputDimensions = nullptr;
    uint32_t outputDimensionCount = 0;
    uint32_t addOutputIndex = 0;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetOutputShape(executor, addOutputIndex,
                                                          &outputDimensions, &outputDimensionCount));

    // 销毁Executor
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensorDesc(inputTensorDescs, outputTensorDescs));
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Count_0100
 * @tc.desc: executor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Count_0100, Function | MediumTest | Level1)
{
    size_t inputCount = ZERO;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetInputCount(nullptr, &inputCount));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Count_0200
 * @tc.desc: inputCount为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Count_0200, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetInputCount(executor, nullptr));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Count_0300
 * @tc.desc: 获取输入个数，返回成功
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Count_0300, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);

    size_t inputCount = ZERO;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetOutputCount(executor, &inputCount));
    ASSERT_LT(ZERO, inputCount);
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Count_0100
 * @tc.desc: executor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Count_0100, Function | MediumTest | Level1)
{
    size_t outputCount = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputCount(nullptr, &outputCount));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Count_0200
 * @tc.desc: outputCount为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Count_0200, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);

    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputCount(executor, nullptr));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Count_0300
 * @tc.desc: 获取输出个数，返回成功
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Output_Count_0300, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);

    size_t outputCount = ZERO;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetOutputCount(executor, &outputCount));
    ASSERT_LT(ZERO, outputCount);
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Create_Input_TensorDesc_0100
 * @tc.desc: excutor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Create_Input_TensorDesc_0100,
         Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);

    size_t index = ZERO;
    ASSERT_EQ(nullptr, OH_NNExecutor_CreateInputTensorDesc(nullptr, index));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Create_Input_TensorDesc_0200
 * @tc.desc: 遍历创建输入tensorDesc，index小于输出个数，成功
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Create_Input_TensorDesc_0200,
         Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    size_t inputCount = ZERO;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputCount(executor, &inputCount));
    NN_TensorDesc* tensorDesc = nullptr;
    for (size_t i = 0; i < inputCount; i++) {
        tensorDesc = OH_NNExecutor_CreateInputTensorDesc(executor, i);
        ASSERT_NE(nullptr, tensorDesc);
        OH_NNTensorDesc_Destroy(&tensorDesc);
    }
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Create_Input_TensorDesc_0300
 * @tc.desc: index等于输出个数，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Create_Input_TensorDesc_0300,
         Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    size_t inputCount = ZERO;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputCount(executor, &inputCount));

    ASSERT_EQ(nullptr, OH_NNExecutor_CreateInputTensorDesc(executor, inputCount));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Create_Output_TensorDesc_0100
 * @tc.desc: excutor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Create_Output_TensorDesc_0100,
         Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);

    size_t index = ZERO;
    ASSERT_EQ(nullptr, OH_NNExecutor_CreateOutputTensorDesc(nullptr, index));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Create_Output_TensorDesc_0200
 * @tc.desc: 遍历创建输入tensorDesc，index小于输出个数，成功
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Create_Output_TensorDesc_0200,
         Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    size_t outputCount = ZERO;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetOutputCount(executor, &outputCount));
    NN_TensorDesc* tensorDesc = nullptr;
    for (size_t i = 0; i < outputCount; i++) {
        tensorDesc = OH_NNExecutor_CreateOutputTensorDesc(executor, i);
        ASSERT_NE(nullptr, tensorDesc);
        OH_NNTensorDesc_Destroy(&tensorDesc);
    }
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Create_Output_TensorDesc_0300
 * @tc.desc: index等于输出个数，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Create_Output_TensorDesc_0300,
         Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);
    size_t outputCount = ZERO;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetOutputCount(executor, &outputCount));
    ASSERT_EQ(nullptr, OH_NNExecutor_CreateOutputTensorDesc(executor, outputCount));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0100
 * @tc.desc: excutor为空，返回失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0100, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    size_t index = ZERO;
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetInputDimRange(executor, index, &minInputDims,
                                                                      &maxInputDims, &shapeLength));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0200
 * @tc.desc: 获取动态输入范围，设置index等于输入个数，超出限制，获取失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0200, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateDynamicExecutor(&executor);

    size_t index = 6;
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;

    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetInputDimRange(executor, index, &minInputDims,
                                                                      &maxInputDims, &shapeLength));
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0300
 * @tc.desc: 变长模型推理，获取输入维度，获取成功，设置输入维度等于最小临界值，成功
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0300, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateDynamicExecutor(&executor);

    // 创建输入输出tensorDesc
    vector<NN_TensorDesc*> inputTensorDescs;
    vector<NN_TensorDesc*> outputTensorDescs;
    size_t inputCount = 0;
    size_t outputCount = 0;
    GetExecutorInputOutputTensorDesc(executor, inputTensorDescs, inputCount, outputTensorDescs, outputCount);

    // 修改tensorDesc中shape为最小临界值
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputDimRange(executor, i, &minInputDims,
                                                                &maxInputDims, &shapeLength));
        std::vector<int32_t> minInputDimsT;
        for (size_t j = 0; j < shapeLength; ++j) {
            minInputDimsT.emplace_back(static_cast<int32_t>(minInputDims[j]));
        }
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(inputTensorDescs[i], minInputDimsT.data(), shapeLength));
    }
    std::vector<int32_t> outputShape{1, 2, 2, 1};
    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(outputTensorDescs[i],
                                                          outputShape.data(), outputShape.size()));
    }
    
    vector<NN_Tensor*> inputTensors;
    vector<NN_Tensor*> outputTensors;
    GetExecutorInputOutputTensorByDesc(executor, inputTensors, inputTensorDescs, outputTensors, outputTensorDescs);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                   outputTensors.data(), outputCount));

    // 销毁Executor
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensorDesc(inputTensorDescs, outputTensorDescs));
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0400
 * @tc.desc: 变长模型编推理，获取输入维度，获取成功，设置输入维度等于最大临界值，成功
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0400, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateDynamicExecutor(&executor);

    // 创建输入输出tensorDesc
    vector<NN_TensorDesc*> inputTensorDescs;
    vector<NN_TensorDesc*> outputTensorDescs;
    size_t inputCount = 0;
    size_t outputCount = 0;
    GetExecutorInputOutputTensorDesc(executor, inputTensorDescs, inputCount, outputTensorDescs, outputCount);

    // 修改tensorDesc中shape为最大临界值
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputDimRange(executor, i, &minInputDims,
                                                                &maxInputDims, &shapeLength));
        std::vector<int32_t> maxInputDimsT;
        for (size_t j = 0; j < shapeLength; ++j) {
            maxInputDimsT.emplace_back(static_cast<int32_t>(maxInputDims[j]));
        }
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(inputTensorDescs[i], maxInputDimsT.data(), shapeLength));
    }
    std::vector<int32_t> outputShape{1, 2, 2, 1};
    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(outputTensorDescs[i],
                                                          outputShape.data(), outputShape.size()));
    }
    
    vector<NN_Tensor*> inputTensors;
    vector<NN_Tensor*> outputTensors;
    GetExecutorInputOutputTensorByDesc(executor, inputTensors, inputTensorDescs, outputTensors, outputTensorDescs);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                   outputTensors.data(), outputCount));

    // 销毁Executor
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensorDesc(inputTensorDescs, outputTensorDescs));
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0500
 * @tc.desc: 变长模型推理，获取输入维度，获取成功，设置输入维度小于最小临界值，推理失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0500, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateDynamicExecutor(&executor);

    // 创建输入输出tensorDesc
    vector<NN_TensorDesc*> inputTensorDescs;
    vector<NN_TensorDesc*> outputTensorDescs;
    size_t inputCount = 0;
    size_t outputCount = 0;
    GetExecutorInputOutputTensorDesc(executor, inputTensorDescs, inputCount, outputTensorDescs, outputCount);

    // 设置输入维度小于最小临界值
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputDimRange(executor, i, &minInputDims,
                                                                &maxInputDims, &shapeLength));
        std::vector<int32_t> minInputDimsT;
        for (size_t j = 0; j < shapeLength; ++j) {
            minInputDimsT.emplace_back(static_cast<int32_t>(minInputDims[j] - 1));
        }
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(inputTensorDescs[i], minInputDimsT.data(), shapeLength));
    }
    std::vector<int32_t> outputShape{1, 2, 2, 1};
    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(outputTensorDescs[i],
                                                          outputShape.data(), outputShape.size()));
    }
    
    vector<NN_Tensor*> inputTensors;
    vector<NN_Tensor*> outputTensors;
    GetExecutorInputOutputTensorByDesc(executor, inputTensors, inputTensorDescs, outputTensors, outputTensorDescs);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                             outputTensors.data(), outputCount));

    // 销毁Executor
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensorDesc(inputTensorDescs, outputTensorDescs));
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0600
 * @tc.desc: 变长模型推理，依次获取输入维度，获取成功，设置输入维度大于最大临界值，推理失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0600, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateDynamicExecutor(&executor);

    // 创建输入输出tensorDesc
    vector<NN_TensorDesc*> inputTensorDescs;
    vector<NN_TensorDesc*> outputTensorDescs;
    size_t inputCount = 0;
    size_t outputCount = 0;
    GetExecutorInputOutputTensorDesc(executor, inputTensorDescs, inputCount, outputTensorDescs, outputCount);

    // 设置输入维度大于最大临界值
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputDimRange(executor, i, &minInputDims,
                                                                &maxInputDims, &shapeLength));
        std::vector<int32_t> maxInputDimsT;
        for (size_t j = 0; j < shapeLength; ++j) {
            maxInputDimsT.emplace_back(static_cast<int32_t>(maxInputDims[j] + 1));
        }
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(inputTensorDescs[i], maxInputDimsT.data(), shapeLength));
    }
    std::vector<int32_t> outputShape{1, 2, 2, 1};
    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(outputTensorDescs[i],
                                                          outputShape.data(), outputShape.size()));
    }
    
    vector<NN_Tensor*> inputTensors;
    vector<NN_Tensor*> outputTensors;
    GetExecutorInputOutputTensorByDesc(executor, inputTensors, inputTensorDescs, outputTensors, outputTensorDescs);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount,
                                                             outputTensors.data(), outputCount));

    // 销毁Executor
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensorDesc(inputTensorDescs, outputTensorDescs));
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0700
 * @tc.desc: 变长模型推理，依次获取输入维度，获取成功，设置输入shape个数超过shapeNum,推理失败
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0700, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateDynamicExecutor(&executor);

    // 创建输入输出tensorDesc
    vector<NN_TensorDesc*> inputTensorDescs;
    vector<NN_TensorDesc*> outputTensorDescs;
    size_t inputCount = 0;
    size_t outputCount = 0;
    GetExecutorInputOutputTensorDesc(executor, inputTensorDescs, inputCount, outputTensorDescs, outputCount);

    // 修改tensorDesc中shape为最大临界值
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputDimRange(executor, i, &minInputDims,
                                                                &maxInputDims, &shapeLength));
        std::vector<int32_t> maxInputDimsT;
        for (size_t j = 0; j < shapeLength; ++j) {
            maxInputDimsT.emplace_back(static_cast<int32_t>(maxInputDims[j]));
        }
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(inputTensorDescs[i], maxInputDimsT.data(), shapeLength));
    }
    std::vector<int32_t> outputShape{1, 2, 2, 1};
    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(outputTensorDescs[i],
                                                          outputShape.data(), outputShape.size()));
    }
    
    vector<NN_Tensor*> inputTensors;
    vector<NN_Tensor*> outputTensors;
    GetExecutorInputOutputTensorByDesc(executor, inputTensors, inputTensorDescs, outputTensors, outputTensorDescs);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_RunSync(executor, inputTensors.data(), inputCount + 1,
                                                             outputTensors.data(), outputCount));

    // 销毁Executor
    OH_NNExecutor_Destroy(&executor);
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensorDesc(inputTensorDescs, outputTensorDescs));
    ASSERT_EQ(OH_NN_SUCCESS, DestroyTensor(inputTensors, outputTensors));
}

/**
 * @tc.name: SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0800
 * @tc.desc: 定长模型推理调用，返回正确
 * @tc.type: FUNC
 */
HWTEST_F(ExecutorTest, SUB_AI_NNRt_Core_Func_North_Executor_Get_Input_Dim_Range_0800, Function | MediumTest | Level1)
{
    OH_NNExecutor* executor = nullptr;
    CreateExecutor(&executor);

    size_t index = ZERO;
    size_t *minInputDims = nullptr;
    size_t *maxInputDims = nullptr;
    size_t shapeLength = ZERO;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetInputDimRange(executor, index, &minInputDims,
                                                            &maxInputDims, &shapeLength));
    OH_NNExecutor_Destroy(&executor);
}
} // namespace OHOS::NeuralNetworkCore
