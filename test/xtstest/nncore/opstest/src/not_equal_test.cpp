/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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
#include <string>
#include <iostream>
#include "nncore_utils.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Test;
class NotEqualTest : public testing::Test {};

struct NotEqualModel1 {
    const std::vector<int32_t> tensor_shape = {3};
    float input0Value[3] = {1, 2, 3};
    float input1Value[3] = {4, 5, 6};
    bool outputValue[3] = {0};

    OHNNOperandTest input0 = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, input0Value, 3*sizeof(float)};
    OHNNOperandTest input1 = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, input1Value, 3*sizeof(float)};
    OHNNOperandTest output = {OH_NN_BOOL, OH_NN_TENSOR, tensor_shape, outputValue, 3*sizeof(bool)};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_NOT_EQUAL,
                               .operands = {input0, input1, output},
                               .paramIndices = {},
                               .inputIndices = {0, 1},
                               .outputIndices = {2}};
};

struct NotEqualModel2 {
    const std::vector<int32_t> tensor_shape = {};
    float input0Value[1] = {1};
    float input1Value[1] = {1};
    bool outputValue[1] = {0};

    OHNNOperandTest input0 = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, input0Value, sizeof(float)};
    OHNNOperandTest input1 = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, input1Value, sizeof(float)};
    OHNNOperandTest output = {OH_NN_BOOL, OH_NN_TENSOR, tensor_shape, outputValue, sizeof(bool)};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_NOT_EQUAL,
                               .operands = {input0, input1, output},
                               .paramIndices = {},
                               .inputIndices = {0, 1},
                               .outputIndices = {2}};
};

struct NotEqualModel3 {
    const std::vector<int32_t> tensor_shape = {3};
    const std::vector<int32_t> input1_shape = {4};
    float input0Value[3] = {1, 2, 3};
    float input1Value[4] = {4, 5, 6, 7};
    bool outputValue[3] = {0};

    OHNNOperandTest input0 = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, input0Value, 3*sizeof(float)};
    OHNNOperandTest input1 = {OH_NN_FLOAT32, OH_NN_TENSOR, input1_shape, input1Value, 4*sizeof(float)};
    OHNNOperandTest output = {OH_NN_BOOL, OH_NN_TENSOR, tensor_shape, outputValue, 3*sizeof(bool)};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_NOT_EQUAL,
                               .operands = {input0, input1, output},
                               .paramIndices = {},
                               .inputIndices = {0, 1},
                               .outputIndices = {2}};
};

struct NotEqualModel4 {
    const std::vector<int32_t> tensor_shape = {};
    float* input0Value = {};
    float* input1Value = {};
    bool* outputValue = {};

    OHNNOperandTest input0 = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, input0Value, 0*sizeof(float)};
    OHNNOperandTest input1 = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, input1Value, 0*sizeof(float)};
    OHNNOperandTest output = {OH_NN_BOOL, OH_NN_TENSOR, tensor_shape, outputValue, 0*sizeof(bool)};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_NOT_EQUAL,
                               .operands = {input0, input1, output},
                               .paramIndices = {},
                               .inputIndices = {0, 1},
                               .outputIndices = {2}};
};

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Build_01
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Build_01
 * @tc.desc   NotEqualModel1模型build测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Build_01, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    EXPECT_NE(nullptr, compilation);

    OHNNCompileParam compileParam{
        .performanceMode = OH_NN_PERFORMANCE_HIGH,
        .priority = OH_NN_PRIORITY_HIGH,
    };
    EXPECT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    EXPECT_NE(nullptr, executor);

    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Build_02
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Build_02
 * @tc.desc   NotEqualModel2模型build测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Build_02, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel2 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    EXPECT_NE(nullptr, compilation);

    OHNNCompileParam compileParam{
        .performanceMode = OH_NN_PERFORMANCE_HIGH,
        .priority = OH_NN_PRIORITY_HIGH,
    };
    EXPECT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    EXPECT_NE(nullptr, executor);

    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Build_03
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Build_03
 * @tc.desc   NotEqualModel3模型build测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Build_03, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel3 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    EXPECT_NE(nullptr, compilation);

    OHNNCompileParam compileParam{
        .performanceMode = OH_NN_PERFORMANCE_HIGH,
        .priority = OH_NN_PRIORITY_HIGH,
    };
    EXPECT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    EXPECT_NE(nullptr, executor);

    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Build_04
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Build_04
 * @tc.desc   NotEqualModel4模型build测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Build_04, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel4 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    EXPECT_NE(nullptr, compilation);

    OHNNCompileParam compileParam{
        .performanceMode = OH_NN_PERFORMANCE_HIGH,
        .priority = OH_NN_PRIORITY_HIGH,
    };
    EXPECT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));

    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    EXPECT_NE(nullptr, executor);

    Free(model, compilation, executor);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Build_05
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Build_05
 * @tc.desc   NotEqualModel1模型输入Tensor+1进行build测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Build_05, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel2 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.operands = {notEqualModel.input0, notEqualModel.input1, notEqualModel.input1, notEqualModel.output};
    graphArgs.inputIndices = {0, 1, 2};
    graphArgs.outputIndices = {3};
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Build_06
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Build_06
 * @tc.desc   NotEqualModel1模型输出Tensor+1进行build测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Build_06, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel2 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.operands = {notEqualModel.input0, notEqualModel.input1, notEqualModel.output, notEqualModel.output};
    graphArgs.inputIndices = {0, 1};
    graphArgs.outputIndices = {2, 3};
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Build_07
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Build_07
 * @tc.desc   NotEqualModel1模型传入非法参数进行build测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Build_07, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel2 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;

    int8_t activationValue = OH_NN_FUSED_NONE;
    OHNNOperandTest activation = {OH_NN_INT8, OH_NN_ADD_ACTIVATIONTYPE, {}, &activationValue, sizeof(int8_t)};
    graphArgs.operands = {notEqualModel.input0, notEqualModel.input1, notEqualModel.output, activation};
    graphArgs.paramIndices = {3};
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_01
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_01
 * @tc.desc   模型构图，未添加操作数
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_01, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    OHNNGraphArgs graphArgs;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, SingleModelBuildEndStep(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_02
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_02
 * @tc.desc   模型构图，未设置输入输出
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_02, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_03
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_03
 * @tc.desc   模型构图，设置输入输出，构图成功
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_Finish_03, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_01
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_01
 * @tc.desc   设置操作数值，操作数不存在
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_01, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;

    NN_TensorDesc* tensorDesc = nullptr;
    std::vector<NN_TensorDesc*> tensorDescVec;

    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        tensorDesc = createTensorDesc(operandTem.shape.data(),
                                      (uint32_t) operandTem.shape.size(),
                                      operandTem.dataType, operandTem.format);
        tensorDescVec.emplace_back(tensorDesc);
        EXPECT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
        EXPECT_EQ(OH_NN_SUCCESS, OH_NNModel_SetTensorType(model, i, operandTem.type));

        if (std::find(graphArgs.paramIndices.begin(), graphArgs.paramIndices.end(), i) !=
            graphArgs.paramIndices.end()) {
            EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorData(
                model, 1000+i, operandTem.data, operandTem.length));
        }
    }

    FreeTensorDescVec(tensorDescVec);
    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_02
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_02
 * @tc.desc   设置操作数值，buufer为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_02, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;

    NN_TensorDesc* tensorDesc = nullptr;
    std::vector<NN_TensorDesc*> tensorDescVec;

    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        tensorDesc = createTensorDesc(operandTem.shape.data(),
                                      (uint32_t) operandTem.shape.size(),
                                      operandTem.dataType, operandTem.format);
        tensorDescVec.emplace_back(tensorDesc);
        EXPECT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
        EXPECT_EQ(OH_NN_SUCCESS, OH_NNModel_SetTensorType(model, i, operandTem.type));

        if (std::find(graphArgs.paramIndices.begin(), graphArgs.paramIndices.end(), i) !=
            graphArgs.paramIndices.end()) {
            EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorData(model, i, nullptr, operandTem.length));
        }
    }

    FreeTensorDescVec(tensorDescVec);
    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_03
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_03
 * @tc.desc   设置操作数值，length为0
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SetOperandValue_03, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;

    NN_TensorDesc* tensorDesc = nullptr;
    std::vector<NN_TensorDesc*> tensorDescVec;

    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        tensorDesc = createTensorDesc(operandTem.shape.data(),
                                      (uint32_t) operandTem.shape.size(),
                                      operandTem.dataType, operandTem.format);
        tensorDescVec.emplace_back(tensorDesc);
        EXPECT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
        EXPECT_EQ(OH_NN_SUCCESS, OH_NNModel_SetTensorType(model, i, operandTem.type));

        if (std::find(graphArgs.paramIndices.begin(), graphArgs.paramIndices.end(), i) !=
            graphArgs.paramIndices.end()) {
            EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorData(model, 1000+i, operandTem.data, 0));
        }
    }

    FreeTensorDescVec(tensorDescVec);
    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_01
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_01
 * @tc.desc   设置输入输出，inputIndices为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_01,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, nullptr, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_02
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_02
 * @tc.desc   设置输入输出，inputindices中data为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_02,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    inputIndices.data = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_03
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_03
 * @tc.desc   设置输入输出，inputindices中data对应序号不存在
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_03,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    graphArgs.inputIndices = {100000};
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_04
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_04
 * @tc.desc   设置输入输出，inputindices中size为0
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_04,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    inputIndices.size = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_05
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_05
 * @tc.desc   设置输入输出，outputindices为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_05,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, nullptr));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_06
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_06
 * @tc.desc   设置输入输出，outputindices中data为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_06,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    outputIndices.data = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_07
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_07
 * @tc.desc   设置输入输出，outputindices中data对应序号不存在
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_07,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    graphArgs.outputIndices = {100000};
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_08
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_08
 * @tc.desc   设置输入输出，outputindices中size为0
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_SpecifyInputsAndOutputs_08,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    outputIndices.size = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_01
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_01
 * @tc.desc   添加算子，paramindices为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_01, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                               nullptr, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_02
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_02
 * @tc.desc   添加算子，paramindices中data为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_02, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    paramIndices.data = nullptr;
    EXPECT_EQ(OH_NN_SUCCESS, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                     &paramIndices, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_03
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_03
 * @tc.desc   添加算子，paramindices中data对应序号不存在
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_03, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    graphArgs.paramIndices = {100000};
    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                               &paramIndices, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_04
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_04
 * @tc.desc   添加算子，paramindices中size为0
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_04, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    paramIndices.size = 0;
    EXPECT_EQ(OH_NN_SUCCESS, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                     &paramIndices, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_05
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_05
 * @tc.desc   添加算子，inputindices为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_05, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                               &paramIndices, nullptr, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_06
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_06
 * @tc.desc   添加算子，inputindices中data为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_06, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    inputIndices.data = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                               &paramIndices, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_07
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_07
 * @tc.desc   添加算子，inputindices中data对应序号不存在
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_07, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    graphArgs.inputIndices = {100000};
    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                               &paramIndices, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_08
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_08
 * @tc.desc   添加算子，inputindices中size为0
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_08, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    inputIndices.size = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                               &paramIndices, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_09
 * @tc.number SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_09
 * @tc.desc   添加算子，outputindices为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(NotEqualTest, SUB_AI_NNRt_Func_North_NotEqual_Model_AddOperation_09, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    NotEqualModel1 notEqualModel;
    OHNNGraphArgs graphArgs = notEqualModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(nullptr, graphArgs.operationType,
                                                               &paramIndices, &inputIndices, nullptr));

    Free(model, nullptr, nullptr);
}