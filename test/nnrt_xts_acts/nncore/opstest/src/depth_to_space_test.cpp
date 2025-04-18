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
class DepthToSpaceTest : public testing::Test {};

struct DepthToSpaceModel1 {
    const std::vector<int32_t> input_shape = {1, 4, 4, 4};
    const std::vector<int32_t> output_shape = {1, 1, 8, 8};
    std::vector<int64_t> blockSizeValue = {2};
    float inputValue[1][4][4][4] = {{{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}},
                                     {{17, 18, 19, 20}, {21, 22, 23, 24}, {25, 26, 27, 28}, {29, 30, 31, 32}},
                                     {{33, 34, 35, 36}, {37, 38, 39, 40}, {41, 42, 43, 44}, {45, 46, 47, 48}},
                                     {{49, 50, 51, 52}, {53, 54, 55, 56}, {57, 58, 59, 60}, {61, 62, 63, 64}}}};
    float outputValue[1][1][8][8] = {0};

    OHNNOperandTest input = {OH_NN_FLOAT32, OH_NN_TENSOR, input_shape, inputValue, 64*sizeof(float)};
    OHNNOperandTest output = {OH_NN_FLOAT32, OH_NN_TENSOR, output_shape, outputValue, 64*sizeof(float)};
    OHNNOperandTest blockSize = {OH_NN_INT64, OH_NN_DEPTH_TO_SPACE_BLOCK_SIZE, {1}, &blockSizeValue, sizeof(int64_t)};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_DEPTH_TO_SPACE,
                               .operands = {input, output, blockSize},
                               .paramIndices = {2},
                               .inputIndices = {0},
                               .outputIndices = {1}};
};

struct DepthToSpaceModel2 {
    const std::vector<int32_t> tensor_shape = {1, 12, 1, 1};
    std::vector<int64_t> blockSizeValue = {0};
    float inputValue[1][12][1][1] = {{{{1}}, {{2}}, {{3}}, {{4}}, {{5}}, {{6}},
                                      {{7}}, {{8}}, {{9}}, {{10}}, {{11}}, {{12}}}};
    float outputValue[1][12][1][1] = {0};

    OHNNOperandTest input = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, inputValue, 12*sizeof(float)};
    OHNNOperandTest output = {OH_NN_FLOAT32, OH_NN_TENSOR, tensor_shape, outputValue, 12*sizeof(float)};
    OHNNOperandTest blockSize = {OH_NN_INT64, OH_NN_DEPTH_TO_SPACE_BLOCK_SIZE, {1}, &blockSizeValue, sizeof(int64_t)};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_DEPTH_TO_SPACE,
                               .operands = {input, output, blockSize},
                               .paramIndices = {2},
                               .inputIndices = {0},
                               .outputIndices = {1}};
};

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Build_01
 * @tc.desc: DepthToSpaceModel1模型build测试
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Build_01, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Build_02
 * @tc.desc: DepthToSpaceModel2模型build测试
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Build_02, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel2 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Build_03
 * @tc.desc: DepthToSpaceModel1模型输入Tensor+1进行build测试
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Build_03, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
    graphArgs.operands = {depthToSpaceModel.input, depthToSpaceModel.input,
                          depthToSpaceModel.output, depthToSpaceModel.blockSize};
    graphArgs.inputIndices = {0, 1};
    graphArgs.outputIndices = {2};
    graphArgs.paramIndices = {3};
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Build_04
 * @tc.desc: DepthToSpaceModel1模型输出Tensor+1进行build测试
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Build_04, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
    graphArgs.operands = {depthToSpaceModel.input, depthToSpaceModel.output,
                          depthToSpaceModel.output, depthToSpaceModel.blockSize};
    graphArgs.inputIndices = {0};
    graphArgs.outputIndices = {1, 2};
    graphArgs.paramIndices = {3};
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Build_05
 * @tc.desc: DepthToSpaceModel1模型传入非法参数进行build测试
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Build_05, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;

    int8_t activationValue = OH_NN_FUSED_NONE;
    OHNNOperandTest activation = {OH_NN_INT8, OH_NN_ADD_ACTIVATIONTYPE, {}, &activationValue, sizeof(int8_t)};
    graphArgs.operands = {depthToSpaceModel.input, depthToSpaceModel.output,
                          depthToSpaceModel.blockSize, activation};
    graphArgs.paramIndices = {2, 3};
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_Finish_01
 * @tc.desc: 模型构图，未添加操作数
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_Finish_01, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    OHNNGraphArgs graphArgs;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, SingleModelBuildEndStep(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_Finish_02
 * @tc.desc: 模型构图，未设置输入输出
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_Finish_02, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
    graphArgs.specifyIO = false;
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_Finish_03
 * @tc.desc: 模型构图，设置输入输出，构图成功
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_Finish_03, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SetOperandValue_01
 * @tc.desc: 设置操作数值，操作数不存在
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SetOperandValue_01,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;

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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SetOperandValue_02
 * @tc.desc: 设置操作数值，buufer为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SetOperandValue_02,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;

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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SetOperandValue_03
 * @tc.desc: 设置操作数值，length为0
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SetOperandValue_03,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;

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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_01
 * @tc.desc: 设置输入输出，inputIndices为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_01,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, nullptr, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_02
 * @tc.desc: 设置输入输出，inputindices中data为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_02,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_03
 * @tc.desc: 设置输入输出，inputindices中data对应序号不存在
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_03,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_04
 * @tc.desc: 设置输入输出，inputindices中size为0
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_04,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_05
 * @tc.desc: 设置输入输出，outputindices为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_05,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, nullptr));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_06
 * @tc.desc: 设置输入输出，outputindices中data为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_06,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_07
 * @tc.desc: 设置输入输出，outputindices中data对应序号不存在
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_07,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_08
 * @tc.desc: 设置输入输出，outputindices中size为0
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_SpecifyInputsAndOutputs_08,
         Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_01
 * @tc.desc: 添加算子，paramindices为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_01, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_02
 * @tc.desc: 添加算子，paramindices中data为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_02, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    paramIndices.data = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                               &paramIndices, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_03
 * @tc.desc: 添加算子，paramindices中data对应序号不存在
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_03, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_04
 * @tc.desc: 添加算子，paramindices中size为0
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_04, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
    graphArgs.addOperation = false;
    graphArgs.specifyIO = false;
    graphArgs.build = false;
    EXPECT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    paramIndices.size = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddOperation(model, graphArgs.operationType,
                                                               &paramIndices, &inputIndices, &outputIndices));

    Free(model, nullptr, nullptr);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_05
 * @tc.desc: 添加算子，inputindices为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_05, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_06
 * @tc.desc: 添加算子，inputindices中data为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_06, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_07
 * @tc.desc: 添加算子，inputindices中data对应序号不存在
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_07, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_08
 * @tc.desc: 添加算子，inputindices中size为0
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_08, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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
 * @tc.number : SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_09
 * @tc.desc: 添加算子，outputindices为nullptr
 * @tc.type: FUNC
 */
HWTEST_F(DepthToSpaceTest, SUB_AI_NNRt_Func_North_DepthToSpace_Model_AddOperation_09, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    EXPECT_NE(nullptr, model);

    DepthToSpaceModel1 depthToSpaceModel;
    OHNNGraphArgs graphArgs = depthToSpaceModel.graphArgs;
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