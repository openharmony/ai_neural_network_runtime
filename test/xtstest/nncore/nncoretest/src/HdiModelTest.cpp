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
#include <cmath>
#include <cstdio>
#include <vector>
#include <thread>

#include "nncore_utils.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Test;
using namespace OHOS::HDI::Nnrt::V2_0;

namespace {

class ModelTest : public testing::Test {
protected:
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
};

void BuildAddTopKGraph(OH_NNModel *model)
{
    AddTopKModel addTopKModel;
    OHNNGraphArgsMulti graphArgsMulti = addTopKModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildMultiOpGraph(model, graphArgsMulti));
}

void BuildModel(OH_NNModel *model, const OHNNGraphArgs &graphArgs)
{
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
}

} // namespace

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_CreateQuantParam_0100
 * @tc.number SUB_AI_NNRt_Func_North_Model_CreateQuantParam_0100
 * @tc.desc   创建量化参数
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_CreateQuantParam_0100, Function | MediumTest | Level0)
{
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    ASSERT_NE(nullptr, quantParam);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_DestroyQuantParam_0100
 * @tc.number SUB_AI_NNRt_Func_North_Model_DestroyQuantParam_0100
 * @tc.desc   释放量化参数,*quantParam为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_DestroyQuantParam_0100, Function | MediumTest | Level0)
{
    NN_QuantParam* quantParam = nullptr;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNQuantParam_Destroy(&quantParam));
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_DestroyQuantParam_0200
 * @tc.number SUB_AI_NNRt_Func_North_Model_DestroyQuantParam_0200
 * @tc.desc   释放量化参数,quantParam为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_DestroyQuantParam_0200, Function | MediumTest | Level0)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNQuantParam_Destroy(nullptr));
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0100
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0100
 * @tc.desc   设置量化参数，参数为空，返回失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0100, Function | MediumTest | Level0)
{
    double scales = 0.2;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNQuantParam_SetScales(nullptr, &scales, 1));
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0200
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0200
 * @tc.desc   设置量化参数，参数合法,count=1,参数的channel=1,调用成功
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0200, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    double scales = 0.2;
    int32_t zeroPoints = 0;
    uint32_t numBits = 8;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetScales(quantParam, &scales, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetZeroPoints(quantParam, &zeroPoints, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetNumBits(quantParam, &numBits, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_SetTensorQuantParams(model, 0, quantParam));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0300
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0300
 * @tc.desc   设置量化参数，count=1,channel=2,Per-Layer模式，设置成功
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0300, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 2, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    double scales = 0.2;
    int32_t zeroPoints = 0;
    uint32_t numBits = 8;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetScales(quantParam, &scales, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetZeroPoints(quantParam, &zeroPoints, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetNumBits(quantParam, &numBits, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_SetTensorQuantParams(model, 0, quantParam));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0400
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0400
 * @tc.desc   设置量化参数，参数合法,count=2,参数的channel=1,调用失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0400, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    double scales = 0.2;
    int32_t zeroPoints = 0;
    uint32_t numBits = 8;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetScales(quantParam, &scales, 2));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetZeroPoints(quantParam, &zeroPoints, 2));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetNumBits(quantParam, &numBits, 2));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorQuantParams(model, 0, quantParam));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0500
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0500
 * @tc.desc   设置量化参数，参数合法,参数count不一致,调用失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0500, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    double scales = 0.2;
    int32_t zeroPoints = 0;
    uint32_t numBits = 8;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetScales(quantParam, &scales, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetZeroPoints(quantParam, &zeroPoints, 2));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetNumBits(quantParam, &numBits, 1));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorQuantParams(model, 0, quantParam));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0600
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0600
 * @tc.desc   设置量化参数，参数合法,不设置zeroPoints,调用失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0600, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    double scales = 0.2;
    uint32_t numBits = 8;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetScales(quantParam, &scales, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetNumBits(quantParam, &numBits, 1));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorQuantParams(model, 0, quantParam));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0700
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0700
 * @tc.desc   设置量化参数，参数合法,不设置numBits,调用失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0700, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    double scales = 0.2;
    int32_t zeroPoints = 0;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetScales(quantParam, &scales, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetZeroPoints(quantParam, &zeroPoints, 1));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorQuantParams(model, 0, quantParam));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0800
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0800
 * @tc.desc   设置量化参数，参数合法,不设置scales,调用失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0800, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    int32_t zeroPoints = 0;
    uint32_t numBits = 8;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetZeroPoints(quantParam, &zeroPoints, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetNumBits(quantParam, &numBits, 1));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorQuantParams(model, 0, quantParam));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetQuantParam_0900
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetQuantParam_0900
 * @tc.desc   设置量化参数，参数合法,tensor下标等于当前tensor个数,返回失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetQuantParam_0900, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    NN_QuantParam* quantParam = OH_NNQuantParam_Create();
    double scales = 0.2;
    int32_t zeroPoints = 0;
    uint32_t numBits = 8;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetScales(quantParam, &scales, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetZeroPoints(quantParam, &zeroPoints, 1));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_SetNumBits(quantParam, &numBits, 1));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorQuantParams(model, 1, quantParam));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNQuantParam_Destroy(&quantParam));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_AddTensorToModel_0100
 * @tc.number SUB_AI_NNRt_Func_North_Model_AddTensorToModel_0100
 * @tc.desc   添加tensor到模型，model为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_AddTensorToModel_0100, Function | MediumTest | Level0)
{
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddTensorToModel(nullptr, tensorDesc));
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_AddTensorToModel_0200
 * @tc.number SUB_AI_NNRt_Func_North_Model_AddTensorToModel_0200
 * @tc.desc   添加tensor到模型，tensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_AddTensorToModel_0200, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_AddTensorToModel(model, nullptr));
    OH_NNModel_Destroy(&model);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetTensorData_0100
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetTensorData_0100
 * @tc.desc   设置操作数值，model为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetTensorData_0100, Function | MediumTest | Level0)
{
    int8_t activationValue{0};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorData(nullptr, 0, (void *)&activationValue, sizeof(int8_t)));
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetTensorData_0200
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetTensorData_0200
 * @tc.desc   设置操作数值，databuffer为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetTensorData_0200, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorData(model, 0, nullptr, sizeof(int8_t)));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetTensorData_0300
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetTensorData_0300
 * @tc.desc   设置操作数值，length = 0
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetTensorData_0300, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    int8_t activationValue{0};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorData(model, 0, (void *)&activationValue, 0));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetTensorData_0400
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetTensorData_0400
 * @tc.desc   设置操作数值，index大于tensor个数
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetTensorData_0400, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    int8_t activationValue{0};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorData(model, 1000, (void *)&activationValue, sizeof(int8_t)));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetTensorType_0100
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetTensorType_0100
 * @tc.desc   设置tensortype，model为nullptr
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetTensorType_0100, Function | MediumTest | Level0)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorType(nullptr, 0, OH_NN_TENSOR));
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetTensorType_0200
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetTensorType_0200
 * @tc.desc   设置tensortype，index大于tensor个数
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetTensorType_0200, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNModel_SetTensorType(model, 1000, OH_NN_TENSOR));
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}

/**
 * @tc.name   SUB_AI_NNRt_Func_North_Model_SetTensorType_0300
 * @tc.number SUB_AI_NNRt_Func_North_Model_SetTensorType_0300
 * @tc.desc   设置tensortype，遍历设置tensorType
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL0
 */
HWTEST_F(ModelTest, SUB_AI_NNRt_Func_North_Model_SetTensorType_0300, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    int32_t inputDims[4] = {1, 1, 2, 3};
    NN_TensorDesc* tensorDesc = createTensorDesc(inputDims, 4, OH_NN_FLOAT32, OH_NN_FORMAT_NCHW);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_AddTensorToModel(model, tensorDesc));
    for (int tensorType = 0; tensorType < 77; tensorType++) {
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_SetTensorType(model, 0, static_cast<OH_NN_TensorType>(tensorType)));
    }
    OH_NNModel_Destroy(&model);
    OH_NNTensorDesc_Destroy(&tensorDesc);
}
