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
#include <string>

#include "nncore_utils.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Test;
using namespace OHOS::HDI::Nnrt::V2_0;
class TensorDescTest : public testing::Test {};


/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescCreate_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescCreate_0100
 * @tc.desc   创建TensorDesc
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescCreate_0100, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0100
 * @tc.desc   销毁TensorDesc，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0100, Function | MediumTest | Level1)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_Destroy(nullptr));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0200
 * @tc.desc   销毁TensorDesc，*TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = nullptr;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0300
 * @tc.desc   销毁TensorDesc
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescDestroy_0300, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0100
 * @tc.desc   设置TensorDesc数据类型，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0100, Function | MediumTest | Level1)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_SetDataType(nullptr, OH_NN_UNKNOWN));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0200
 * @tc.desc   设置TensorDesc数据类型，遍历DataType
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_UNKNOWN));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_FLOAT64));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_BOOL));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0300
 * @tc.desc   设置TensorDesc数据类型，DataType不合法
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetDataType_0300, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    int dataType = static_cast<int>(OH_NN_FLOAT64)+1;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_SetDataType(tensorDesc, static_cast<OH_NN_DataType>(dataType)));
    dataType = static_cast<int>(OH_NN_UNKNOWN)-1;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_SetDataType(tensorDesc, static_cast<OH_NN_DataType>(dataType)));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetDataType_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetDataType_0100
 * @tc.desc   获取TensorDesc数据类型，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetDataType_0100, Function | MediumTest | Level1)
{
    OH_NN_DataType dataType;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_GetDataType(nullptr, &dataType));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetDataType_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetDataType_0200
 * @tc.desc   获取TensorDesc数据类型，未设置DataType
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetDataType_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    OH_NN_DataType dataType = OH_NN_UNKNOWN;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetDataType(tensorDesc, &dataType));
    ASSERT_EQ(OH_NN_UNKNOWN, dataType);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetShape_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetShape_0100
 * @tc.desc   设置TensorDescShape，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetShape_0100, Function | MediumTest | Level1)
{
    int32_t inputDims[4] = {1, 2, 2, 3};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_SetShape(nullptr, inputDims, 4));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetShape_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetShape_0200
 * @tc.desc   设置TensorDescShape，shape为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetShape_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_SetShape(tensorDesc, nullptr, 0));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0100
 * @tc.desc   获取TensorDescShape，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0100, Function | MediumTest | Level1)
{
    size_t shapeLength = 0;
    int32_t* shape = nullptr;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_GetShape(nullptr, &shape, &shapeLength));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0200
 * @tc.desc   获取TensorDescShape，未设置shape
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    size_t shapeLength = 0;
    int32_t* shape = nullptr;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetShape(tensorDesc, &shape, &shapeLength));
    ASSERT_EQ(nullptr, shape);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0300
 * @tc.desc   获取TensorDescShape，合法设置返回成功
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetShape_0300, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    int32_t inputDims[4] = {1, 2, 2, 3};
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(tensorDesc, inputDims, 4));
    size_t shapeLength = 0;
    int32_t* shape = nullptr;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetShape(tensorDesc, &shape, &shapeLength));
    ASSERT_EQ(*inputDims, *shape);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetFormat_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetFormat_0100
 * @tc.desc   设置TensorDescFormat，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetFormat_0100, Function | MediumTest | Level1)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_SetFormat(nullptr, OH_NN_FORMAT_NONE));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetFormat_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetFormat_0200
 * @tc.desc   设置TensorDescFormat，遍历设置format
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetFormat_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NONE));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NCHW));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NHWC));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_ND));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0100
 * @tc.desc   获取TensorDescFormat，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0100, Function | MediumTest | Level1)
{
    OH_NN_Format format = OH_NN_FORMAT_NONE;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_GetFormat(nullptr, &format));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0200
 * @tc.desc   获取TensorDescFormat，未设置Format
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    OH_NN_Format format = OH_NN_FORMAT_NONE;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetFormat(tensorDesc, &format));
    ASSERT_EQ(OH_NN_FORMAT_NONE, format);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0300
 * @tc.desc   获取TensorDescFormat，合法获取，返回成功
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetFormat_0300, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetFormat(tensorDesc, OH_NN_FORMAT_NCHW));
    OH_NN_Format format = OH_NN_FORMAT_NONE;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetFormat(tensorDesc, &format));
    ASSERT_EQ(OH_NN_FORMAT_NCHW, format);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0100
 * @tc.desc   获取TensorDescElementCount，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0100, Function | MediumTest | Level1)
{
    size_t elementCount = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_GetElementCount(nullptr, &elementCount));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0200
 * @tc.desc   获取TensorDescElementCount，合理设置返回正确
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    int32_t inputDims[4] = {1, 2, 2, 3};
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(tensorDesc, inputDims, 4));
    size_t elementCount = 0;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetElementCount(tensorDesc, &elementCount));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0300
 * @tc.desc   获取TensorDescElementCount，不设置，返回错误
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetElementCount_0300, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    size_t elementCount = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_GetElementCount(tensorDesc, &elementCount));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0100
 * @tc.desc   获取TensorDescByteSize，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0100, Function | MediumTest | Level1)
{
    size_t byteSize = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_GetByteSize(nullptr, &byteSize));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0200
 * @tc.desc   获取TensorDescByteSize，合理设置返回正确
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    int32_t inputDims[4] = {1, 2, 2, 3};
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetDataType(tensorDesc, OH_NN_FLOAT32));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetShape(tensorDesc, inputDims, 4));
    size_t byteSize = 0;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetByteSize(tensorDesc, &byteSize));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0300
 * @tc.desc   获取TensorDescByteSize，不设置，返回错误
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetByteSize_0300, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    size_t byteSize = 0;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_GetByteSize(tensorDesc, &byteSize));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetName_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetName_0100
 * @tc.desc   设置TensorDescName，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetName_0100, Function | MediumTest | Level1)
{
    const char* name = "tensorDesc";
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_SetName(nullptr, name));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescSetName_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescSetName_0200
 * @tc.desc   设置TensorDescName，name为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescSetName_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_SetName(tensorDesc, nullptr));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0100
 * @tc.desc   获取TensorDescName，TensorDesc为空
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0100, Function | MediumTest | Level1)
{
    const char* name = nullptr;
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNTensorDesc_GetName(nullptr, &name));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0200
 * @tc.desc   获取TensorDescName，合理设置返回正确
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0200, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    const char* nameIn = "tensorDesc";
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_SetName(tensorDesc, nameIn));
    const char* nameOut = nullptr;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetName(tensorDesc, &nameOut));
    ASSERT_EQ(*nameIn, *nameOut);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0300
 * @tc.desc   获取TensorDescName，不设置，返回错误
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(TensorDescTest, SUB_AI_NNRt_Core_Func_North_TensorDescGetName_0300, Function | MediumTest | Level1)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    ASSERT_NE(nullptr, tensorDesc);
    const char* nameOut = nullptr;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_GetName(tensorDesc, &nameOut));
    std::string name(nameOut);
    std::string empty("");
    ASSERT_EQ(empty, name);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNTensorDesc_Destroy(&tensorDesc));
}

