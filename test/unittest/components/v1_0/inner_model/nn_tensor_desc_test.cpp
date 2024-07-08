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

#include <gtest/gtest.h>

#include "validation.h"
#include "tensor_desc.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Validation;

namespace NNRT {
namespace UnitTest {
class NnTensorDescTest : public testing::Test {
};

/**
 * @tc.name: nn_get_datatype_001
 * @tc.desc: Verify the success of the GetDataType function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_datatype_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    OH_NN_DataType* testdatatype = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetDataType(testdatatype));
}

/**
 * @tc.name: nn_get_datatype_002
 * @tc.desc: Verify the success of the GetDataType function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_datatype_002, TestSize.Level1)
{
    TensorDesc tensordesc;
    OH_NN_DataType testdatatype = OH_NN_BOOL;
    EXPECT_EQ(OH_NN_SUCCESS, tensordesc.GetDataType(&testdatatype));
}

/**
 * @tc.name: nn_set_datatype_001
 * @tc.desc: Verify the success of the SetDataType function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_set_datatype_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    int dataTypeTest = 13;
    OH_NN_DataType testdataType = (OH_NN_DataType)dataTypeTest;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.SetDataType(testdataType));
}

/**
 * @tc.name: nn_get_format_001
 * @tc.desc: Verify the success of the GetFormat function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_format_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    OH_NN_Format* testformat = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetFormat(testformat));
}

/**
 * @tc.name: nn_set_format_001
 * @tc.desc: Verify the success of the SetFormat function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_set_format_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    OH_NN_Format testformat = OH_NN_FORMAT_NCHW;
    EXPECT_EQ(OH_NN_SUCCESS, tensordesc.SetFormat(testformat));
}

/**
 * @tc.name: nn_get_shape_001
 * @tc.desc: Verify the success of the GetShape function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_shape_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    int32_t** testshape = nullptr;
    size_t* testshapeNum = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetShape(testshape, testshapeNum));
}

/**
 * @tc.name: nn_get_shape_002
 * @tc.desc: Verify the success of the GetShape function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_shape_002, TestSize.Level1)
{
    TensorDesc tensordesc;
    int32_t shapDim[2] = {3, 3};
    int32_t* ptr = shapDim;
    int32_t** testshape = &ptr;
    size_t* testshapeNum = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetShape(testshape, testshapeNum));
}

/**
 * @tc.name: nn_get_shape_002
 * @tc.desc: Verify the success of the GetShape function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_shape_003, TestSize.Level1)
{
    TensorDesc tensordesc;
    int32_t** testshape = new int32_t*[1];
    testshape[0] = nullptr;
    size_t* testshapenum = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetShape(testshape, testshapenum));
}

/**
 * @tc.name: nn_set_shape_001
 * @tc.desc: Verify the success of the SetShape function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_set_shape_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    const int32_t* testshape = nullptr;
    size_t testshapenum = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.SetShape(testshape, testshapenum));
}

/**
 * @tc.name: nn_set_shape_002
 * @tc.desc: Verify the success of the SetShape function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_set_shape_002, TestSize.Level1)
{
    TensorDesc tensordesc;
    const int32_t testShape[] = { 2, 3, 5 };
    size_t shapenum = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.SetShape(testShape, shapenum));
}

/**
 * @tc.name: nn_get_elementnum_001
 * @tc.desc: Verify the success of the GetElementNum function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_elementnum_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    size_t* testelementNum = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetElementNum(testelementNum));
}

/**
 * @tc.name: nn_get_bytesize_001
 * @tc.desc: Verify the success of the GetByteSize function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_bytesize_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    size_t* testbytesize = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetByteSize(testbytesize));
}

/**
 * @tc.name: nn_get_bytesize_002
 * @tc.desc: Verify the success of the GetByteSize function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_bytesize_002, TestSize.Level1)
{
    TensorDesc tensordesc;
    size_t* testbytesize = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetByteSize(testbytesize));
}

/**
 * @tc.name: nn_set_name_001
 * @tc.desc: Verify the success of the SetName function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_set_name_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    const char* testsetname = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.SetName(testsetname));
}

/**
 * @tc.name: nn_get_name_001
 * @tc.desc: Verify the success of the GetName function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorDescTest, nn_get_name_001, TestSize.Level1)
{
    TensorDesc tensordesc;
    const char** testgetname = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, tensordesc.GetName(testgetname));
}
} // namespace UnitTest
} // namespace NNRT
