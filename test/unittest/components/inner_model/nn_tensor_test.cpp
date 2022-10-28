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
#include <gmock/gmock.h>

#include "frameworks/native/validation.h"
#include "frameworks/native/nn_tensor.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Validation;

namespace NNRT {
namespace UnitTest {
class NnTensorTest : public testing::Test {
};

/**
 * @tc.name: nn_tensor_parse_dimensions_001
 * @tc.desc: Verify the success of the parse_dimensions function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_dimensions_001, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_parse_dimensions_002
 * @tc.desc: Verify the invalid dimensions of the parse_dimensions function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_dimensions_002, TestSize.Level1)
{
    OH_NN_Tensor tensor;
    tensor.dataType = OH_NN_FLOAT32;
    tensor.dimensionCount = 2;
    tensor.dimensions = nullptr;
    tensor.quantParam = nullptr;
    tensor.type = OH_NN_TENSOR;

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_parse_dimensions_003
 * @tc.desc: Verify the invalid shape tensor of the parse_dimensions function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_dimensions_003, TestSize.Level1)
{
    const int dim[2] = {2, -2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_parse_dimensions_004
 * @tc.desc: Verify the dynamic shape of the parse_dimensions function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_dimensions_004, TestSize.Level1)
{
    const int dim[2] = {2, -1};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_parse_dimensions_005
 * @tc.desc: Verify the dims out of bounds of the parse_dimensions function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_dimensions_005, TestSize.Level1)
{
    const int dim[3] = {1000000, 1000000, 10000000};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.BuildFromOHNNTensor(tensor));
}


/**
 * @tc.name: nn_tensor_parse_quant_params_001
 * @tc.desc: Verify the success of the parse_quant_params function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_quant_params_001, TestSize.Level1)
{
    const double scale = 1.0;
    const int32_t zeroPoint = 0;
    const uint32_t numBits = 8;
    const OH_NN_QuantParam quantParam = {1, &numBits, &scale, &zeroPoint};

    NNTensor nnTensor;
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, &quantParam, OH_NN_TENSOR};

    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_parse_quant_params_002
 * @tc.desc: Verify the invalid numbits of the parse_quant_params function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_quant_params_002, TestSize.Level1)
{
    const double scale = 1.0;
    const int32_t zeroPoint = 0;
    const uint32_t numBits = 16;
    const OH_NN_QuantParam quantParam = {1, &numBits, &scale, &zeroPoint};

    NNTensor nnTensor;
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, &quantParam, OH_NN_TENSOR};

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_parse_quant_params_004
 * @tc.desc: Verify the invalid scale of the parse_quant_params function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_quant_params_004, TestSize.Level1)
{
    const int32_t zeroPoint = 0;
    const uint32_t numBits = 8;
    const OH_NN_QuantParam quantParam = {1, &numBits, nullptr, &zeroPoint};

    NNTensor nnTensor;
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, &quantParam, OH_NN_TENSOR};

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_parse_quant_params_005
 * @tc.desc: Verify the invalid zeropoint of the parse_quant_params function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_parse_quant_params_005, TestSize.Level1)
{
    const double scale = 1.0;
    const uint32_t numBits = 8;
    const OH_NN_QuantParam quantParam = {1, &numBits, &scale, nullptr};

    NNTensor nnTensor;
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, &quantParam, OH_NN_TENSOR};

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_set_dimensions_001
 * @tc.desc: Verify the success of the set_dimensions function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_set_dimensions_001, TestSize.Level1)
{
    const int dim[2] = {2, -1};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};
    const std::vector<int32_t> dimensions = {2, 3};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.SetDimensions(dimensions));
}

/**
 * @tc.name: nn_tensor_set_dimensions_002
 * @tc.desc: Verify the dim out of bounds of the set_dimensions function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_set_dimensions_002, TestSize.Level1)
{
    const int dim[2] = {2, -1};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    const std::vector<int32_t> dimensions = {2, 3, 5};
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.SetDimensions(dimensions));
}

/**
 * @tc.name: nn_tensor_compare_attribute_001
 * @tc.desc: Verify the success of the CompareAttribute function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_compare_attribute_001, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    NNTensor expectTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
    expectTensor = std::move(nnTensor);
    EXPECT_EQ(true, nnTensor.CompareAttribute(nnTensor));
}

/**
 * @tc.name: nn_tensor_compare_attribute_002
 * @tc.desc: Verify the datatype not equal of the CompareAttribute function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_compare_attribute_002, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    NNTensor expectTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));

    const int dimExpect[2] = {2, 2};
    OH_NN_Tensor tensorExpect{OH_NN_INT32, 2, dimExpect, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, expectTensor.BuildFromOHNNTensor(tensorExpect));

    EXPECT_EQ(false, nnTensor.CompareAttribute(expectTensor));
}

/**
 * @tc.name: nn_tensor_compare_attribute_003
 * @tc.desc: Verify the dim size not equal of the CompareAttribute function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_compare_attribute_003, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    NNTensor expectTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));

    const int dimExpect[3] = {2, 2, 3};
    OH_NN_Tensor tensorExpect{OH_NN_FLOAT32, 3, dimExpect, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, expectTensor.BuildFromOHNNTensor(tensorExpect));

    EXPECT_EQ(false, nnTensor.CompareAttribute(expectTensor));
}

/**
 * @tc.name: nn_tensor_compare_attribute_004
 * @tc.desc: Verify the dim value not equal of the CompareAttribute function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_compare_attribute_004, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    NNTensor expectTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));

    const int dimExpect[2] = {2, 3};
    OH_NN_Tensor tensorExpect{OH_NN_FLOAT32, 2, dimExpect, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, expectTensor.BuildFromOHNNTensor(tensorExpect));

    EXPECT_EQ(false, nnTensor.CompareAttribute(expectTensor));
}

/**
 * @tc.name: nn_tensor_is_scalar_001
 * @tc.desc: Verify the success of the is_scalar function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_is_scalar_001, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
    EXPECT_EQ(false, nnTensor.IsScalar());
}

/**
 * @tc.name: nn_tensor_build_from_tensor_001
 * @tc.desc: Verify the success of the build_from_tensor function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_convert_to_io_tensor_001, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));

    int8_t* activationValue = new (std::nothrow) int8_t[1]{0};
    EXPECT_NE(nullptr, activationValue);

    // After SetBuffer, this memory is released by NNTensor
    nnTensor.SetBuffer(activationValue, sizeof(int8_t));
    IOTensor ioTensor;
    nnTensor.ConvertToIOTensor(ioTensor);
    EXPECT_EQ(sizeof(int8_t), ioTensor.length);
}

/**
 * @tc.name: nn_tensor_get_buffer_length_001
 * @tc.desc: Verify the success of the get_buffer_length function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_get_buffer_length_001, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
    int8_t* activationValue = new (std::nothrow) int8_t[1]{0};
    EXPECT_NE(nullptr, activationValue);

    // After SetBuffer, this memory is released by NNTensor
    nnTensor.SetBuffer(activationValue, sizeof(int8_t));
    size_t length = sizeof(int8_t);
    EXPECT_EQ(length, nnTensor.GetBufferLength());
}

/**
 * @tc.name: nn_tensor_get_format_001
 * @tc.desc: Verify the success of the get_format function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_get_format_001, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));
    OH_NN_Format format = OH_NN_FORMAT_NHWC;
    EXPECT_EQ(format, nnTensor.GetFormat());
}

/**
 * @tc.name: nn_tensor_get_name_001
 * @tc.desc: Verify the success of the get name function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_get_name_001, TestSize.Level1)
{
    NNTensor nnTensor;
    const std::string& name = "test";
    nnTensor.SetName(name);
    EXPECT_EQ(name, nnTensor.GetName());
}

/**
 * @tc.name: nn_tensor_get_quant_param_001
 * @tc.desc: Verify the success of the get_quant_param function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_get_quant_param_001, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));

    std::vector<QuantParam> quantParam = nnTensor.GetQuantParam();
    size_t quantSize = 0;
    EXPECT_EQ(quantSize, quantParam.size());
}

/**
 * @tc.name: nn_tensor_build_from_tensor_002
 * @tc.desc: Verify the invalid datatype value of the build_from_tensor function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_build_from_tensor_002, TestSize.Level1)
{
    const int dim[2] = {2, 2};

    int dataTypeTest = 13;
    OH_NN_DataType dataType = (OH_NN_DataType)dataTypeTest;
    OH_NN_Tensor tensor{dataType, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.BuildFromOHNNTensor(tensor));
}

/**
 * @tc.name: nn_tensor_convert_to_lite_graph_tensor_001
 * @tc.desc: Verify the success of the convert_to_lite_graph function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_convert_to_lite_graph_tensor_001, TestSize.Level1)
{
    const int dim[2] = {2, 2};
    OH_NN_Tensor tensor{OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));

    LiteGraphTensorPtr tensorPtr = {nullptr, DestroyLiteGraphTensor};
    EXPECT_NE(tensorPtr, nnTensor.ConvertToLiteGraphTensor());
}

/**
 * @tc.name: nn_tensor_convert_to_lite_graph_tensor_002
 * @tc.desc: Verify the success with quant of the convert_to_lite_graph function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_convert_to_lite_graph_tensor_002, TestSize.Level1)
{
    const int dim[2] = {2, 2};

    OH_NN_Tensor tensor;
    tensor.dataType = OH_NN_FLOAT32;
    tensor.dimensionCount = 2;
    tensor.dimensions = dim;
    const double scale = 1.0;
    const int32_t zeroPoint = 0;
    const uint32_t numBits = 8;
    const OH_NN_QuantParam quantParam = {1, &numBits, &scale, &zeroPoint};
    tensor.quantParam = &quantParam;
    tensor.type = OH_NN_TENSOR;

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.BuildFromOHNNTensor(tensor));

    LiteGraphTensorPtr tensorPtr = {nullptr, DestroyLiteGraphTensor};
    EXPECT_NE(tensorPtr, nnTensor.ConvertToLiteGraphTensor());
}

/**
 * @tc.name: nn_tensor_build_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_build_001, TestSize.Level1)
{
    OH_NN_DataType dataType = OH_NN_FLOAT32;
    const std::vector<int32_t> dimensions = {2, 2};
    const std::vector<QuantParam> quantParam = {{8, 1.0, 0}, {8, 1.0, 0}, {8, 1.0, 0}};
    OH_NN_TensorType type =  OH_NN_ADD_ACTIVATIONTYPE;

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_SUCCESS, nnTensor.Build(dataType, dimensions, quantParam, type));
}

/**
 * @tc.name: nn_tensor_build_002
 * @tc.desc: Verify the invalid datatype value of the build function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_build_002, TestSize.Level1)
{
    int dataTypeTest = 13;
    OH_NN_DataType dataType = (OH_NN_DataType)dataTypeTest;
    const std::vector<int32_t> dimensions = {2, 2};
    const std::vector<QuantParam> quantParam = {{8, 1.0, 0}, {8, 1.0, 0}, {8, 1.0, 0}};
    OH_NN_TensorType type =  OH_NN_ADD_ACTIVATIONTYPE;

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.Build(dataType, dimensions, quantParam, type));
}

/**
 * @tc.name: nn_tensor_build_003
 * @tc.desc: Verify the dynamic shape of the build function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_build_003, TestSize.Level1)
{
    OH_NN_DataType dataType = OH_NN_FLOAT32;
    const std::vector<int32_t> dimensions = {2, -2};
    const std::vector<QuantParam> quantParam = {{8, 1.0, 0}, {8, 1.0, 0}, {8, 1.0, 0}};
    OH_NN_TensorType type =  OH_NN_ADD_ACTIVATIONTYPE;

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.Build(dataType, dimensions, quantParam, type));
}

/**
 * @tc.name: nn_tensor_build_004
 * @tc.desc: Verify the invalid numbits of the build function
 * @tc.type: FUNC
 */
HWTEST_F(NnTensorTest, nn_tensor_build_004, TestSize.Level1)
{
    OH_NN_DataType dataType = OH_NN_FLOAT32;
    const std::vector<int32_t> dimensions = {2, 2};
    const std::vector<QuantParam> quantParam = {{2, 1.0, 0}, {2, 1.0, 0}, {2, 1.0, 0}};
    OH_NN_TensorType type =  OH_NN_ADD_ACTIVATIONTYPE;

    NNTensor nnTensor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, nnTensor.Build(dataType, dimensions, quantParam, type));
}
} // namespace UnitTest
} // namespace NNRT
