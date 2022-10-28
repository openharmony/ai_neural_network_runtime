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

#include "frameworks/native/ops/quant_dtype_cast_builder.h"

#include "ops_test.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Ops;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class QuantDTypeCastBuilderTest : public OpsTest {
public:
    void SetUp() override;
    void TearDown() override;

protected:
    void SaveSrcTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);
    void SaveDstTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
        const OH_NN_QuantParam* quantParam, OH_NN_TensorType type);

protected:
    QuantDTypeCastBuilder m_builder;
    std::vector<uint32_t> m_inputs {0};
    std::vector<uint32_t> m_outputs {1};
    std::vector<uint32_t> m_params {2, 3};
    std::vector<int32_t> m_dim {3, 3};
    std::vector<int32_t> m_paramDim {};
};

void QuantDTypeCastBuilderTest::SetUp() {}

void QuantDTypeCastBuilderTest::TearDown() {}

void QuantDTypeCastBuilderTest::SaveSrcTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> srcTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t *srcValue = new (std::nothrow) int64_t(1);
    EXPECT_NE(nullptr, srcValue);
    srcTensor->SetBuffer(srcValue, sizeof(int64_t));
    m_allTensors.emplace_back(srcTensor);
}

void QuantDTypeCastBuilderTest::SaveDstTensor(OH_NN_DataType dataType, const std::vector<int32_t> &dim,
    const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> dstTensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t *dstValue = new (std::nothrow) int64_t(1);
    EXPECT_NE(nullptr, dstValue);
    dstTensor->SetBuffer(dstValue, sizeof(int64_t));
    m_allTensors.emplace_back(dstTensor);
}

/**
 * @tc.name: quantdtypecast_build_001
 * @tc.desc: Provide normal input, output, and parameters to verify the normal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_001, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveSrcTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    SaveDstTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: quantdtypecast_build_002
 * @tc.desc: Call Build func twice to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveSrcTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    SaveDstTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: quantdtypecast_build_003
 * @tc.desc: Provide one more than normal input to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_003, TestSize.Level0)
{
    m_inputs = {0, 1};
    m_outputs = {2};
    m_params = {3, 4};

    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveSrcTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    SaveDstTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: quantdtypecast_build_004
 * @tc.desc: Provide one more than normal output to verify the abnormal behavior of the Build function
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_004, TestSize.Level0)
{
    m_outputs = {1, 2};
    m_params = {3, 4};

    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveSrcTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    SaveDstTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: quantdtypecast_build_005
 * @tc.desc: Verify that the build function return a failed message with null allTensor
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_005, TestSize.Level0)
{
    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputs, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: quantdtypecast_build_006
 * @tc.desc: Verify that the build function return a failed message without output tensor
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_006, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: quantdtypecast_build_007
 * @tc.desc: Verify that the build function return a failed message with invalided src's dataType
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_007, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveDstTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);

    std::shared_ptr<NNTensor> srcTensor = TransToNNTensor(OH_NN_INT32, m_paramDim,
        nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    int32_t srcValue = 1;
    srcTensor->SetBuffer(&srcValue, sizeof(srcValue));
    m_allTensors.emplace_back(srcTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    srcTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: quantdtypecast_build_008
 * @tc.desc: Verify that the build function return a failed message with invalided dst's dataType
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_008, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveSrcTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);


    std::shared_ptr<NNTensor> dstTensor = TransToNNTensor(OH_NN_INT32, m_paramDim,
        nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);
    int32_t dstValue = 1;
    dstTensor->SetBuffer(&dstValue, sizeof(dstValue));
    m_allTensors.emplace_back(dstTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    dstTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: quantdtypecast_build_009
 * @tc.desc: Verify that the build function return a failed message with invalided parameter
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_009, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveSrcTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    SaveDstTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_REDUCE_ALL_KEEP_DIMS);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: quantdtypecast_build_010
 * @tc.desc: Verify that the build function return a failed message with empty src's buffer
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_010, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveDstTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);

    std::shared_ptr<NNTensor> srcTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    m_allTensors.emplace_back(srcTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    srcTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: quantdtypecast_build_011
 * @tc.desc: Verify that the build function return a failed message with empty dst's buffer
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_build_011, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveSrcTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);


    std::shared_ptr<NNTensor> dstTensor = TransToNNTensor(OH_NN_INT64, m_paramDim,
        nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);
    m_allTensors.emplace_back(dstTensor);

    OH_NN_ReturnCode ret = m_builder.Build(m_params, m_inputsIndex, m_outputs, m_allTensors);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    dstTensor->SetBuffer(nullptr, 0);
}

/**
 * @tc.name: quantdtypecast_get_primitive_001
 * @tc.desc: Verify the GetPrimitive function return nullptr
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_get_primitive_001, TestSize.Level0)
{
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_EQ(primitive, expectPrimitive);
}

/**
 * @tc.name: quantdtypecast_get_primitive_002
 * @tc.desc: Verify the normal params return behavior of the getprimitive function
 * @tc.type: FUNC
 */
HWTEST_F(QuantDTypeCastBuilderTest, quantdtypecast_get_primitive_002, TestSize.Level0)
{
    SaveInputTensor(m_inputs, OH_NN_INT8, m_dim, nullptr);
    SaveOutputTensor(m_outputs, OH_NN_INT8, m_dim, nullptr);
    SaveSrcTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_SRC_T);
    SaveDstTensor(OH_NN_INT64, m_paramDim, nullptr, OH_NN_QUANT_DTYPE_CAST_DST_T);

    EXPECT_EQ(OH_NN_SUCCESS, m_builder.Build(m_params, m_inputsIndex, m_outputsIndex, m_allTensors));
    LiteGraphTensorPtr primitive = m_builder.GetPrimitive();
    LiteGraphTensorPtr expectPrimitive = {nullptr, DestroyLiteGraphPrimitive};
    EXPECT_NE(primitive, expectPrimitive);

    int64_t srcValue = 1;
    int64_t dstValue = 1;
    auto srcReturn = mindspore::lite::MindIR_QuantDTypeCast_GetSrcT(primitive.get());
    EXPECT_EQ(srcReturn, srcValue);
    auto dstReturn = mindspore::lite::MindIR_QuantDTypeCast_GetDstT(primitive.get());
    EXPECT_EQ(dstReturn, dstValue);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS