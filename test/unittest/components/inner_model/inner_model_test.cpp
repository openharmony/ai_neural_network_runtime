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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "common/utils.h"
#include "common/log.h"
#include "frameworks/native/nn_tensor.h"
#include "frameworks/native/inner_model.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;

namespace NNRT {
namespace UnitTest {
class InnerModelTest : public testing::Test {
public:
    void SetLiteGraph(mindspore::lite::LiteGraph* liteGraph);
    void SetTensors();
    void SetIndices();

public:
    InnerModel m_innerModelTest;

    std::vector<int32_t> m_dimInput{3, 3};
    std::vector<int32_t> m_dimOutput{3, 3};
    std::vector<uint32_t> m_inputIndices{0};
    std::vector<uint32_t> m_outputIndices{1};

    OH_NN_OperationType m_opType{OH_NN_OPS_ADD};

    OH_NN_UInt32Array m_inputs;
    OH_NN_UInt32Array m_outputs;
    OH_NN_UInt32Array m_params;

    uint32_t m_paramIndexs[1]{3};
    uint32_t m_inputIndexs[2]{0, 1};
    uint32_t m_outputIndexs[1]{2};
};

void InnerModelTest::SetLiteGraph(mindspore::lite::LiteGraph* liteGraph)
{
    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_ = m_inputIndices;
    liteGraph->output_indices_ = m_outputIndices;

    const std::vector<mindspore::lite::QuantParam> quant_params {};

    for (size_t indexInput = 0; indexInput < liteGraph->input_indices_.size(); ++indexInput) {
        const std::vector<uint8_t> data(36, 1);
        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create(liteGraph->name_,
            mindspore::lite::DATA_TYPE_FLOAT32, m_dimInput, mindspore::lite::FORMAT_NCHW, data, quant_params));
    }

    for (size_t indexOutput = 0; indexOutput < liteGraph->output_indices_.size(); ++indexOutput) {
        const std::vector<uint8_t> dataOut(36, 1);
        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create(liteGraph->name_,
            mindspore::lite::DATA_TYPE_FLOAT32, m_dimOutput, mindspore::lite::FORMAT_NCHW, dataOut, quant_params));
    }
}

void InnerModelTest::SetTensors()
{
    const int dim[2] = {2, 2};
    const OH_NN_Tensor& tensor = {OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));

    const OH_NN_Tensor& tensorParam = {OH_NN_INT8, 0, nullptr, nullptr, OH_NN_ADD_ACTIVATIONTYPE};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensorParam));
}

void InnerModelTest::SetIndices()
{
    m_params.data = m_paramIndexs;
    m_params.size = sizeof(m_paramIndexs) / sizeof(uint32_t);

    m_inputs.data = m_inputIndexs;
    m_inputs.size = sizeof(m_inputIndexs) / sizeof(uint32_t);

    m_outputs.data = m_outputIndexs;
    m_outputs.size = sizeof(m_outputIndexs) / sizeof(uint32_t);
}

/**
 * @tc.name: inner_model_construct_nntensor_from_litegraph_001
 * @tc.desc: Verify the input_indices is empty of the construct_nntensor_from_litegraph function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_construct_nntensor_from_litegraph_001, TestSize.Level1)
{
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    m_inputIndices = {};

    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.BuildFromLiteGraph(liteGraph));
    mindspore::lite::MindIR_LiteGraph_Destroy(&liteGraph);
}

/**
 * @tc.name: inner_model_construct_nntensor_from_litegraph_002
 * @tc.desc: Verify the input_indices is out of bounds of the construct_nntensor_from_litegraph function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_construct_nntensor_from_litegraph_002, TestSize.Level1)
{
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    m_inputIndices = {6};

    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.BuildFromLiteGraph(liteGraph));
    mindspore::lite::MindIR_LiteGraph_Destroy(&liteGraph);
}

/**
 * @tc.name: inner_model_construct_nntensor_from_litegraph_003
 * @tc.desc: Verify the success of the construct_nntensor_from_litegraph function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_construct_nntensor_from_litegraph_003, TestSize.Level1)
{
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);

    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.BuildFromLiteGraph(liteGraph));
}

/**
 * @tc.name: inner_model_construct_nntensor_from_litegraph_004
 * @tc.desc: Verify the nntensor build failed nullptr return of the construct_nntensor_from_litegraph function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_construct_nntensor_from_litegraph_004, TestSize.Level1)
{
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    m_dimInput = {3, -3};

    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_NULL_PTR, m_innerModelTest.BuildFromLiteGraph(liteGraph));
    mindspore::lite::MindIR_LiteGraph_Destroy(&liteGraph);
}

/**
 * @tc.name: inner_model_construct_nntensor_from_litegraph_005
 * @tc.desc: Verify the output indices out of bounds of the construct_nntensor_from_litegraph function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_construct_nntensor_from_litegraph_005, TestSize.Level1)
{
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    m_outputIndices = {6};

    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.BuildFromLiteGraph(liteGraph));
    mindspore::lite::MindIR_LiteGraph_Destroy(&liteGraph);
}

/**
 * @tc.name: inner_model_build_from_lite_graph_001
 * @tc.desc: Verify the litegraph is nullptr of the build_from_lite_graph function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_build_from_lite_graph_001, TestSize.Level1)
{
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.BuildFromLiteGraph(nullptr));
}

/**
 * @tc.name: inner_model_build_from_lite_graph_002
 * @tc.desc: Verify the buildfromlitegraph twice forbidden of the build_from_lite_graph function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_build_from_lite_graph_002, TestSize.Level1)
{
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);

    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.BuildFromLiteGraph(liteGraph));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.BuildFromLiteGraph(liteGraph));
}

/**
 * @tc.name: inner_model_build_from_lite_graph_003
 * @tc.desc: Verify the litegraph->alltensors is empty of the build_from_lite_graph function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_build_from_lite_graph_003, TestSize.Level1)
{
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_ = {0};
    liteGraph->output_indices_ = {1};

    const int32_t dimInput[2] = {2, 2};
    const OH_NN_Tensor& tensor = {OH_NN_INT8, 2, dimInput, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));

    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.BuildFromLiteGraph(liteGraph));
    mindspore::lite::MindIR_LiteGraph_Destroy(&liteGraph);
}


/**
 * @tc.name: inner_model_add_tensor_001
 * @tc.desc: Verify the success of the addtensor function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_tensor_001, TestSize.Level1)
{
    const int32_t dimInput[2] = {2, 2};
    const OH_NN_Tensor& tensor = {OH_NN_INT8, 2, dimInput, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
}

/**
 * @tc.name: inner_model_add_tensor_002
 * @tc.desc: Verify the addtensor after buildfromlitegraph of the addtensor function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_tensor_002, TestSize.Level1)
{
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.BuildFromLiteGraph(liteGraph));

    const int32_t dimInput[2] = {2, 2};
    const OH_NN_Tensor& tensor = {OH_NN_INT8, 2, dimInput, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.AddTensor(tensor));
}

/**
 * @tc.name: inner_model_add_tensor_003
 * @tc.desc: Verify the buildfromnntensor failed of the addtensor function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_tensor_003, TestSize.Level1)
{
    const int32_t dimInput[2] = {2, -2};
    const OH_NN_Tensor& tensor = {OH_NN_INT8, 2, dimInput, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddTensor(tensor));
}


/**
 * @tc.name: inner_model_set_tensor_value_001
 * @tc.desc: Verify the success of the set_tensor_value function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_set_tensor_value_001, TestSize.Level1)
{
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));
}

/**
 * @tc.name: inner_model_set_tensor_value_002
 * @tc.desc: Verify the index out of bounds of the set_tensor_value function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_set_tensor_value_002, TestSize.Level1)
{
    SetTensors();

    uint32_t index = 6;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));
}

/**
 * @tc.name: inner_model_set_tensor_value_003
 * @tc.desc: Verify the buffer value is nullptr of the set_tensor_value function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_set_tensor_value_003, TestSize.Level1)
{
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       nullptr, sizeof(activation)));
}

/**
 * @tc.name: inner_model_set_tensor_value_004
 * @tc.desc: Verify the length invalid of the set_tensor_value function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_set_tensor_value_004, TestSize.Level1)
{
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), 0));
}

/**
 * @tc.name: inner_model_set_tensor_value_005
 * @tc.desc: Verify the after buildgraph of the set_tensor_value function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_set_tensor_value_005, TestSize.Level1)
{
    uint32_t index = 3;
    const int8_t activation = 0;

    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.BuildFromLiteGraph(liteGraph));

    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));
}

/**
 * @tc.name: inner_model_set_tensor_value_006
 * @tc.desc: Verify the set value twice of the set_tensor_value function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_set_tensor_value_006, TestSize.Level1)
{
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));
}

/**
 * @tc.name: inner_model_set_tensor_value_007
 * @tc.desc: Verify the tensor dynamicShape of the set_tensor_value function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_set_tensor_value_007, TestSize.Level1)
{
    const int32_t dimInput[2] = {2, -1};
    const OH_NN_Tensor& tensor = {OH_NN_FLOAT32, 2, dimInput, nullptr, OH_NN_TENSOR};

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    const OH_NN_Tensor& tensorParam = {OH_NN_INT8, 0, nullptr, nullptr, OH_NN_ADD_ACTIVATIONTYPE};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensorParam));

    uint32_t index = 0;
    float x[4] = {0, 1, 2, 3};
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.SetTensorValue(index,
       x, sizeof(x)- 1));
}

/**
 * @tc.name: inner_model_add_operation_001
 * @tc.desc: Verify the success of the addoperation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_001, TestSize.Level1)
{
    SetIndices();

    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_002
 * @tc.desc: Verify the after buildgraph of the addtensor function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_002, TestSize.Level1)
{
    OH_NN_OperationType m_opType = OH_NN_OPS_ADD;
    OH_NN_UInt32Array m_inputs;
    OH_NN_UInt32Array m_outputs;
    OH_NN_UInt32Array m_params;

    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.BuildFromLiteGraph(liteGraph));

    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs,
        m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_003
 * @tc.desc: Verify the without set buffer of the addtensor function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_003, TestSize.Level1)
{
    SetIndices();
    SetTensors();

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_004
 * @tc.desc: Verify the output indices equal to input indices  of the addtensor function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_004, TestSize.Level1)
{
    m_outputIndexs[0] = 0;

    SetIndices();
    SetTensors();

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_005
 * @tc.desc: Verify the optype invalid of the addtensor function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_005, TestSize.Level1)
{
    m_opType = OH_NN_OperationType(99);

    SetIndices();
    SetTensors();

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_006
 * @tc.desc: Verify the input indices out of bounds of the addoperation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_006, TestSize.Level1)
{
    m_inputIndexs[1] = 6;

    SetIndices();
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_007
 * @tc.desc: Verify the param indices out of bounds of the addoperation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_007, TestSize.Level1)
{
    m_paramIndexs[0] = 6;

    SetIndices();
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_008
 * @tc.desc: Verify the input indices size is 0 of the addoperation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_008, TestSize.Level1)
{
    SetIndices();

    m_inputs.size = 0;
    m_inputs.data = nullptr;
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_009
 * @tc.desc: Verify the output indices size is 0 of the addoperation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_009, TestSize.Level1)
{
    SetIndices();

    m_outputs.size = 0;
    m_outputs.data = nullptr;
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_add_operation_010
 * @tc.desc: Verify the ops build failed of the addoperation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_add_operation_010, TestSize.Level1)
{
    SetIndices();

    const int32_t dimInput1[2] = {2, 2};
    const OH_NN_Tensor& tensor = {OH_NN_FLOAT32, 2, dimInput1, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    const int32_t dimInput2[2] = {2, 2};
    const OH_NN_Tensor& tensor1 = {OH_NN_FLOAT32, 2, dimInput2, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor1));
    const int32_t dimOutput[2] = {2, 2};
    const OH_NN_Tensor& tensor2 = {OH_NN_FLOAT32, 2, dimOutput, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor2));
    const OH_NN_Tensor& tensor3 = {OH_NN_INT8, 0, nullptr, nullptr, OH_NN_DIV_ACTIVATIONTYPE};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor3));

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_specify_inputs_and_outputs_001
 * @tc.desc: Verify the success of the specify_inputs_and_outputs function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_specify_inputs_and_outputs_001, TestSize.Level1)
{
    SetIndices();
    SetTensors();

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));

    std::vector<std::shared_ptr<NNTensor>> inTensors = m_innerModelTest.GetInputTensors();
    EXPECT_EQ(inTensors.size(), m_inputs.size);
    std::vector<std::shared_ptr<NNTensor>> outTensors = m_innerModelTest.GetOutputTensors();
    EXPECT_EQ(outTensors.size(), m_outputs.size);
}

/**
 * @tc.name: inner_model_specify_inputs_and_outputs_002
 * @tc.desc: Verify the after buildgraph of the specify_inputs_and_outputs function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_specify_inputs_and_outputs_002, TestSize.Level1)
{
    OH_NN_UInt32Array inputs;
    OH_NN_UInt32Array outputs;
    inputs.data = m_inputIndexs;
    inputs.size = sizeof(m_inputIndexs) / sizeof(uint32_t);
    outputs.data = nullptr;
    outputs.size = sizeof(m_outputIndexs) / sizeof(uint32_t);

    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.BuildFromLiteGraph(liteGraph));

    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.SpecifyInputsAndOutputs(inputs, outputs));
}

/**
 * @tc.name: inner_model_specify_inputs_and_outputs_003
 * @tc.desc: Verify the output indices is nullptr but length not 0 of the specify_inputs_and_outputs function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_specify_inputs_and_outputs_003, TestSize.Level1)
{
    SetIndices();

    m_outputs.data = nullptr;
    SetTensors();

    EXPECT_EQ(OH_NN_INVALID_PARAMETER, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_specify_inputs_and_outputs_004
 * @tc.desc: Verify the specift twice of the specify_inputs_and_outputs function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_specify_inputs_and_outputs_004, TestSize.Level1)
{
    SetIndices();
    SetTensors();

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));

    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));
}

/**
 * @tc.name: inner_model_build_001
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_build_001, TestSize.Level1)
{
    SetIndices();
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.Build());
    EXPECT_EQ(true, m_innerModelTest.IsBuild());
}

/**
 * @tc.name: inner_model_build_002
 * @tc.desc: Verify the build twice forbidden of the build function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_build_002, TestSize.Level1)
{
    SetIndices();
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.Build());
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.Build());
}

/**
 * @tc.name: inner_model_build_003
 * @tc.desc: Verify the params not match optype of the build function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_build_003, TestSize.Level1)
{
    OH_NN_OperationType m_opType = OH_NN_OPS_DIV;

    SetIndices();

    const int dim[2] = {2, 2};
    const OH_NN_Tensor& tensor = {OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensor));
    const OH_NN_Tensor& tensorParam = {OH_NN_INT8, 0, nullptr, nullptr, OH_NN_DIV_ACTIVATIONTYPE};
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddTensor(tensorParam));

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_FAILED, m_innerModelTest.Build());
}

/**
 * @tc.name: inner_model_build_004
 * @tc.desc: Verify the success of the build function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_build_004, TestSize.Level1)
{
    SetIndices();
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.Build());
}

/**
 * @tc.name: inner_model_get_supported_operation_001
 * @tc.desc: Verify the success of the get_supported_operation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_get_supported_operation_001, TestSize.Level1)
{
    const bool *isSupported = nullptr;
    uint32_t opCount = 1;

    SetIndices();
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    size_t deviceID = 10;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.Build());
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.GetSupportedOperations(deviceID, &isSupported, opCount));
}

/**
 * @tc.name: inner_model_get_supported_operation_002
 * @tc.desc: Verify the mock hdi device result of the get_supported_operation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_get_supported_operation_002, TestSize.Level1)
{
    size_t deviceID = 10;
    const bool *isSupported = nullptr;
    uint32_t opCount = 1;

    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph();
    EXPECT_NE(nullptr, liteGraph);
    SetLiteGraph(liteGraph);
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.BuildFromLiteGraph(liteGraph));

    EXPECT_EQ(OH_NN_UNAVALIDABLE_DEVICE, m_innerModelTest.GetSupportedOperations(deviceID, &isSupported, opCount));
}

/**
 * @tc.name: inner_model_get_supported_operation_003
 * @tc.desc: Verify the mock device manager of the get_supported_operation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_get_supported_operation_003, TestSize.Level1)
{
    const bool *isSupported = nullptr;
    uint32_t opCount = 1;

    SetIndices();
    SetTensors();

    uint32_t index = 3;
    const int8_t activation = 0;
    size_t deviceID = 0;
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SetTensorValue(index,
       static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.AddOperation(m_opType, m_params, m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.SpecifyInputsAndOutputs(m_inputs, m_outputs));
    EXPECT_EQ(OH_NN_SUCCESS, m_innerModelTest.Build());
    EXPECT_EQ(OH_NN_FAILED, m_innerModelTest.GetSupportedOperations(deviceID, &isSupported, opCount));

    std::shared_ptr<mindspore::lite::LiteGraph> liteGraph = m_innerModelTest.GetLiteGraphs();
    EXPECT_EQ(liteGraph->name_, "NNR_Model");
}

/**
 * @tc.name: inner_model_get_supported_operation_004
 * @tc.desc: Verify the before build of the get_supported_operation function
 * @tc.type: FUNC
 */
HWTEST_F(InnerModelTest, inner_model_get_supported_operation_004, TestSize.Level1)
{
    size_t deviceID = 10;
    const bool *isSupported = nullptr;
    uint32_t opCount = 1;

    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, m_innerModelTest.GetSupportedOperations(deviceID, &isSupported, opCount));
}
} // namespace UnitTest
} // namespace NNRT
