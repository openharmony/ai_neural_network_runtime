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

#include "neural_network_runtime_inner_test.h"

#include "mindir.h"
#include "inner_model.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Unittest {
void NeuralNetworkRuntimeInnerTest::SetUpTestCase(void)
{
}

void NeuralNetworkRuntimeInnerTest::TearDownTestCase(void)
{
}

void NeuralNetworkRuntimeInnerTest::SetUp(void)
{
}

void NeuralNetworkRuntimeInnerTest::TearDown(void)
{
}

/*
 * @tc.name: build_from_lite_graph_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNModel_BuildFromLiteGraph function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, build_from_lite_graph_001, testing::ext::TestSize.Level0)
{
    OH_NNModel* model = nullptr;
    OH_NN_Extension* extensions = nullptr;
    size_t extensionSize = 0;
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph;
    EXPECT_NE(nullptr, liteGraph);
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph, extensions, extensionSize);
    delete liteGraph;
    liteGraph = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: build_from_lite_graph_002
 * @tc.desc: Verify the liteGraph is nullptr of the OH_NNModel_BuildFromLiteGraph function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, build_from_lite_graph_002, testing::ext::TestSize.Level0)
{
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);
    const void* liteGraph = nullptr;
    OH_NN_Extension* extensions = nullptr;
    size_t extensionSize = 0;
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph, extensions, extensionSize);
    delete innerModel;
    innerModel = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: build_from_lite_graph_003
 * @tc.desc: Verify the success of the OH_NNModel_BuildFromLiteGraph function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, build_from_lite_graph_003, testing::ext::TestSize.Level0)
{
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph;
    OH_NN_Extension* extensions = nullptr;
    size_t extensionSize = 0;
    EXPECT_NE(nullptr, liteGraph);
    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_ = {0};
    liteGraph->output_indices_ = {1};
    const std::vector<mindspore::lite::QuantParam> quant_params {};
    for (size_t indexInput = 0; indexInput < liteGraph->input_indices_.size(); ++indexInput) {
        const std::vector<int32_t> dim = {3, 3};
        const std::vector<uint8_t> data(36, 1);

        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    }
    for (size_t indexOutput = 0; indexOutput < liteGraph->output_indices_.size(); ++indexOutput) {
        const std::vector<int32_t> dimOut = {3, 3};
        const std::vector<uint8_t> dataOut(36, 1);
        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    }
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph, extensions, extensionSize);
    delete innerModel;
    innerModel = nullptr;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: build_from_lite_graph_004
 * @tc.desc: Verify that the liteGraph parameter passed to the OH_NNModel_BuildFromLiteGraph function is invalid.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, build_from_lite_graph_004, testing::ext::TestSize.Level0)
{
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph;
    OH_NN_Extension* extensions = nullptr;
    size_t extensionSize = 0;
    EXPECT_NE(nullptr, liteGraph);
    liteGraph->name_ = "testGraph";
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph, extensions, extensionSize);
    delete innerModel;
    delete liteGraph;
    innerModel = nullptr;
    liteGraph = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: build_from_lite_graph_005
 * @tc.desc: Verify the success of the OH_NNModel_BuildFromLiteGraph function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, build_from_lite_graph_005, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_BuildFromLiteGraph build_from_lite_graph_005");
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph;
    char a = 'a';
    OH_NN_Extension extension1 = {"QuantBuffer", &a, 8};
    OH_NN_Extension extension2 = {"ModelName", &a, 8};
    OH_NN_Extension extension3 = {"Profiling", &a, 8};
    OH_NN_Extension extension7 = {"isProfiling", &a, 8};
    OH_NN_Extension extension4 = {"opLayout", &a, 8};
    OH_NN_Extension extension5 = {"InputDims", &a, 8};
    OH_NN_Extension extension6 = {"DynamicDims", &a, 8};
    OH_NN_Extension extension[7] = {extension1, extension2, extension7, extension4, extension5, extension6, extension3};
    size_t extensionSize = 7;
    EXPECT_NE(nullptr, liteGraph);
    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_ = {0};
    liteGraph->output_indices_ = {1};
    mindspore::lite::DataType data_type = mindspore::lite::DataType::DATA_TYPE_INT32;
    int dim = 1;
    int32_t *dims = &dim;
    uint32_t dims_size = 1;
    mindspore::lite::Format format = mindspore::lite::Format::FORMAT_HWCK;
    uint8_t datas = 0;
    uint8_t *data = &datas;
    uint32_t data_size = 2;
    mindspore::lite::QuantParam quant_params;
    uint32_t quant_params_size = 0;
    mindspore::lite::TensorPtr ptr2 = mindspore::lite::MindIR_Tensor_Create(&a, data_type, dims, dims_size,
                               format, data, data_size,
                               &quant_params, quant_params_size);

    for (size_t indexInput = 0; indexInput < liteGraph->input_indices_.size(); ++indexInput) {
        const std::vector<int32_t> dim = {3, 3};
        const std::vector<uint8_t> data(36, 1);

        liteGraph->all_tensors_.emplace_back(ptr2);
    }
    for (size_t indexOutput = 0; indexOutput < liteGraph->output_indices_.size(); ++indexOutput) {
        const std::vector<int32_t> dimOut = {3, 3};
        const std::vector<uint8_t> dataOut(36, 1);
        liteGraph->all_tensors_.emplace_back(ptr2);
    }
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph, extension, extensionSize);
    delete innerModel;
    innerModel = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_buildfrommetagraph_001
 * @tc.desc: Verify that the liteGraph parameter passed to the OH_NNModel_BuildFromLiteGraph function is invalid.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, oh_nnmodel_buildfrommetagraph_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_BuildFromMetaGraph oh_nnmodel_buildfrommetagraph_001");
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph;
    char a = 'a';
    OH_NN_Extension extension1 = {"QuantBuffer", &a, 1};
    OH_NN_Extension extension2 = {"ModelName", &a, 1};
    OH_NN_Extension extension3 = {"Profiling", &a, 1};
    OH_NN_Extension extension4 = {"opLayout", &a, 1};
    OH_NN_Extension extension[4] = {extension1, extension2, extension3, extension4};

    size_t extensionSize = 4;
    EXPECT_NE(nullptr, liteGraph);
    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_ = {0};
    liteGraph->output_indices_ = {1};
    const std::vector<mindspore::lite::QuantParam> quant_params {};
    for (size_t indexInput = 0; indexInput < liteGraph->input_indices_.size(); ++indexInput) {
        const std::vector<int32_t> dim = {3, 3};
        const std::vector<uint8_t> data(36, 1);

        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    }
    for (size_t indexOutput = 0; indexOutput < liteGraph->output_indices_.size(); ++indexOutput) {
        const std::vector<int32_t> dimOut = {3, 3};
        const std::vector<uint8_t> dataOut(36, 1);
        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    }
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromMetaGraph(model, liteGraph, extension, extensionSize);
    delete innerModel;
    innerModel = nullptr;
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: oh_nnmodel_buildfrommetagraph_002
 * @tc.desc: Verify that the liteGraph parameter passed to the OH_NNModel_BuildFromLiteGraph function is invalid.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, oh_nnmodel_buildfrommetagraph_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_BuildFromMetaGraph oh_nnmodel_buildfrommetagraph_002");
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph;
    OH_NN_Extension* extensions = nullptr;

    size_t extensionSize = 0;
    EXPECT_NE(nullptr, liteGraph);
    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_ = {0};
    liteGraph->output_indices_ = {1};
    const std::vector<mindspore::lite::QuantParam> quant_params {};
    for (size_t indexInput = 0; indexInput < liteGraph->input_indices_.size(); ++indexInput) {
        const std::vector<int32_t> dim = {3, 3};
        const std::vector<uint8_t> data(36, 1);

        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    }
    for (size_t indexOutput = 0; indexOutput < liteGraph->output_indices_.size(); ++indexOutput) {
        const std::vector<int32_t> dimOut = {3, 3};
        const std::vector<uint8_t> dataOut(36, 1);
        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create());
    }
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromMetaGraph(nullptr, liteGraph, extensions, extensionSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_buildfrommetagraph_003
 * @tc.desc: Verify that the liteGraph parameter passed to the OH_NNModel_BuildFromLiteGraph function is invalid.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, oh_nnmodel_buildfrommetagraph_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_BuildFromMetaGraph oh_nnmodel_buildfrommetagraph_003");
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);
    OH_NN_Extension* extensions = nullptr;
    size_t extensionSize = 0;
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromMetaGraph(model, nullptr, extensions, extensionSize);
    delete innerModel;
    innerModel = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_setinputsandoutputsinfo_001
 * @tc.desc: Verify that the liteGraph parameter passed to the OH_NNModel_BuildFromLiteGraph function is invalid.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, oh_nnmodel_setinputsandoutputsinfo_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetInputsAndOutputsInfo oh_nnmodel_setinputsandoutputsinfo_001");
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);

    OH_NN_TensorInfo inputsInfo;
    size_t inputSize = 1;
    OH_NN_TensorInfo outputsInfo;
    size_t outputSize = 1 ;
    OH_NN_ReturnCode ret = OH_NNModel_SetInputsAndOutputsInfo(model, &inputsInfo, inputSize, &outputsInfo, outputSize);
    delete innerModel;
    innerModel = nullptr;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: oh_nnmodel_setinputsandoutputsinfo_002
 * @tc.desc: Verify that the liteGraph parameter passed to the OH_NNModel_BuildFromLiteGraph function is invalid.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, oh_nnmodel_setinputsandoutputsinfo_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetInputsAndOutputsInfo oh_nnmodel_setinputsandoutputsinfo_002");
    OH_NN_TensorInfo inputsInfo;
    size_t inputSize = 1;
    OH_NN_TensorInfo outputsInfo;
    size_t outputSize = 1 ;
    OH_NN_ReturnCode ret = OH_NNModel_SetInputsAndOutputsInfo(nullptr, &inputsInfo, inputSize, &outputsInfo, outputSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_setinputsandoutputsinfo_003
 * @tc.desc: Verify that the liteGraph parameter passed to the OH_NNModel_BuildFromLiteGraph function is invalid.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, oh_nnmodel_setinputsandoutputsinfo_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetInputsAndOutputsInfo oh_nnmodel_setinputsandoutputsinfo_003");
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);

    OH_NN_TensorInfo inputsInfo;
    size_t inputSize = 0;
    OH_NN_TensorInfo outputsInfo;
    size_t outputSize = 1 ;
    OH_NN_ReturnCode ret = OH_NNModel_SetInputsAndOutputsInfo(model, &inputsInfo, inputSize, &outputsInfo, outputSize);
    delete innerModel;
    innerModel = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_setinputsandoutputsinfo_004
 * @tc.desc: Verify that the liteGraph parameter passed to the OH_NNModel_BuildFromLiteGraph function is invalid.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeInnerTest, oh_nnmodel_setinputsandoutputsinfo_004, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetInputsAndOutputsInfo oh_nnmodel_setinputsandoutputsinfo_004");
    OHOS::NeuralNetworkRuntime::InnerModel* innerModel = new (std::nothrow) OHOS::NeuralNetworkRuntime::InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);

    OH_NN_TensorInfo inputsInfo;
    size_t inputSize = 1;
    OH_NN_TensorInfo outputsInfo;
    size_t outputSize = 0;
    OH_NN_ReturnCode ret = OH_NNModel_SetInputsAndOutputsInfo(model, &inputsInfo, inputSize, &outputsInfo, outputSize);
    delete innerModel;
    innerModel = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}
} // namespace Unittest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
