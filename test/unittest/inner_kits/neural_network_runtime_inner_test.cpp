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
#include "frameworks/native/inner_model.h"

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
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph;
    EXPECT_NE(nullptr, liteGraph);
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph);
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
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph);
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
    EXPECT_NE(nullptr, liteGraph);
    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_ = {0};
    liteGraph->output_indices_ = {1};
    const std::vector<mindspore::lite::QuantParam> quant_params {};
    for (size_t indexInput = 0; indexInput < liteGraph->input_indices_.size(); ++indexInput) {
        const std::vector<int32_t> dim = {3, 3};
        const std::vector<uint8_t> data(36, 1);

        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create(liteGraph->name_,
            mindspore::lite::DATA_TYPE_FLOAT32, dim, mindspore::lite::FORMAT_NCHW, data, quant_params));
    }
    for (size_t indexOutput = 0; indexOutput < liteGraph->output_indices_.size(); ++indexOutput) {
        const std::vector<int32_t> dimOut = {3, 3};
        const std::vector<uint8_t> dataOut(36, 1);
        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create(liteGraph->name_,
            mindspore::lite::DATA_TYPE_FLOAT32, dimOut, mindspore::lite::FORMAT_NCHW, dataOut, quant_params));
    }
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph);
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
    EXPECT_NE(nullptr, liteGraph);
    liteGraph->name_ = "testGraph";
    OH_NN_ReturnCode ret = OH_NNModel_BuildFromLiteGraph(model, liteGraph);
    delete innerModel;
    delete liteGraph;
    innerModel = nullptr;
    liteGraph = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}
} // namespace Unittest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
