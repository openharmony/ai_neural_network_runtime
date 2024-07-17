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

#include <sys/mman.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "common/log.h"
#include "hdi_device_v1_0.h"
#include "test/unittest/common/v1_0/mock_idevice.h"

#include "lite_graph_to_hdi_model_v1_0.h"
#include "device.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"
#include "nnbackend.h"
#include "ops_registry.h"
#include "transform.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using LiteGraphPrimitvePtr = std::unique_ptr<void, void(*)(void*)>;

namespace MSLITE = mindspore::lite;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace V1 {
namespace UnitTest {
class LiteGraphToHDIModelTest : public testing::Test {
public:
    LiteGraphToHDIModelTest() = default;
    ~LiteGraphToHDIModelTest() = default;
public:
    std::vector<uint32_t> m_inputs{0, 1};
    std::vector<uint32_t> m_outputs{2};
    std::vector<uint32_t> m_param{3};
    std::vector<int32_t> m_input_dim{3, 3};
    std::vector<int32_t> m_output_dim{3, 3};
    std::vector<int32_t> m_param_dim{};
};

MSLITE::LiteGraph::Node* getNode(void* primitive)
{
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();
    node->name_ = "NNRt_SubGraph";
    node->quant_type_ = 1;
    node->primitive_ = primitive;
    node->input_indices_ = {1, 1, 1, 1};
    node->output_indices_ = {1, 1, 1, 1};
    return node;
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_001, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_001");
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(nullptr, tensorBuffer);
    EXPECT_EQ(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_002, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_002");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {0, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_EQ(nullptr, model);

    uint8_t *mmapPtr = static_cast<uint8_t *>(mmap(nullptr,
        tensorBuffer.bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, tensorBuffer.fd, 0));
    EXPECT_EQ(MAP_FAILED, mmapPtr);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_003, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_003");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::SubGraph* subGraph = new (std::nothrow) MSLITE::LiteGraph::SubGraph();
    subGraph->name_ = "NNRt_SubGraph";
    subGraph->input_indices_ = {1, 1, 1, 1};
    subGraph->output_indices_ = {1, 1, 1, 1};
    subGraph->node_indices_ = {1, 1, 1, 1};

    void* tp = MSLITE::MindIR_Tensor_Create();

    liteGraph.get()->all_tensors_.emplace_back(tp);
    liteGraph.get()->sub_graphs_.emplace_back(subGraph);

    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 1, 1, 1};

    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);

    uint8_t *mmapPtr = static_cast<uint8_t *>(mmap(nullptr,
        tensorBuffer.bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, tensorBuffer.fd, 0));
    EXPECT_EQ(MAP_FAILED, mmapPtr);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_004, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_004");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    liteGraph.get()->all_nodes_.emplace_back(nullptr);
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_EQ(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_005, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_005");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::Node* node = new(std::nothrow) MSLITE::LiteGraph::Node();
    liteGraph.get()->all_nodes_.emplace_back(node);
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_EQ(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_006
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_006, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_006");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_ABS};

    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_007
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_007, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_007");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int8_t num = 1;
    int8_t* fuseData = &num;
    mindspore::lite::ActivationType type = NNToMS::TransfromFusionType(static_cast<OH_NN_FuseType>(*fuseData));
    void* primitive = mindspore::lite::MindIR_AddFusion_CreatePrimitive(type);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_008
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_008, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_008");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t keepDims {0};
    void* primitive = mindspore::lite::MindIR_All_CreatePrimitive(keepDims);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_009
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_009, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_009");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis {-1};
    int64_t topK {1};
    bool keepDims {false};
    bool outMaxValue {false};
    void* primitive = mindspore::lite::MindIR_ArgMaxFusion_CreatePrimitive(axis, topK, keepDims, outMaxValue);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_010
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_010, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_010");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t summarize {0};
    void* primitive = mindspore::lite::MindIR_Assert_CreatePrimitive(summarize);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_011
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_011, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_011");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> kernelSize;
    std::vector<int64_t> pad;
    std::vector<int64_t> strides;
    mindspore::lite::PadMode padMode {mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    mindspore::lite::RoundMode roundMode {mindspore::lite::ROUND_MODE_FLOOR};
    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    bool global {false};
    void* primitive = mindspore::lite::MindIR_AvgPoolFusion_CreatePrimitive(kernelSize, strides, pad,
        padMode, roundMode, format, global, activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_012
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_012, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_012");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> blockSize;
    std::vector<std::vector<int64_t>> crops;
    void* primitive = mindspore::lite::MindIR_BatchToSpaceND_CreatePrimitive(blockSize, crops);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_013
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_013, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_013");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float epsilon {0.0001f};
    void* primitive = mindspore::lite::MindIR_FusedBatchNorm_CreatePrimitive(epsilon);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_014
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_014, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_014");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float epsilon {0.0001f};
    void* primitive = mindspore::lite::MindIR_FusedBatchNorm_CreatePrimitive(epsilon);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_015
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_015, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_015");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_BiasAdd_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_016
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_016, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_016");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> shape;
    void* primitive = mindspore::lite::MindIR_BroadcastTo_CreatePrimitive(shape);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_017
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_017, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_017");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Cast_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_018
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_018, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_018");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Ceil_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_019
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_019, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_019");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float max {0.0f};
    float min {0.0f};
    void* primitive = mindspore::lite::MindIR_Clip_CreatePrimitive(max, min);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_020
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_020, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_020");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis{0};
    void* primitive = mindspore::lite::MindIR_Concat_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_021
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_021, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_021");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t dataType {0};
    std::vector<float> value;
    void* primitive = mindspore::lite::MindIR_ConstantOfShape_CreatePrimitive(dataType, value);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_022
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_022, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_022");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t group {1};
    int64_t inChannel {0};
    int64_t outChannel {0};
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> strides;
    std::vector<int64_t> padList;
    std::vector<int64_t> dilation;
    std::vector<int64_t> outputPaddings;
    mindspore::lite::PadMode padMode{mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = MindIR_Conv2dTransposeFusion_CreatePrimitive(kernelSize,
        strides, dilation, padMode, padList, group, inChannel, outChannel,
        activationType, outputPaddings);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_023
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_023, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_023");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Cos_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_024
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_024, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_024");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis {0};
    std::vector<int64_t> offset;
    void* primitive = mindspore::lite::MindIR_Crop_CreatePrimitive(axis, offset);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_025
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_025, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_025");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t blockSize {0};
    std::string mode;
    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    void* primitive = mindspore::lite::MindIR_DepthToSpace_CreatePrimitive(blockSize, format, mode);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_026
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_026, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_026");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t inputSize {0};
    std::vector<float> scale;
    float nmsIoUThreshold {0.0f};
    float nmsScoreThreshold {0.0f};
    int64_t maxDetections {0};
    int64_t detectionsPerClass {0};
    int64_t maxClassesPerDetection {0};
    int64_t numClasses {0};
    bool useRegularNms {false};
    bool outQuantized {false};
    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    void* primitive = mindspore::lite::MindIR_DetectionPostProcess_CreatePrimitive(format, inputSize, scale,
        nmsIoUThreshold, nmsScoreThreshold, maxDetections, detectionsPerClass, maxClassesPerDetection,
        numClasses, useRegularNms, outQuantized);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_027
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_027, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_027");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::EltwiseMode mode {mindspore::lite::ELTWISE_MODE_PROD};
    void* primitive = mindspore::lite::MindIR_Eltwise_CreatePrimitive(mode);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_028
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_028, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_028");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Equal_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_029
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_029, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_029");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Erf_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_030
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_030, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_030");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float base {-1.0f};
    float scale {1.0f};
    float shift {0.0f};
    void* primitive = mindspore::lite::MindIR_ExpFusion_CreatePrimitive(base, scale, shift);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_031
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_031, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_031");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_ExpandDims_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_032
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_032, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_032");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Fill_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_033
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_033, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_033");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis {1};
    void* primitive = mindspore::lite::MindIR_Flatten_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_034
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_034, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_034");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Floor_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_035
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_035, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_035");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    bool hasBias {false};
    bool useAxis {false};
    int64_t axis {0};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_FullConnection_CreatePrimitive(hasBias, useAxis,
        axis, activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_036
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_036, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_036");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Gather_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_037
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_037, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_037");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_GatherNd_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_038
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_038, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_038");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::ActivationType activationType = mindspore::lite::ACTIVATION_TYPE_GELU;
    float alpha = 0.0f;
    float minVal = 0.0f;
    float maxVal = 0.0f;
    bool approximate = false;
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType,
        alpha, minVal, maxVal, approximate);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_039
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_039, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_039");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Greater_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_040
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_040, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_040");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_GreaterEqual_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_041
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_041, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_041");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::ActivationType activationType = mindspore::lite::ACTIVATION_TYPE_HSIGMOID;
    float alpha = 0.0f;
    float minVal = 0.0f;
    float maxVal = 0.0f;
    bool approximate = false;
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType,
        alpha, minVal, maxVal, approximate);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_042
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_042, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_042");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float epsilon {0.0f};
    void* primitive = mindspore::lite::MindIR_InstanceNorm_CreatePrimitive(epsilon);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_043
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_043, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_043");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float epsilon {0.0f};
    void* primitive = mindspore::lite::MindIR_InstanceNorm_CreatePrimitive(epsilon);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_044
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_044, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_044");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> axis;
    float epsilon {1e-6};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_L2NormalizeFusion_CreatePrimitive(axis, epsilon, activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_045
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_045, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_045");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t beginNormAxis {1};
    float epsilon {1e-7};
    bool elementwiseAffine {true};
    int64_t beginParamsAxis {1};
    void* primitive = mindspore::lite::MindIR_LayerNormFusion_CreatePrimitive(beginNormAxis,
        epsilon, elementwiseAffine, beginParamsAxis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_046
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_046, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_046");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_LEAKY_RELU};
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_047
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_047, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_047");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Less_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_048
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_048, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_048");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_LessEqual_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_049
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_049, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_049");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Log_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_050
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_050, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_050");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis {0};
    void* primitive = mindspore::lite::MindIR_LogSoftmax_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_051
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_051, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_051");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_LogicalAnd_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_052
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_052, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_052");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_LogicalNot_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_053
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_053, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_053");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_LogicalOr_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_054
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_054, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_054");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t depthRadius {0};
    float bias {0.0f};
    float alpha {0.0f};
    float beta {0.0f};
    std::string normRegion {"ACROSS_CHANNELS"};
    void* primitive = mindspore::lite::MindIR_LRN_CreatePrimitive(depthRadius, bias, alpha,
        beta, normRegion);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_055
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_055, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_055");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    bool bidirectional {false};
    bool hasBias {false};
    int64_t inputSize {0};
    int64_t hiddenSize {0};
    int64_t numLayers {0};
    int64_t numDirections {0};
    float dropout {0.0f};
    float zoneoutCell {0.0f};
    float zoneoutHidden {0.0f};
    int64_t projSize {0};
    void* primitive = mindspore::lite::MindIR_LSTM_CreatePrimitive(bidirectional, hasBias, inputSize,
        hiddenSize, numLayers, numDirections, dropout, zoneoutCell, zoneoutHidden, projSize);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_056
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_056, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_056");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Maximum_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_057
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_057, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_057");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> kernelSize;
    std::vector<int64_t> pad;
    std::vector<int64_t> strides;
    mindspore::lite::PadMode padMode {mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    bool global {false};
    void* primitive = MindIR_MaxPoolFusion_CreatePrimitive(kernelSize, strides, pad,
        padMode, format, global, activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_058
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_058, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_058");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Minimum_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_059
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_059, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_059");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Mod_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_060
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_060, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_060");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_MulFusion_CreatePrimitive(activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_061
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_061, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_061");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Neg_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_062
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_062, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_062");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_NotEqual_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_063
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_063, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_063");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis {-1};
    void* primitive = mindspore::lite::MindIR_OneHot_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_064
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_064, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_064");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<std::vector<int64_t>> paddings;
    float constantValue {0.0f};
    mindspore::lite::PaddingMode paddingMode {mindspore::lite::PADDING_MODE_CONSTANT};
    void* primitive = MindIR_PadFusion_CreatePrimitive(paddings, paddingMode, constantValue);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_065
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_065, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_065");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float scale {1.0f};
    float shift {0.0f};
    void* primitive = mindspore::lite::MindIR_PowFusion_CreatePrimitive(scale, shift);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_066
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_066, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_066");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    bool channelShared{false};
    void* primitive = mindspore::lite::MindIR_PReLUFusion_CreatePrimitive(channelShared);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_067
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_067, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_067");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    const uint64_t* srcT{nullptr};
    const uint64_t* dstT{nullptr};
    int64_t axis {0};
    void* primitive = mindspore::lite::MindIR_QuantDTypeCast_CreatePrimitive(*srcT, *dstT, axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_068
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_068, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_068");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t dType {0.0f};
    int64_t start {0};
    int64_t limit {0};
    int64_t delta {1};
    void* primitive = mindspore::lite::MindIR_Range_CreatePrimitive(dType, start, limit, delta);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_069
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_069, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_069");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Rank_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_070
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_070, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_070");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Reciprocal_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_071
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_071, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_071");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::ReduceMode mode {mindspore::lite::REDUCE_MODE_ALL};
    float coeff {0.0f};
    bool reduceToEnd {false};
    bool keepDims {false};
    void* primitive = mindspore::lite::MindIR_ReduceFusion_CreatePrimitive(keepDims, mode, reduceToEnd, coeff);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_072
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_072, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_072");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float alpha{0.0f};
    float minVal{0.0f};
    float maxVal{0.0f};
    bool approximate{false};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_RELU6};
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_073
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_073, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_073");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Reshape_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_074
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_074, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_074");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float cubicCoeff{0.0f};
    float extrapolationValue{0.0f};
    mindspore::lite::NearestMode nearestMode{mindspore::lite::NEAREST_MODE_NORMAL};
    mindspore::lite::ResizeMethod method {mindspore::lite::RESIZE_METHOD_LINEAR};
    uint64_t newHeight{0};
    uint64_t newWidth{0};
    bool preserveAspectRatio{false};
    mindspore::lite::CoordinateTransformMode coordinateTransformMode {
        mindspore::lite::COORDINATE_TRANSFORM_MODE_ASYMMETRIC};
    uint64_t excludeOutside{0};
    void* primitive = mindspore::lite::MindIR_Resize_CreatePrimitive(method, newHeight, newWidth,
        preserveAspectRatio, coordinateTransformMode, cubicCoeff, excludeOutside,
        extrapolationValue, nearestMode);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_075
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_075, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_075");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Round_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_076
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_076, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_076");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Rsqrt_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_077
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_077, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_077");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    const uint64_t* axis{nullptr};
    void* primitive = mindspore::lite::MindIR_ScaleFusion_CreatePrimitive(*axis, activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_078
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_078, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_078");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_ScatterNd_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_079
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_079, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_079");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Select_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_080
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_080, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_080");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float alpha{0.0f};
    float minVal{0.0f};
    float maxVal{0.0f};
    bool approximate{false};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_SIGMOID};
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha, minVal,
        maxVal, approximate);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_081
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_081, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_081");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Sin_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_082
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_082, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_082");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::Format format {mindspore::lite::FORMAT_NCHW};
    int64_t blockSize {0};
    void* primitive = mindspore::lite::MindIR_SpaceToDepth_CreatePrimitive(blockSize, format);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_083
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_083, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_083");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_SparseToDense_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_084
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_084, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_084");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Square_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_085
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_085, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_085");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_SWISH};
    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_086
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_086, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_086");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis {0};
    void* primitive = mindspore::lite::MindIR_Unstack_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_087
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_087, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_087");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Where_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_088
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_088, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_088");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Shape_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_089
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_089, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_089");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Unsqueeze_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_090
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_090, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_090");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t inChannel{0};
    int64_t outChannel{0};
    std::vector<int64_t> kernelSize;
    std::vector<int64_t> strides;
    std::vector<int64_t> pad;
    std::vector<int64_t> dilation;
    mindspore::lite::PadMode padMode{mindspore::lite::PAD_MODE_PAD};
    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_Conv2DFusion_CreatePrimitive(kernelSize, strides,
        dilation, padMode, pad, inChannel, inChannel, outChannel, activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_091
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_091, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_091");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_DivFusion_CreatePrimitive(activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_092
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_092, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_092");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::ActivationType activationType{mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    bool transposeA{false};
    bool transposeB{false};
    void* primitive = mindspore::lite::MindIR_MatMulFusion_CreatePrimitive(transposeA, transposeB, activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_093
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_093, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_093");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> axes;
    void* primitive = mindspore::lite::MindIR_SliceFusion_CreatePrimitive(axes);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_094
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_094, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_094");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Softmax_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_095
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_095, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_095");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<std::vector<int64_t>> paddings;
    std::vector<int64_t> block_shape {};
    void* primitive = mindspore::lite::MindIR_SpaceToBatchND_CreatePrimitive(block_shape, paddings);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_096
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_096, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_096");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t outputNum {0};
    std::vector<int64_t> sizeSplits;
    int64_t axis {0};
    void* primitive = mindspore::lite::MindIR_Split_CreatePrimitive(outputNum, sizeSplits, axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_097
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_097, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_097");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Sqrt_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_098
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_098, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_098");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_SquaredDifference_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_099
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_099, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_099");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Squeeze_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_100
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_100, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_100");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis = {0};
    void* primitive = mindspore::lite::MindIR_Stack_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_101
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_101, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_101");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t beginMask = {0};
    int64_t endMask = {0};
    int64_t ellipsisMask = {0};
    int64_t newAxisMask = {0};
    int64_t shrinkAxisMask = {0};
    void* primitive = mindspore::lite::MindIR_StridedSlice_CreatePrimitive(beginMask, endMask, ellipsisMask,
        newAxisMask, shrinkAxisMask);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_102
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_102, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_102");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    mindspore::lite::ActivationType  activationType {mindspore::lite::ACTIVATION_TYPE_NO_ACTIVATION};
    void* primitive = mindspore::lite::MindIR_SubFusion_CreatePrimitive(activationType);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_103
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_103, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_103");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> dims {0};
    void* primitive = mindspore::lite::MindIR_TileFusion_CreatePrimitive(dims);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_104
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_104, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_104");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    int64_t axis {0};
    bool sorted {true};
    void* primitive = mindspore::lite::MindIR_TopKFusion_CreatePrimitive(sorted, axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_105
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_105, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_105");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    void* primitive = mindspore::lite::MindIR_Transpose_CreatePrimitive();

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_106
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_106, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_106");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Unsqueeze_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_107
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_107, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_107");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    std::vector<int64_t> axis;
    void* primitive = mindspore::lite::MindIR_Unsqueeze_CreatePrimitive(axis);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
}

/**
 * @tc.name: litegraphtohdimodeltest_litegraph_to_hdimodel_108
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_litegraph_to_hdimodel_108, TestSize.Level0)
{
    LOGE("LiteGraph_To_HDIModel litegraphtohdimodeltest_litegraph_to_hdimodel_108");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();
    MSLITE::LiteGraph::SubGraph* subGraph = new (std::nothrow) MSLITE::LiteGraph::SubGraph();
    subGraph->name_ = "NNRt_SubGraph";
    subGraph->input_indices_ = {1, 1, 1, 1};
    subGraph->output_indices_ = {1, 1, 1, 1};
    subGraph->node_indices_ = {1, 1, 1, 1};

    void* tp = MSLITE::MindIR_Tensor_Create();

    liteGraph.get()->all_tensors_.emplace_back(tp);
    liteGraph.get()->all_tensors_.emplace_back(nullptr);
    liteGraph.get()->sub_graphs_.emplace_back(subGraph);

    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 1, 1, 1};

    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);

    uint8_t *mmapPtr = static_cast<uint8_t *>(mmap(nullptr,
        tensorBuffer.bufferSize, PROT_READ | PROT_WRITE, MAP_SHARED, tensorBuffer.fd, 0));
    EXPECT_EQ(MAP_FAILED, mmapPtr);
}

/**
 * @tc.name: litegraphtohdimodeltest_hdimodel_destroy_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(LiteGraphToHDIModelTest, litegraphtohdimodeltest_hdimodel_destroy_001, TestSize.Level0)
{
    LOGE("HDIModel_Destroy litegraphtohdimodeltest_hdimodel_destroy_001");
    std::shared_ptr<MSLITE::LiteGraph> liteGraph = std::make_shared<MSLITE::LiteGraph>();

    float alpha {0.0f};
    float minVal {0.0f};
    float maxVal {0.0f};
    bool approximate {false};
    mindspore::lite::ActivationType activationType {mindspore::lite::ACTIVATION_TYPE_ABS};

    void* primitive = mindspore::lite::MindIR_Activation_CreatePrimitive(activationType, alpha,
        minVal, maxVal, approximate);

    liteGraph.get()->all_nodes_.emplace_back(getNode(primitive));
    OHOS::HDI::Nnrt::V1_0::SharedBuffer tensorBuffer {-1, 0, 0, 0};
    OHOS::HDI::Nnrt::V1_0::Model * model = LiteGraph_To_HDIModel(liteGraph.get(), tensorBuffer);
    EXPECT_NE(nullptr, model);
    HDIModel_Destroy(&model);
}
} // namespace UnitTest
} // namespace V1
} // namespace NeuralNetworkRuntime
} // namespace OHOS