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

#include "nnbackend.h"
#include "nncompiler.h"
#include "device.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"
#include "common/utils.h"
#include "inner_model.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class NNCompilerTest : public testing::Test {
public:
    NNCompilerTest() = default;
    ~NNCompilerTest() = default;
    OH_NN_ReturnCode BuildModel(InnerModel& innerModel);
};

class MockIDevice : public Device {
public:
    MOCK_METHOD1(GetDeviceName, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetVendorName, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetVersion, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetDeviceType, OH_NN_ReturnCode(OH_NN_DeviceType&));
    MOCK_METHOD1(GetDeviceStatus, OH_NN_ReturnCode(DeviceStatus&));
    MOCK_METHOD2(GetSupportedOperation, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
        std::vector<bool>&));
    MOCK_METHOD1(IsFloat16PrecisionSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsPerformanceModeSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsPrioritySupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsDynamicInputSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsModelCacheSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD3(PrepareModel, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
                                          const ModelConfig&,
                                          std::shared_ptr<PreparedModel>&));
    MOCK_METHOD3(PrepareModel, OH_NN_ReturnCode(const void*,
                                          const ModelConfig&,
                                          std::shared_ptr<PreparedModel>&));
    MOCK_METHOD4(PrepareModelFromModelCache, OH_NN_ReturnCode(const std::vector<Buffer>&,
                                                        const ModelConfig&,
                                                        std::shared_ptr<PreparedModel>&,
                                                        bool&));
    MOCK_METHOD3(PrepareOfflineModel, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
                                                 const ModelConfig&,
                                                 std::shared_ptr<PreparedModel>&));
    MOCK_METHOD1(AllocateBuffer, void*(size_t));
    MOCK_METHOD2(AllocateTensorBuffer, void*(size_t, std::shared_ptr<TensorDesc>));
    MOCK_METHOD2(AllocateTensorBuffer, void*(size_t, std::shared_ptr<NNTensor>));
    MOCK_METHOD1(ReleaseBuffer, OH_NN_ReturnCode(const void*));
    MOCK_METHOD2(AllocateBuffer, OH_NN_ReturnCode(size_t, int&));
    MOCK_METHOD2(ReleaseBuffer, OH_NN_ReturnCode(int, size_t));
};

class MockIPreparedModel : public PreparedModel {
public:
    MOCK_METHOD1(ExportModelCache, OH_NN_ReturnCode(std::vector<Buffer>&));
    MOCK_METHOD4(Run, OH_NN_ReturnCode(const std::vector<IOTensor>&,
                                 const std::vector<IOTensor>&,
                                 std::vector<std::vector<int32_t>>&,
                                 std::vector<bool>&));
    MOCK_METHOD4(Run, OH_NN_ReturnCode(const std::vector<NN_Tensor*>&,
                                 const std::vector<NN_Tensor*>&,
                                 std::vector<std::vector<int32_t>>&,
                                 std::vector<bool>&));
    MOCK_CONST_METHOD1(GetModelID, OH_NN_ReturnCode(uint32_t&));
    MOCK_METHOD2(GetInputDimRanges, OH_NN_ReturnCode(std::vector<std::vector<uint32_t>>&,
                                               std::vector<std::vector<uint32_t>>&));
};

class MockInnerModel : public InnerModel {
public:
    MOCK_CONST_METHOD0(IsBuild, bool());
    MOCK_METHOD2(BuildFromLiteGraph, OH_NN_ReturnCode(const mindspore::lite::LiteGraph*,
                                        const ExtensionConfig&));
    MOCK_METHOD2(BuildFromMetaGraph, OH_NN_ReturnCode(const void*, const ExtensionConfig&));
    MOCK_METHOD1(AddTensor, OH_NN_ReturnCode(const OH_NN_Tensor&));
    MOCK_METHOD1(AddTensorDesc, OH_NN_ReturnCode(const NN_TensorDesc*));
    MOCK_METHOD2(SetTensorQuantParam, OH_NN_ReturnCode(uint32_t, const NN_QuantParam*));
    MOCK_METHOD2(SetTensorType, OH_NN_ReturnCode(uint32_t, OH_NN_TensorType));
    MOCK_METHOD3(SetTensorValue, OH_NN_ReturnCode(uint32_t, const void*, size_t));
    MOCK_METHOD4(AddOperation, OH_NN_ReturnCode(OH_NN_OperationType,
                                  const OH_NN_UInt32Array&,
                                  const OH_NN_UInt32Array&,
                                  const OH_NN_UInt32Array&));
    MOCK_METHOD3(GetSupportedOperations, OH_NN_ReturnCode(size_t, const bool**, uint32_t&));
    MOCK_METHOD2(SpecifyInputsAndOutputs, OH_NN_ReturnCode(const OH_NN_UInt32Array&, const OH_NN_UInt32Array&));
    MOCK_METHOD4(SetInputsAndOutputsInfo, OH_NN_ReturnCode(const OH_NN_TensorInfo*, size_t,
        const OH_NN_TensorInfo*, size_t));
    MOCK_METHOD0(Build, OH_NN_ReturnCode());
    MOCK_CONST_METHOD0(GetInputTensors, std::vector<std::shared_ptr<NNTensor>>());
    MOCK_CONST_METHOD0(GetOutputTensors, std::vector<std::shared_ptr<NNTensor>>());
    MOCK_CONST_METHOD0(GetInputTensorDescs, std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>());
    MOCK_CONST_METHOD0(GetOutputTensorDescs, std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>>());
    MOCK_CONST_METHOD0(GetLiteGraphs, std::shared_ptr<mindspore::lite::LiteGraph>());
    MOCK_CONST_METHOD0(GetMetaGraph, void*());
    MOCK_CONST_METHOD0(GetExtensionConfig, ExtensionConfig());
};


OH_NN_ReturnCode NNCompilerTest::BuildModel(InnerModel& innerModel)
{
    int32_t inputDims[4] = {1, 2, 2, 3};
    OH_NN_Tensor input1 = {OH_NN_FLOAT32, 4, inputDims, nullptr, OH_NN_TENSOR};
    OH_NN_ReturnCode ret = innerModel.AddTensor(input1);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 添加Add算子的第二个输入Tensor，类型为float32，张量形状为[1, 2, 2, 3]
    OH_NN_Tensor input2 = {OH_NN_FLOAT32, 4, inputDims, nullptr, OH_NN_TENSOR};
    ret = innerModel.AddTensor(input2);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 添加Add算子的参数Tensor，该参数Tensor用于指定激活函数的类型，Tensor的数据类型为int8。
    int32_t activationDims = 1;
    int8_t activationValue = OH_NN_FUSED_NONE;
    OH_NN_Tensor activation = {OH_NN_INT8, 1, &activationDims, nullptr, OH_NN_ADD_ACTIVATIONTYPE};
    ret = innerModel.AddTensor(activation);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 将激活函数类型设置为OH_NN_FUSED_NONE，表示该算子不添加激活函数。
    uint32_t index = 2;
    ret = innerModel.SetTensorValue(index, &activationValue, sizeof(int8_t));
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 设置Add算子的输出，类型为float32，张量形状为[1, 2, 2, 3]
    OH_NN_Tensor output = {OH_NN_FLOAT32, 4, inputDims, nullptr, OH_NN_TENSOR};
    ret = innerModel.AddTensor(output);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 指定Add算子的输入、参数和输出索引
    uint32_t inputIndicesValues[2] = {0, 1};
    uint32_t paramIndicesValues = 2;
    uint32_t outputIndicesValues = 3;
    OH_NN_UInt32Array paramIndices = {&paramIndicesValues, 1};
    OH_NN_UInt32Array inputIndices = {inputIndicesValues, 2};
    OH_NN_UInt32Array outputIndices = {&outputIndicesValues, 1};

    // 向模型实例添加Add算子
    ret = innerModel.AddOperation(OH_NN_OPS_ADD, paramIndices, inputIndices, outputIndices);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 设置模型实例的输入、输出索引
    ret = innerModel.SpecifyInputsAndOutputs(inputIndices, outputIndices);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 完成模型实例的构建
    ret = innerModel.Build();
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    return ret;
}

/**
 * @tc.name: nncompilertest_construct_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_construct_001, TestSize.Level0)
{
    LOGE("NNCompiler nncompilertest_construct_001");
    size_t backendID = 1;
    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_construct_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_construct_002, TestSize.Level0)
{
    LOGE("NNCompiler nncompilertest_construct_002");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_getbackendid_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_getbackendid_001, TestSize.Level0)
{
    LOGE("GetBackendID nncompilertest_getbackendid_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    size_t ret = nncompiler->GetBackendID();
    EXPECT_NE(0, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setcachedir_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setcachedir_001, TestSize.Level0)
{
    LOGE("SetCacheDir nncompilertest_setcachedir_001");
    size_t backendID = 1;

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(nullptr, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = 0;
    OH_NN_ReturnCode ret = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: nncompilertest_setcachedir_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setcachedir_002, TestSize.Level0)
{
    LOGE("SetCacheDir nncompilertest_setcachedir_002");
    size_t backendID = 1;

    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_OPERATION_FORBIDDEN));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = 0;
    OH_NN_ReturnCode ret = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setcachedir_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setcachedir_003, TestSize.Level0)
{
    LOGE("SetCacheDir nncompilertest_setcachedir_003");
    size_t backendID = 1;

    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = 0;
    OH_NN_ReturnCode ret = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setcachedir_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setcachedir_004, TestSize.Level0)
{
    LOGE("SetCacheDir nncompilertest_setcachedir_004");
    size_t backendID = 1;

    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = 0;
    OH_NN_ReturnCode ret = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setperformance_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setperformance_001, TestSize.Level0)
{
    LOGE("SetPerformance nncompilertest_setperformance_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_PerformanceMode performance = OH_NN_PERFORMANCE_NONE;
    OH_NN_ReturnCode ret = nncompiler->SetPerformance(performance);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setperformance_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setperformance_002, TestSize.Level0)
{
    LOGE("SetPerformance nncompilertest_setperformance_002");
    size_t backendID = 1;

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(nullptr, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_PerformanceMode performance = OH_NN_PERFORMANCE_NONE;
    OH_NN_ReturnCode ret = nncompiler->SetPerformance(performance);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: nncompilertest_setperformance_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setperformance_003, TestSize.Level0)
{
    LOGE("SetPerformance nncompilertest_setperformance_003");
    size_t backendID = 1;

    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsPerformanceModeSupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_PerformanceMode performance = OH_NN_PERFORMANCE_NONE;
    OH_NN_ReturnCode ret = nncompiler->SetPerformance(performance);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setperformance_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setperformance_004, TestSize.Level0)
{
    LOGE("SetPerformance nncompilertest_setperformance_004");
    size_t backendID = 1;

    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsPerformanceModeSupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_PerformanceMode performance = OH_NN_PERFORMANCE_LOW;
    OH_NN_ReturnCode ret = nncompiler->SetPerformance(performance);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setpriority_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setpriority_001, TestSize.Level0)
{
    LOGE("SetPriority nncompilertest_setpriority_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_Priority priority = OH_NN_PRIORITY_NONE;
    OH_NN_ReturnCode ret = nncompiler->SetPriority(priority);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setpriority_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setpriority_002, TestSize.Level0)
{
    LOGE("SetPriority nncompilertest_setpriority_002");
    size_t backendID = 1;

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(nullptr, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_Priority priority = OH_NN_PRIORITY_NONE;
    OH_NN_ReturnCode ret = nncompiler->SetPriority(priority);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: nncompilertest_setpriority_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setpriority_003, TestSize.Level0)
{
    LOGE("SetPriority nncompilertest_setpriority_003");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsPrioritySupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_Priority priority = OH_NN_PRIORITY_NONE;
    OH_NN_ReturnCode ret = nncompiler->SetPriority(priority);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setpriority_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setpriority_004, TestSize.Level0)
{
    LOGE("SetPriority nncompilertest_setpriority_004");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsPrioritySupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_Priority priority = OH_NN_PRIORITY_LOW;
    OH_NN_ReturnCode ret = nncompiler->SetPriority(priority);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setenablefp16_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setenablefp16_001, TestSize.Level0)
{
    LOGE("SetEnableFp16 nncompilertest_setenablefp16_001");
    size_t backendID = 1;

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(nullptr, backendID);
    EXPECT_NE(nullptr, nncompiler);

    bool isFp16 = true;
    OH_NN_ReturnCode ret = nncompiler->SetEnableFp16(isFp16);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: nncompilertest_setenablefp16_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setenablefp16_002, TestSize.Level0)
{
    LOGE("SetEnableFp16 nncompilertest_setenablefp16_002");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsFloat16PrecisionSupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    bool isFp16 = true;
    OH_NN_ReturnCode ret = nncompiler->SetEnableFp16(isFp16);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setenablefp16_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setenablefp16_003, TestSize.Level0)
{
    LOGE("SetEnableFp16 nncompilertest_setenablefp16_003");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsFloat16PrecisionSupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    bool isFp16 = true;
    OH_NN_ReturnCode ret = nncompiler->SetEnableFp16(isFp16);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setenablefp16_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setenablefp16_004, TestSize.Level0)
{
    LOGE("SetEnableFp16 nncompilertest_setenablefp16_004");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsPrioritySupported(::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_SUCCESS));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    bool isFp16 = false;
    OH_NN_ReturnCode ret = nncompiler->SetEnableFp16(isFp16);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_isbuild_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_isbuild_001, TestSize.Level0)
{
    LOGE("IsBuild nncompilertest_isbuild_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    bool ret = nncompiler->IsBuild();
    EXPECT_EQ(false, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_build_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_build_001, TestSize.Level0)
{
    LOGE("Build nncompilertest_build_001");
    size_t backendID = 1;

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(nullptr, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode ret = nncompiler->Build();
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/**
 * @tc.name: nncompilertest_build_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_build_002, TestSize.Level0)
{
    LOGE("Build nncompilertest_build_002");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode ret = nncompiler->Build();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_build_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_build_003, TestSize.Level0)
{
    LOGE("Build nncompilertest_build_003");
    size_t backendID = 1;
    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode ret = nncompiler->Build();
    EXPECT_EQ(OH_NN_SUCCESS, ret);
    
    OH_NN_ReturnCode retBuild = nncompiler->Build();
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, retBuild);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_build_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_build_004, TestSize.Level0)
{
    LOGE("Build nncompilertest_build_004");
    size_t backendID = 1;
    InnerModel innerModel;
    void* model = &innerModel;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode ret = nncompiler->Build();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_build_005
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_build_005, TestSize.Level0)
{
    LOGE("Build nncompilertest_build_005");
    size_t backendID = 1;
    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = UINT32_MAX;
    OH_NN_ReturnCode retSetCacheDir = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_SUCCESS, retSetCacheDir);

    OH_NN_ReturnCode ret = nncompiler->Build();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_build_006
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_build_006, TestSize.Level0)
{
    LOGE("Build nncompilertest_build_006");
    size_t backendID = 1;
    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = 0;
    OH_NN_ReturnCode retSetCacheDir = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_SUCCESS, retSetCacheDir);

    OH_NN_ReturnCode ret = nncompiler->Build();
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_build_007
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_build_007, TestSize.Level0)
{
    LOGE("Build nncompilertest_build_007");
    size_t backendID = 1;
    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode ret = nncompiler->Build();
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_savetocachefile_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_savetocachefile_001, TestSize.Level0)
{
    LOGE("SaveToCacheFile nncompilertest_savetocachefile_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode ret = nncompiler->SaveToCacheFile();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_savetocachefile_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_savetocachefile_002, TestSize.Level0)
{
    LOGE("SaveToCacheFile nncompilertest_savetocachefile_002");
    size_t backendID = 1;

    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = UINT32_MAX;
    OH_NN_ReturnCode retSetCacheDir = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_SUCCESS, retSetCacheDir);

    OH_NN_ReturnCode retSave = nncompiler->SaveToCacheFile();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, retSave);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_savetocachefile_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_savetocachefile_003, TestSize.Level0)
{
    LOGE("SaveToCacheFile nncompilertest_savetocachefile_003");
    size_t backendID = 1;

    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = 0;
    OH_NN_ReturnCode retSetCacheDir = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_SUCCESS, retSetCacheDir);

    OH_NN_ReturnCode retSave = nncompiler->SaveToCacheFile();
    EXPECT_EQ(OH_NN_FAILED, retSave);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_savetocachefile_004
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_savetocachefile_004, TestSize.Level0)
{
    LOGE("SaveToCacheFile nncompilertest_savetocachefile_004");
    size_t backendID = 1;

    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode retBuild = nncompiler->Build();
    EXPECT_EQ(OH_NN_SUCCESS, retBuild);

    std::string cacheModelPath = "mock";
    uint32_t version = 0;
    OH_NN_ReturnCode retSetCacheDir = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_SUCCESS, retSetCacheDir);;

    OH_NN_ReturnCode retSave = nncompiler->SaveToCacheFile();
    EXPECT_EQ(OH_NN_FAILED, retSave);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_restorefromcachefile_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_restorefromcachefile_001, TestSize.Level0)
{
    LOGE("RestoreFromCacheFile nncompilertest_restorefromcachefile_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode ret = nncompiler->RestoreFromCacheFile();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_restorefromcachefile_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_restorefromcachefile_002, TestSize.Level0)
{
    LOGE("RestoreFromCacheFile nncompilertest_restorefromcachefile_002");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::string cacheModelPath = "mock";
    uint32_t version = UINT32_MAX;
    OH_NN_ReturnCode retSetCacheDir = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_SUCCESS, retSetCacheDir);;

    OH_NN_ReturnCode ret = nncompiler->RestoreFromCacheFile();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_restorefromcachefile_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_restorefromcachefile_003, TestSize.Level0)
{
    LOGE("RestoreFromCacheFile nncompilertest_restorefromcachefile_003");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::shared_ptr<MockIPreparedModel> prepared = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS;
            }));

    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    EXPECT_CALL(*((MockIDevice *) device.get()), PrepareModel(testing::A<std::shared_ptr<const mindspore::lite::LiteGraph>>(), ::testing::_, ::testing::_))
        .WillOnce(Invoke([&prepared](std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                          const ModelConfig& config,
                                          std::shared_ptr<PreparedModel>& preparedModel) {
                preparedModel = prepared;
                return OH_NN_SUCCESS;
            }));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode retBuild = nncompiler->Build();
    EXPECT_EQ(OH_NN_SUCCESS, retBuild);

    std::string cacheModelPath = "/data/data";
    uint32_t version = UINT32_MAX;
    OH_NN_ReturnCode retSetCacheDir = nncompiler->SetCacheDir(cacheModelPath, version);
    EXPECT_EQ(OH_NN_SUCCESS, retSetCacheDir);;

    OH_NN_ReturnCode ret = nncompiler->RestoreFromCacheFile();
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(device.get());
    testing::Mock::AllowLeak(prepared.get());
}

/**
 * @tc.name: nncompilertest_savetocachebuffer_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_savetocachebuffer_001, TestSize.Level0)
{
    LOGE("SaveToCacheBuffer nncompilertest_savetocachebuffer_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    size_t length = 10; 
    size_t* modelSize = &length;
    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    OH_NN_ReturnCode ret = nncompiler->SaveToCacheBuffer(model, length, modelSize);
    EXPECT_EQ(OH_NN_UNSUPPORTED, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_restorefromcachebuffer_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_restorefromcachebuffer_001, TestSize.Level0)
{
    LOGE("RestoreFromCacheBuffer nncompilertest_restorefromcachebuffer_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    size_t length = 10; 
    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    OH_NN_ReturnCode ret = nncompiler->RestoreFromCacheBuffer(model, length);
    EXPECT_EQ(OH_NN_UNSUPPORTED, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setextensionconfig_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setextensionconfig_001, TestSize.Level0)
{
    LOGE("SetExtensionConfig nncompilertest_setextensionconfig_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::unordered_map<std::string, std::vector<char>> configs;
    OH_NN_ReturnCode ret = nncompiler->SetExtensionConfig(configs);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_setoptions_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_setoptions_001, TestSize.Level0)
{
    LOGE("SetOptions nncompilertest_setoptions_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    std::vector<std::shared_ptr<void>> options;
    OH_NN_ReturnCode ret = nncompiler->SetOptions(options);
    EXPECT_EQ(OH_NN_UNSUPPORTED, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_createexecutor_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_createexecutor_001, TestSize.Level0)
{
    LOGE("CreateExecutor nncompilertest_createexecutor_001");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    NNExecutor* ret = nncompiler->CreateExecutor();
    EXPECT_EQ(nullptr, ret);

    testing::Mock::AllowLeak(device.get());
}

/**
 * @tc.name: nncompilertest_createexecutor_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_createexecutor_002, TestSize.Level0)
{
    LOGE("CreateExecutor nncompilertest_createexecutor_002");
    size_t backendID = 1;

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(nullptr, backendID);
    EXPECT_NE(nullptr, nncompiler);

    NNExecutor* ret = nncompiler->CreateExecutor();
    EXPECT_EQ(nullptr, ret);
}

/**
 * @tc.name: nncompilertest_createexecutor_003
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(NNCompilerTest, nncompilertest_createexecutor_003, TestSize.Level0)
{
    LOGE("CreateExecutor nncompilertest_createexecutor_003");
    size_t backendID = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::shared_ptr<MockIPreparedModel> prepared = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIDevice *) device.get()), IsModelCacheSupported(::testing::_))
        .WillOnce(Invoke([](bool& isSupportedCache) {
                // 这里直接修改传入的引用参数
                isSupportedCache = true;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    InnerModel innerModel;
    BuildModel(innerModel);
    void* model = &innerModel;
    EXPECT_CALL(*((MockIDevice *) device.get()), PrepareModel(testing::A<std::shared_ptr<const mindspore::lite::LiteGraph>>(), ::testing::_, ::testing::_))
        .WillOnce(Invoke([&prepared](std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                          const ModelConfig& config,
                                          std::shared_ptr<PreparedModel>& preparedModel) {
                // 这里直接修改传入的引用参数
                preparedModel = prepared;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    NNCompiler* nncompiler = new (std::nothrow) NNCompiler(model, device, backendID);
    EXPECT_NE(nullptr, nncompiler);

    OH_NN_ReturnCode retBuild = nncompiler->Build();
    EXPECT_EQ(OH_NN_SUCCESS, retBuild);

    NNExecutor* ret = nncompiler->CreateExecutor();
    EXPECT_NE(nullptr, ret);

    delete nncompiler;
    nncompiler = nullptr;

    testing::Mock::AllowLeak(device.get());
    testing::Mock::AllowLeak(prepared.get());
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS