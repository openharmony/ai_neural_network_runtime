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

#include "compilation_test.h"

#include <fstream>

#include "mindir.h"

#include "test/unittest/common/v1_0/mock_idevice.h"

using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::HDI::Nnrt::V1_0;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
OH_NN_ReturnCode CompilationTest::BuildModel(InnerModel& innerModel)
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

void CompilationTest::SetConfig(Compilation& compilationTest)
{
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetPerformance(OH_NN_PERFORMANCE_EXTREME));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetPriority(OH_NN_PRIORITY_HIGH));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetEnableFp16(true));
}

void CompilationTest::WriteFile(uint64_t version, uint64_t fileNumber, std::size_t cacheDeviceId)
{
    uint64_t cacheSize = 4;
    uint64_t writeSize = 7;
    uint64_t cacheInfo[7] = {};
    auto cacheInfoPtr = cacheInfo;
    *cacheInfoPtr++ = fileNumber;
    *cacheInfoPtr++ = version;
    *cacheInfoPtr++ = cacheDeviceId;
    for (uint64_t i = 0; i < cacheSize; ++i) {
        *cacheInfoPtr++ = i;
    }
    std::ofstream inFile("cache_info.nncache", std::ios::binary | std::ios::out | std::ios::trunc);
    inFile.write(reinterpret_cast<const char*>(cacheInfo), writeSize * sizeof(uint64_t));
    inFile.close();
}

void CompilationTest::BuildCompilation(InnerModel& innerModel)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.Build());
}

/*
 * @tc.name: compilation_set_device_001
 * @tc.desc: Verify the set deviceId after compilation finish of the SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_device_001, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    Compilation compilationTest(&innerModel);
    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.Build());

    std::size_t deviceId = 1;
    OH_NN_ReturnCode ret = compilationTest.SetDevice(deviceId);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_device_002
 * @tc.desc: Verify the deviceId does not exist of the SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_device_002, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);
    size_t deviceId = 0;
    OH_NN_ReturnCode ret = compilationTest.SetDevice(deviceId);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_device_003
 * @tc.desc: Verify the error happened when getting supported operation of the SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_device_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);
    std::size_t deviceId = 1;
    OH_NN_ReturnCode ret = compilationTest.SetDevice(deviceId);
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
}

/*
 * @tc.name: compilation_set_device_004
 * @tc.desc: Verify the current device not support the model of the SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_device_004, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);
    std::size_t deviceId = 1;
    MockIPreparedModel::m_ExpectRetCode = OH_NN_SUCCESS;
    OH_NN_ReturnCode ret = compilationTest.SetDevice(deviceId);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_set_device_005
 * @tc.desc: Verify the error happened when checking whether device supports dynamic input of the SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_device_005, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);
    std::size_t deviceId = 1;
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    OH_NN_ReturnCode ret = compilationTest.SetDevice(deviceId);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_set_device_006
 * @tc.desc: Verify the device does not support dynamic shape inputs of the SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_device_006, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);
    std::size_t deviceId = 1;
    MockIPreparedModel::m_ExpectRetCode = OH_NN_INVALID_PATH;
    OH_NN_ReturnCode ret = compilationTest.SetDevice(deviceId);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_set_device_007
 * @tc.desc: Verify the set normal deviceId of the SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_device_007, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    OH_NN_ReturnCode ret = compilationTest.SetDevice(deviceId);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_cachedir_001
 * @tc.desc: Verify the set cache after compilation finish of the SetCacheDir function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_cachedir_001, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.Build());

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, compilationTest.SetDevice(deviceId));

    OH_NN_ReturnCode ret = compilationTest.SetCacheDir("../", 1);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_cachedir_002
 * @tc.desc: Verify the not set device of the SetCacheDir function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_cachedir_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);
    OH_NN_ReturnCode ret = compilationTest.SetCacheDir("../", 1);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_cachedir_003
 * @tc.desc: Verify the Fail to query whether the device is available to save cache model of the SetCacheDir function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_cachedir_003, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    OH_NN_ReturnCode ret = compilationTest.SetCacheDir("../", 1);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_set_cachedir_004
 * @tc.desc: Verify the device is unavailable to save cache model of the SetCacheDir function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_cachedir_004, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_SUCCESS;
    OH_NN_ReturnCode ret = compilationTest.SetCacheDir("../", 1);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_cachedir_005
 * @tc.desc: Verify the cache model path is invalid of the SetCacheDir function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_cachedir_005, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    Compilation compilationTest(&innerModel);
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));
    OH_NN_ReturnCode ret = compilationTest.SetCacheDir("../compilation_test.cpp", 1);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_cachedir_006
 * @tc.desc: Verify the cache model path is not a directory of the SetCacheDir function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_cachedir_006, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));
    OH_NN_ReturnCode ret = compilationTest.SetCacheDir("./CompilationTest", 1);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_cachedir_007
 * @tc.desc: Verify the success of the SetCacheDir function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_cachedir_007, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    Compilation compilationTest(&innerModel);
    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));
    OH_NN_ReturnCode ret = compilationTest.SetCacheDir("../", 1);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_performance_001
 * @tc.desc: Verify the set performance after compilation finish of the SetPerformance function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_performance_001, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.Build());

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, compilationTest.SetDevice(deviceId));

    OH_NN_ReturnCode ret = compilationTest.SetPerformance(OH_NN_PERFORMANCE_NONE);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_performance_002
 * @tc.desc: Verify the set performance before set device of the SetPerformance function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_performance_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);
    OH_NN_ReturnCode ret = compilationTest.SetPerformance(OH_NN_PERFORMANCE_NONE);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_performance_003
 * @tc.desc: Verify the call device failed of the SetPerformance function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_performance_003, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    OH_NN_ReturnCode ret = compilationTest.SetPerformance(OH_NN_PERFORMANCE_NONE);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_set_performance_004
 * @tc.desc: Verify the device is not support performance setting of the SetPerformance function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_performance_004, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_SUCCESS;
    OH_NN_ReturnCode ret = compilationTest.SetPerformance(OH_NN_PERFORMANCE_NONE);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_performance_005
 * @tc.desc: Verify the passed invalid performance of the SetPerformance function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_performance_005, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    OH_NN_PerformanceMode performance = static_cast<OH_NN_PerformanceMode>(5);
    OH_NN_ReturnCode ret = compilationTest.SetPerformance(performance);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_performance_006
 * @tc.desc: Verify the success of the SetPerformance function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_performance_006, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    OH_NN_ReturnCode ret = compilationTest.SetPerformance(OH_NN_PERFORMANCE_NONE);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_priority_001
 * @tc.desc: Verify the set priority after compilation finish of the SetPriority function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_priority_001, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.Build());

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, compilationTest.SetDevice(deviceId));

    OH_NN_ReturnCode ret = compilationTest.SetPriority(OH_NN_PRIORITY_LOW);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_priority_002
 * @tc.desc: Verify the set priority before set device of the SetPriority function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_priority_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);

    OH_NN_ReturnCode ret = compilationTest.SetPriority(OH_NN_PRIORITY_LOW);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_priority_003
 * @tc.desc: Verify the call device failed of the SetPriority function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_priority_003, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_INVALID_PARAMETER;
    OH_NN_ReturnCode ret = compilationTest.SetPriority(OH_NN_PRIORITY_LOW);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_priority_004
 * @tc.desc: Verify the device is not support priority setting of the SetPriority function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_priority_004, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_SUCCESS;
    OH_NN_ReturnCode ret = compilationTest.SetPriority(OH_NN_PRIORITY_LOW);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_priority_005
 * @tc.desc: Verify the  passed invalid priority of the SetPriority function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_priority_005, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    OH_NN_Priority priority = static_cast<OH_NN_Priority>(5);;
    OH_NN_ReturnCode ret = compilationTest.SetPriority(priority);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_priority_006
 * @tc.desc: Verify the success of the SetPriority function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_priority_006, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    OH_NN_ReturnCode ret = compilationTest.SetPriority(OH_NN_PRIORITY_LOW);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_enable_fp16_001
 * @tc.desc: Verify the enable float16 after compilation finish of the SetEnableFp16 function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_enable_fp16_001, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.Build());

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, compilationTest.SetDevice(deviceId));

    OH_NN_ReturnCode ret = compilationTest.SetEnableFp16(true);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_enable_fp16_002
 * @tc.desc: Verify the set enable fp16 before set device of the SetEnableFp16 function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_enable_fp16_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);

    OH_NN_ReturnCode ret = compilationTest.SetEnableFp16(true);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_enable_fp16_003
 * @tc.desc: Verify the call device failed of the SetEnableFp16 function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_enable_fp16_003, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_MEMORY_ERROR;
    OH_NN_ReturnCode ret = compilationTest.SetEnableFp16(true);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, ret);
}

/*
 * @tc.name: compilation_set_enable_fp16_004
 * @tc.desc: Verify the device is not support float16 precision setting of the SetEnableFp16 function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_enable_fp16_004, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_SUCCESS;
    OH_NN_ReturnCode ret = compilationTest.SetEnableFp16(true);
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_set_enable_fp16_005
 * @tc.desc: Verify the success of the SetEnableFp16 function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_set_enable_fp16_005, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));

    OH_NN_ReturnCode ret = compilationTest.SetEnableFp16(true);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_get_input_tensors_001
 * @tc.desc: Verify the normal input tensors of the GetInputTensors function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_get_input_tensors_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);
    EXPECT_EQ(innerModel.GetInputTensors(), compilationTest.GetInputTensors());
}

/*
 * @tc.name: compilation_get_output_tensors_001
 * @tc.desc: Verify the normal output tensors of the GetOutputTensors function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_get_output_tensors_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);
    EXPECT_EQ(innerModel.GetOutputTensors(), compilationTest.GetOutputTensors());
}

/*
 * @tc.name: compilation_get_execution_plan_001
 * @tc.desc: Verify the passed nullptr of the GetExecutionPlan function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_get_execution_plan_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);
    EXPECT_EQ(nullptr, compilationTest.GetExecutionPlan());
}

/*
 * @tc.name: compilation_is_dynamic_shape_001
 * @tc.desc: Verify the input tensor is empth of the IsDynamicShape function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_is_dynamic_shape_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);
    EXPECT_EQ(false, compilationTest.IsDynamicShape());
}

/*
 * @tc.name: compilation_is_dynamic_shape_002
 * @tc.desc: Verify the return true of the IsDynamicShape function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_is_dynamic_shape_002, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    Compilation compilationTest(&innerModel);
    EXPECT_EQ(true, compilationTest.IsDynamicShape());
}

/*
 * @tc.name: compilation_is_dynamic_shape_003
 * @tc.desc: Verify the return false of the IsDynamicShape function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_is_dynamic_shape_003, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);
    EXPECT_EQ(false, compilationTest.IsDynamicShape());
}

/*
 * @tc.name: compilation_is_build_001
 * @tc.desc: Verify return false of the IsBuild function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_is_build_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    Compilation compilationTest(&innerModel);
    EXPECT_EQ(false, compilationTest.IsBuild());
}

/*
 * @tc.name: compilation_build_001
 * @tc.desc: Verify the build after compilation finish of the Build function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_001, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    Compilation compilationTest(&innerModel);
    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.Build());

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_build_002
 * @tc.desc: Verify the not set device of the Build function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_002, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_build_003
 * @tc.desc: Verify the preparing model failed of the Build function without set cache path.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_003, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    SetConfig(compilationTest);
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_build_004
 * @tc.desc: Verify the preparing model failed of the Build function without cache file.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_004, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);
    
    std::size_t deviceId = 1;
    MockIPreparedModel::m_ExpectRetCode = OH_NN_INVALID_FILE;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetEnableFp16(true));
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_build_005
 * @tc.desc: Verify the export model cache failed of the Build function without cache file.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_005, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetEnableFp16(true));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_build_006
 * @tc.desc: Verify the model cache file is invalid to generating cache mode of the Build function without cache file.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_006, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("/sys", 1));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetEnableFp16(true));
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(OH_NN_INVALID_FILE, ret);
}

/*
 * @tc.name: compilation_build_007
 * @tc.desc: Verify the success to generating cache mode of the Build function without cache file.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_007, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetEnableFp16(true));
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_build_008
 * @tc.desc: Verify the Fail to get the content of info cache file of the Build.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_008, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    std::ofstream createFile("cache_info.nncache");
    createFile.close();
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_INVALID_FILE, ret);
}

/*
 * @tc.name: compilation_build_009
 * @tc.desc: Verify the deviceId in the cache files is different from current deviceId of the Build function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_009, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);
    WriteFile(1, 4, 2);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_build_010
 * @tc.desc: Verify the info cache file has been changed of the Build function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_010, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);
    WriteFile(1, 100, 1);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_INVALID_FILE, ret);
}

/*
 * @tc.name: compilation_build_011
 * @tc.desc: Verify the Preparing model failed of the Build function model version is greater than cached versio.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_011, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    Compilation compilationTest(&innerModel);

    MockIPreparedModel::m_ExpectRetCode = OH_NN_INVALID_FILE;
    SetConfig(compilationTest);
    WriteFile(0, 4, 1);

    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    std::ofstream inFile("0.nncache", std::ios::binary | std::ios::out | std::ios::trunc);
    inFile.close();

    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_build_012
 * @tc.desc: Verify that the build function return success message with model version is greater than cached version
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_012, testing::ext::TestSize.Level0)
{
    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    Compilation compilationTest(&innerModel);
    MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
    SetConfig(compilationTest);
    WriteFile(0, 1, 1);

    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    std::ofstream inFile("0.nncache", std::ios::binary | std::ios::out | std::ios::trunc);
    inFile.close();

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_build_013
 * @tc.desc: Verify the model version is less than version cache of the Build function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_013, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);
    WriteFile(3, 4, 1);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);

    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_OPERATION_FORBIDDEN, ret);
}

/*
 * @tc.name: compilation_build_014
 * @tc.desc: Verify the checking cache model failed of the Build function with release buffer.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_014, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);
    EXPECT_EQ(0, remove("1.nncache"));

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_INVALID_FILE, ret);
}

/*
 * @tc.name: compilation_build_015
 * @tc.desc: Verify the get cache file length of the Build function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_015, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    std::ofstream inFile("0.nncache", std::ios::binary | std::ios::out | std::ios::trunc);
    inFile.close();

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_INVALID_FILE, ret);
}

/*
 * @tc.name: compilation_build_016
 * @tc.desc: Verify the fail to create file buffer of the Build function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_016, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_NULL_PTR;
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
}

/*
 * @tc.name: compilation_build_017
 * @tc.desc: Verify the cache model file has been changed of the Build function.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_017, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    uint64_t version = 1;
    uint64_t fileNumber = 1;
    std::size_t cacheDeviceId = 1;
    uint64_t cacheInfo[7] = {};
    auto cacheInfoPtr = cacheInfo;
    *cacheInfoPtr++ = fileNumber;
    *cacheInfoPtr++ = version;
    *cacheInfoPtr++ = cacheDeviceId;
    for (uint64_t i = 0; i < 4; ++i) {
        *cacheInfoPtr++ = i;
    }

    std::ofstream onFile("0.nncache", std::ios::binary | std::ios::out | std::ios::trunc);
    onFile.write(reinterpret_cast<const char*>(cacheInfo), 7 * sizeof(uint64_t));
    onFile.close();

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_INVALID_FILE, ret);
}

/*
 * @tc.name: compilation_build_018
 * @tc.desc: Verify the preparing model from cache failed of the Build function with load cache build.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_018, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_build_019
 * @tc.desc: Verify the success of the Build function with load cache build.
 * @tc.type: FUNC
 */
HWTEST_F(CompilationTest, compilation_build_019, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildCompilation(innerModel);

    Compilation compilationTest(&innerModel);
    SetConfig(compilationTest);
    EXPECT_EQ(OH_NN_SUCCESS, compilationTest.SetCacheDir("./", 1));

    OH_NN_ReturnCode ret = compilationTest.Build();
    EXPECT_EQ(0, remove("0.nncache"));
    EXPECT_EQ(0, remove("1.nncache"));
    EXPECT_EQ(0, remove("cache_info.nncache"));
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
