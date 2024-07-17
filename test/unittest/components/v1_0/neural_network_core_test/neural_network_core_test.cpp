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
#include <gmock/gmock.h>

#include "nnbackend.h"
#include "common/utils.h"
#include "neural_network_core_test.h"
#include "compilation.h"
#include "tensor.h"
#include "device.h"
#include "backend.h"
#include "backend_manager.h"
#include "backend_registrar.h"
#include "common/log.h"
#include "interfaces/kits/c/neural_network_runtime/neural_network_core.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Unittest {
const size_t SIZE_ONE = 1;
OH_NN_ReturnCode NeuralNetworkCoreTest::BuildModel(InnerModel& model)
{
    int32_t inputDims[2] = {3, 4};
    OH_NN_Tensor input1 = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    OH_NN_ReturnCode ret = model.AddTensor(input1);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 添加Add算子的第二个输入Tensor，类型为float32，张量形状为[3, 4]
    OH_NN_Tensor input2 = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    ret = model.AddTensor(input2);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 添加Add算子的参数Tensor，该参数Tensor用于指定激活函数的类型，Tensor的数据类型为int8。
    int32_t activationDims = 1;
    int8_t activationValue = OH_NN_FUSED_NONE;
    OH_NN_Tensor activation = {OH_NN_INT8, 1, &activationDims, nullptr, OH_NN_ADD_ACTIVATIONTYPE};
    ret = model.AddTensor(activation);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 将激活函数类型设置为OH_NN_FUSED_NONE，表示该算子不添加激活函数。
    uint32_t index = 2;
    ret = model.SetTensorValue(index, &activationValue, sizeof(int8_t));
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 设置Add算子的输出，类型为float32，张量形状为[3, 4]
    OH_NN_Tensor output = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    ret = model.AddTensor(output);
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
    ret = model.AddOperation(OH_NN_OPS_ADD, paramIndices, inputIndices, outputIndices);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 设置模型实例的输入、输出索引
    ret = model.SpecifyInputsAndOutputs(inputIndices, outputIndices);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 完成模型实例的构建
    ret = model.Build();
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    return ret;
}

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

class MockBackend : public Backend {
public:
    MOCK_CONST_METHOD0(GetBackendID, size_t());
    MOCK_CONST_METHOD1(GetBackendName, OH_NN_ReturnCode(std::string&));
    MOCK_CONST_METHOD1(GetBackendType, OH_NN_ReturnCode(OH_NN_DeviceType&));
    MOCK_CONST_METHOD1(GetBackendStatus, OH_NN_ReturnCode(DeviceStatus&));
    MOCK_METHOD1(CreateCompiler, Compiler*(Compilation*));
    MOCK_METHOD1(DestroyCompiler, OH_NN_ReturnCode(Compiler*));
    MOCK_METHOD1(CreateExecutor, Executor*(Compilation*));
    MOCK_METHOD1(DestroyExecutor, OH_NN_ReturnCode(Executor*));
    MOCK_METHOD1(CreateTensor, Tensor*(TensorDesc*));
    MOCK_METHOD1(DestroyTensor, OH_NN_ReturnCode(Tensor*));
    MOCK_METHOD2(GetSupportedOperation, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
                                           std::vector<bool>&));
};

std::shared_ptr<Backend> Creator4()
{
    size_t backendID = 4;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceStatus(::testing::_))
        .WillRepeatedly(::testing::Invoke([](DeviceStatus& status) {
                // 这里直接修改传入的引用参数
                status = AVAILABLE;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    std::string backendName = "mock";
    EXPECT_CALL(*((MockIDevice *) device.get()), GetDeviceName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVendorName(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), GetVersion(::testing::_))
        .WillRepeatedly(::testing::DoAll(::testing::SetArgReferee<0>(backendName), ::testing::Return(OH_NN_SUCCESS)));

    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateBuffer(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Invoke([](size_t length, int& fd) {
                // 这里直接修改传入的引用参数
                fd = -1;
                return OH_NN_SUCCESS; // 假设成功的状态码
            }));

    std::shared_ptr<Backend> backend = std::make_unique<NNBackend>(device, backendID);

    testing::Mock::AllowLeak(device.get());

    return backend;
}

/*
 * @tc.name: alldevicesid_001
 * @tc.desc: Verify the allDeviceIds is nullptr of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, alldevicesid_001, testing::ext::TestSize.Level0)
{
    const size_t* allDeviceIds = nullptr;
    uint32_t count {0};
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(&allDeviceIds, &count);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: alldeviceid_002
 * @tc.desc: Verify the allDeviceIds is nullptr of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, alldeviceid_002, testing::ext::TestSize.Level0)
{
    uint32_t count {0};
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(nullptr, &count);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: alldeviceid_003
 * @tc.desc: Verify the allDeviceIds is nullptr of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, alldeviceid_003, testing::ext::TestSize.Level0)
{
    const size_t* allDeviceIds = nullptr;
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(&allDeviceIds, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: alldeviceid_004
 * @tc.desc: Verify the allDeviceIds is nullptr of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, alldeviceid_004, testing::ext::TestSize.Level0)
{
    const size_t allDeviceIds = 0;
    const size_t* pAllDeviceIds = &allDeviceIds;
    uint32_t count {0};

    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(&pAllDeviceIds, &count);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_name_001
 * @tc.desc: Verify the name is nullptr of the OH_NNDevice_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, device_name_001, testing::ext::TestSize.Level0)
{
    const size_t deviceId = 0;
    const char* name = nullptr;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceId, &name);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: device_name_002
 * @tc.desc: Verify the name is no nullptr of the OH_NNDevice_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, device_name_002, testing::ext::TestSize.Level0)
{
    const size_t deviceId = 0;
    const char* name = "name";
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceId, &name);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_name_003
 * @tc.desc: Verify the name is nullptr of the OH_NNDevice_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, device_name_003, testing::ext::TestSize.Level0)
{
    const size_t deviceId = 0;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceId, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_type_001
 * @tc.desc: Verify the device is nullptr of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, device_get_type_001, testing::ext::TestSize.Level0)
{
    size_t deviceID = 0;
    OH_NN_DeviceType deviceType = OH_NN_CPU;
    OH_NN_DeviceType* pDeviceType = &deviceType;
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceID, pDeviceType);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_type_002
 * @tc.desc: Verify the device is nullptr of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, device_get_type_002, testing::ext::TestSize.Level0)
{
    size_t deviceID = 0;
    OH_NN_DeviceType* pDeviceType = nullptr;
    BackendManager& backendManager = BackendManager::GetInstance();
    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator4;

    backendManager.RegisterBackend(backendName, creator);
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceID, pDeviceType);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_type_003
 * @tc.desc: Verify the device is nullptr of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, device_get_type_003, testing::ext::TestSize.Level0)
{
    size_t deviceID = 0;
    OH_NN_DeviceType deviceType = OH_NN_OTHERS;
    OH_NN_DeviceType* pDeviceType = &deviceType;
    BackendManager& backendManager = BackendManager::GetInstance();
    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator4;

    backendManager.RegisterBackend(backendName, creator);
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceID, pDeviceType);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: device_get_type_004
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, device_get_type_004, testing::ext::TestSize.Level0)
{
    size_t deviceID =  1;
    OH_NN_DeviceType deviceType = OH_NN_CPU;
    OH_NN_DeviceType* pDeviceType = &deviceType;
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceID, pDeviceType);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_construct_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNCompilation_Construct function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_construct_001, testing::ext::TestSize.Level0)
{
    const OH_NNModel* model = nullptr;
    OH_NNCompilation* ret = OH_NNCompilation_Construct(model);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_002
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNCompilation_Construct function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_construct_002, testing::ext::TestSize.Level0)
{
    const OH_NNModel* model = OH_NNModel_Construct();
    OH_NNCompilation* ret = OH_NNCompilation_Construct(model);
    EXPECT_NE(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_with_off_modelfile_001
 * @tc.desc: Verify the modelpath is nullptr of the OH_NNCompilation_ConstructWithOfflineModelFile function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_construct_with_off_modelfile_001, testing::ext::TestSize.Level0)
{
    const char* modelpath = nullptr;
    OH_NNCompilation* ret = OH_NNCompilation_ConstructWithOfflineModelFile(modelpath);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_with_off_modelfile_002
 * @tc.desc: Verify the modelpath is no nullptr of the OH_NNCompilation_ConstructWithOfflineModelFile function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_construct_with_off_modelfile_002, testing::ext::TestSize.Level0)
{
    const char* modelpath = "nrrtmodel";
    OH_NNCompilation* ret = OH_NNCompilation_ConstructWithOfflineModelFile(modelpath);
    EXPECT_NE(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_with_off_modelbuffer_001
 * @tc.desc: Verify the modelbuffer is nullptr of the OH_NNCompilation_ConstructWithOfflineModelBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_construct_with_off_modelbuffer_001, testing::ext::TestSize.Level0)
{
    const void* modelbuffer = nullptr;
    size_t modelsize = 0;
    OH_NNCompilation* ret = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelbuffer, modelsize);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_with_off_modelbuffer_002
 * @tc.desc: Verify the modelbuffer is no nullptr of the OH_NNCompilation_ConstructWithOfflineModelBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_construct_with_off_modelbuffer_002, testing::ext::TestSize.Level0)
{
    char modelbuffer[SIZE_ONE];
    size_t modelsize = 0;
    OH_NNCompilation* ret = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelbuffer, modelsize);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_with_off_modelbuffer_003
 * @tc.desc: Verify the modelbuffer is no nullptr of the OH_NNCompilation_ConstructWithOfflineModelBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_construct_with_off_modelbuffer_003, testing::ext::TestSize.Level0)
{
    char modelbuffer[SIZE_ONE];
    size_t modelsize = 1;
    OH_NNCompilation* ret = OH_NNCompilation_ConstructWithOfflineModelBuffer(modelbuffer, modelsize);
    EXPECT_NE(nullptr, ret);
}

/*
 * @tc.name: compilation_constructforcache_001
 * @tc.desc: Verify the nnCompilation is no nullptr of the OH_NNCompilation_ConstructForCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_constructforcache_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* ret = OH_NNCompilation_ConstructForCache();
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    delete compilation;
    EXPECT_NE(nnCompilation, ret);
}

/*
 * @tc.name: compilation_exportchachetobuffer_001
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_exportchachetobuffer_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* compilation = nullptr;
    const void* buffer = nullptr;
    size_t length = 0;
    size_t* modelSize = nullptr;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(compilation, buffer, length, modelSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_exportchachetobuffer_002
 * @tc.desc: Verify the buffer is nullptr of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_exportchachetobuffer_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    const void* buffer = nullptr;
    size_t length = 0;
    size_t* modelSize = nullptr;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(nnCompilation, buffer, length, modelSize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_exportchachetobuffer_003
 * @tc.desc: Verify the length is 0 of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_exportchachetobuffer_003, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    char buffer[SIZE_ONE];
    size_t length = 0;
    size_t* modelSize = nullptr;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(nnCompilation, buffer, length, modelSize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_exportchachetobuffer_004
 * @tc.desc: Verify the modelSize is nullptr of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_exportchachetobuffer_004, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    char buffer[SIZE_ONE];
    size_t length = 0;
    size_t* modelSize = nullptr;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(nnCompilation, buffer, length, modelSize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_exportchachetobuffer_005
 * @tc.desc: Verify the modelSize is nullptr of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_exportchachetobuffer_005, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    char buffer[SIZE_ONE];
    size_t* modelSize = nullptr;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(nnCompilation, buffer, SIZE_ONE, modelSize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_exportchachetobuffer_006
 * @tc.desc: Verify the length is 0 of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_exportchachetobuffer_006, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    char buffer[SIZE_ONE];
    size_t modelSize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(nnCompilation, buffer, SIZE_ONE, &modelSize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_exportchachetobuffer_007
 * @tc.desc: Verify the length is 0 of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_exportchachetobuffer_007, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    char buffer[SIZE_ONE];
    size_t modelSize = 0;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    size_t backid = 1;

    NNCompiler nnCompiler(device, backid);
    compilation->compiler = &nnCompiler;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(nnCompilation, buffer, SIZE_ONE, &modelSize);
    delete compilation;
    EXPECT_NE(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: compilation_importcachefrombuffer_001
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_ImportCacheFromBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_importcachefrombuffer_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* compilation = nullptr;
    const void* buffer = nullptr;
    size_t modelsize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_ImportCacheFromBuffer(compilation, buffer, modelsize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_importcachefrombuffer_002
 * @tc.desc: Verify the buffer is nullptr of the OH_NNCompilation_ImportCacheFromBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_importcachefrombuffer_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    const void* buffer = nullptr;
    size_t modelsize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_ImportCacheFromBuffer(nnCompilation, buffer, modelsize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_importcachefrombuffer_003
 * @tc.desc: Verify the modelsize is 0 of the OH_NNCompilation_ImportCacheFromBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_importcachefrombuffer_003, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    char buffer[SIZE_ONE];
    size_t modelsize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_ImportCacheFromBuffer(nnCompilation, buffer, modelsize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_importcachefrombuffer_004
 * @tc.desc: Verify the modelsize is 0 of the OH_NNCompilation_ImportCacheFromBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_importcachefrombuffer_004, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    char buffer[SIZE_ONE];
    OH_NN_ReturnCode ret = OH_NNCompilation_ImportCacheFromBuffer(nnCompilation, buffer, SIZE_ONE);
    delete compilation;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_addextensionconfig_001
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_AddExtensionConfig function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_addextensionconfig_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* compilation = nullptr;
    const char* configname = nullptr;
    const void* configvalue = nullptr;
    const size_t configvaluesize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(compilation, configname, configvalue, configvaluesize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_addextensionconfig_002
 * @tc.desc: Verify the configname is nullptr of the OH_NNCompilation_AddExtensionConfig function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_addextensionconfig_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    const char* configname = nullptr;
    const void* cofigvalue = nullptr;
    const size_t configvaluesize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(nnCompilation, configname, cofigvalue, configvaluesize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_addextensionconfig_003
 * @tc.desc: Verify the cofigvalue is nullptr of the OH_NNCompilation_AddExtensionConfig function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_addextensionconfig_003, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    const char* configname = "ConfigName";
    const void* cofigvalue = nullptr;
    const size_t configvaluesize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(nnCompilation, configname, cofigvalue, configvaluesize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_addextensionconfig_004
 * @tc.desc: Verify the cofigvalue is nullptr of the OH_NNCompilation_AddExtensionConfig function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_addextensionconfig_004, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    const char* configname = "ConfigName";
    char cofigvalue[SIZE_ONE];
    const size_t configvaluesize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(nnCompilation, configname, cofigvalue, configvaluesize);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_addextensionconfig_005
 * @tc.desc: Verify the cofigvalue is nullptr of the OH_NNCompilation_AddExtensionConfig function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_addextensionconfig_005, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    const char* configname = "ConfigName";
    char cofigvalue[SIZE_ONE];
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(nnCompilation, configname, cofigvalue, SIZE_ONE);
    delete compilation;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_device_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_device_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* compilation = nullptr;
    size_t deviceId = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetDevice(compilation, deviceId);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_device_002
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_device_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    size_t deviceId = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetDevice(nnCompilation, deviceId);
    delete compilation;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_cache_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_cache_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* nnCompilation = nullptr;
    const char* cacheDir = "../";
    uint32_t version = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetCache(nnCompilation, cacheDir, version);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_cache_002
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_cache_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    const char* cacheDir = nullptr;
    uint32_t version = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetCache(nnCompilation, cacheDir, version);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_cache_003
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_cache_003, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    const char* cacheDir = "../";
    uint32_t version = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetCache(nnCompilation, cacheDir, version);
    delete compilation;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_performancemode_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_performancemode_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    OH_NNCompilation* nnCompilation = nullptr;
    OH_NN_PerformanceMode performanceMode = OH_NN_PERFORMANCE_NONE;

    OH_NN_ReturnCode ret = OH_NNCompilation_SetPerformanceMode(nnCompilation, performanceMode);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_performancemode_002
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_performancemode_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    OH_NN_PerformanceMode performanceMode = OH_NN_PERFORMANCE_NONE;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetPerformanceMode(nnCompilation, performanceMode);
    delete compilation;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_priority_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_priority_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* nnCompilation = nullptr;
    OH_NN_Priority priority = OH_NN_PRIORITY_NONE;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetPriority(nnCompilation, priority);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_priority_002
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_set_priority_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    OH_NN_Priority priority = OH_NN_PRIORITY_NONE;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetPriority(nnCompilation, priority);
    delete compilation;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_enablefloat16_001
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_enablefloat16_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* nnCompilation = nullptr;
    bool enableFloat16 = true;
    OH_NN_ReturnCode ret = OH_NNCompilation_EnableFloat16(nnCompilation, enableFloat16);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_enablefloat16_002
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_enablefloat16_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    bool enableFloat16 = true;
    OH_NN_ReturnCode ret = OH_NNCompilation_EnableFloat16(nnCompilation, enableFloat16);
    delete compilation;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_build_001
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_build_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation *nncompilation = nullptr;
    OH_NN_ReturnCode ret = OH_NNCompilation_Build(nncompilation);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_build_002
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_build_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    OH_NN_ReturnCode ret = OH_NNCompilation_Build(nnCompilation);
    delete compilation;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_build_002
 * @tc.desc: Verify the success of the OH_NNCompilation_Build function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_build_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);

    OH_NN_ReturnCode ret = OH_NNCompilation_Build(nnCompilation);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: nnt_tensordesc_destroy_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_destroy_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_Destroy(&tensorDesc);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_destroy_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_destroy_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    OH_NN_ReturnCode ret = OH_NNTensorDesc_Destroy(&tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: nnt_tensordesc_setname_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setname_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    const char* name = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetName(tensorDesc, name);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setname_002
 * @tc.desc: Verify the name is nullptr of the OH_NNTensorDesc_SetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setname_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    const char* name = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetName(tensorDesc, name);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setname_003
 * @tc.desc: Verify the name is nullptr of the OH_NNTensorDesc_SetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setname_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    const char* name = "name";
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetName(tensorDesc, name);
    EXPECT_NE(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getname_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getname_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    const char* name = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetName(tensorDesc, &name);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getname_002
 * @tc.desc: Verify the name is nullptr of the OH_NNTensorDesc_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getname_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    const char* name = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetName(tensorDesc, &name);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: nnt_tensordesc_getname_003
 * @tc.desc: Verify the name is nullptr of the OH_NNTensorDesc_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getname_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    const char* name = "name";
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetName(tensorDesc, &name);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setdatatype_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetDataType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setdatatype_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    OH_NN_DataType datatype = OH_NN_UNKNOWN;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetDataType(tensorDesc, datatype);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setdatatype_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetDataType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setdatatype_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    OH_NN_DataType datatype = OH_NN_UNKNOWN;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetDataType(tensorDesc, datatype);
    EXPECT_NE(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getdatatype_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_GetDataType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getdatatype_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    OH_NN_DataType* datatype = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetDataType(tensorDesc, datatype);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getdatatype_002
 * @tc.desc: Verify the OH_NN_DataType is nullptr of the OH_NNTensorDesc_GetDataType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getdatatype_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    OH_NN_DataType datatype = OH_NN_BOOL;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetDataType(tensorDesc, &datatype);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: nnt_tensordesc_getdatatype_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_GetDataType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getdatatype_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    OH_NN_DataType* datatype = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetDataType(tensorDesc, datatype);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setshape_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setshape_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    const int32_t* shape = nullptr;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetShape(tensorDesc, shape, shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setshape_002
 * @tc.desc: Verify the shape is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setshape_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    const int32_t* shape = nullptr;
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetShape(tensorDesc, shape, shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setshape_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setshape_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    int32_t inputDims[4] = {1, 2, 2, 3};
    size_t shapeLength = 0;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetShape(tensorDesc, inputDims, shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setshape_004
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setshape_004, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    int32_t inputDims[4] = {1, 2, 2, 3};
    size_t shapeLength = 1;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetShape(tensorDesc, inputDims, shapeLength);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: nnt_tensordesc_Getshape_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_Getshape_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    int32_t* shape = nullptr;
    size_t* shapeLength = 0;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetShape(tensorDesc, &shape, shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_Getshape_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_Getshape_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    int32_t* shape = nullptr;
    size_t* shapeLength = 0;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetShape(tensorDesc, &shape, shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_Getshape_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_Getshape_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    int32_t* shape = nullptr;
    int lengthValue = 1;
    size_t* shapeLength = new size_t(lengthValue);
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetShape(tensorDesc, &shape, shapeLength);
    delete shapeLength;
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: nnt_tensordesc_setformat_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setformat_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    OH_NN_Format format = static_cast<OH_NN_Format>(OH_NN_FLOAT32);
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetFormat(tensorDesc, format);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_setformat_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_setformat_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    OH_NN_Format format = static_cast<OH_NN_Format>(OH_NN_FLOAT32);
    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetFormat(tensorDesc, format);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getformat_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getformat_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    OH_NN_Format* format = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetFormat(tensorDesc, format);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getformat_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getformat_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    OH_NN_Format* format = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetFormat(tensorDesc, format);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getformat_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getformat_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    OH_NN_Format format = OH_NN_FORMAT_NONE;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetFormat(tensorDesc, &format);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: nnt_tensordesc_getelementcount_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getelementcount_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    size_t* elementCount = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetElementCount(tensorDesc, elementCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getelementcount_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getelementcount_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t* elementCount = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetElementCount(tensorDesc, elementCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getelementcount_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getelementcount_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t elementCount = 0;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetElementCount(tensorDesc, &elementCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getelementcount_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getbytesize_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    size_t* byteSize = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetByteSize(tensorDesc, byteSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getelementcount_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getbytesize_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t* byteSize = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetByteSize(tensorDesc, byteSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getelementcount_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getbytesize_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t byteSize = 0;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetByteSize(tensorDesc, &byteSize);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nntensor_create_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_create_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    size_t deviceid = 0;
    NN_Tensor* ret = OH_NNTensor_Create(deviceid, tensorDesc);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_create_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_create_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 1;
    NN_Tensor* ret = OH_NNTensor_Create(deviceid, tensorDesc);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_create_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_create_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 0;
    BackendManager& backendManager = BackendManager::GetInstance();
    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator4;

    backendManager.RegisterBackend(backendName, creator);
    NN_Tensor* ret = OH_NNTensor_Create(deviceid, tensorDesc);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithsize_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    size_t deviceid = 0;
    size_t size = 0;
    NN_Tensor* ret = OH_NNTensor_CreateWithSize(deviceid, tensorDesc, size);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithsize_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 1;
    size_t size = 0;
    NN_Tensor* ret = OH_NNTensor_CreateWithSize(deviceid, tensorDesc, size);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithsize_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 0;
    size_t size = 0;
    BackendManager& backendManager = BackendManager::GetInstance();
    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator4;

    backendManager.RegisterBackend(backendName, creator);
    NN_Tensor* ret = OH_NNTensor_CreateWithSize(deviceid, tensorDesc, size);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_001
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithfd_001, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    size_t deviceid = 0;
    int fd = 0;
    size_t size = 0;
    size_t offset = 0;
    NN_Tensor* ret = OH_NNTensor_CreateWithFd(deviceid, tensorDesc, fd, size, offset);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_002
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithfd_002, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 0;
    int fd = -1;
    size_t size = 0;
    size_t offset = 0;
    NN_Tensor* ret = OH_NNTensor_CreateWithFd(deviceid, tensorDesc, fd, size, offset);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithfd_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 0;
    int fd = 1;
    size_t size = 0;
    size_t offset = 0;
    NN_Tensor* ret = OH_NNTensor_CreateWithFd(deviceid, tensorDesc, fd, size, offset);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_004
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithfd_004, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 0;
    int fd = 1;
    size_t size = 1;
    size_t offset = 2;
    NN_Tensor* ret = OH_NNTensor_CreateWithFd(deviceid, tensorDesc, fd, size, offset);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_005
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithfd_005, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 0;
    int fd = 1;
    size_t size = 1;
    size_t offset = 0;
    NN_Tensor* ret = OH_NNTensor_CreateWithFd(deviceid, tensorDesc, fd, size, offset);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_createwithsize_006
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_createwithsize_006, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = OH_NNTensorDesc_Create();
    size_t deviceid = 0;
    int fd = 1;
    size_t size = 1;
    size_t offset = 2;
    NN_Tensor* ret = OH_NNTensor_CreateWithFd(deviceid, tensorDesc, fd, size, offset);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_destroy_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_destroy_001, testing::ext::TestSize.Level0)
{
    NN_Tensor* tensor = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensor_Destroy(&tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nntensor_destroy_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_destroy_002, testing::ext::TestSize.Level0)
{
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    OH_NN_ReturnCode ret = OH_NNTensor_Destroy(&tensor);
    EXPECT_EQ(OH_NN_NULL_PTR, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nntensor_gettensordesc_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_gettensordesc_001, testing::ext::TestSize.Level0)
{
    const NN_Tensor* tensor = nullptr;
    NN_TensorDesc* ret = OH_NNTensor_GetTensorDesc(tensor);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_gettensordesc_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_gettensordesc_002, testing::ext::TestSize.Level0)
{
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    NN_TensorDesc* ret = OH_NNTensor_GetTensorDesc(tensor);
    EXPECT_NE(nullptr, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nntensor_getdatabuffer_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getdatabuffer_001, testing::ext::TestSize.Level0)
{
    const NN_Tensor* tensor = nullptr;
    void* ret = OH_NNTensor_GetDataBuffer(tensor);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nntensor_getdatabuffer_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getdatabuffer_002, testing::ext::TestSize.Level0)
{
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    void* ret = OH_NNTensor_GetDataBuffer(tensor);
    EXPECT_EQ(nullptr, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nntensor_getsize_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getsize_001, testing::ext::TestSize.Level0)
{
    const NN_Tensor* tensor = nullptr;
    size_t* size = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensor_GetSize(tensor, size);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nntensor_getsize_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getsize_002, testing::ext::TestSize.Level0)
{
    size_t* size = nullptr;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    OH_NN_ReturnCode ret = OH_NNTensor_GetSize(tensor, size);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nntensor_getsize_003
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getsize_003, testing::ext::TestSize.Level0)
{
    size_t size = 1;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    OH_NN_ReturnCode ret = OH_NNTensor_GetSize(tensor, &size);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nntensor_getfd_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getfd_001, testing::ext::TestSize.Level0)
{
    const NN_Tensor* tensor = nullptr;
    int* fd = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensor_GetFd(tensor, fd);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nntensor_getfd_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getfd_002, testing::ext::TestSize.Level0)
{
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    int* fd = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensor_GetFd(tensor, fd);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nntensor_getfd_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getfd_003, testing::ext::TestSize.Level0)
{
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    int fd = 1;
    OH_NN_ReturnCode ret = OH_NNTensor_GetFd(tensor, &fd);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nntensor_getoffset_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getoffset_001, testing::ext::TestSize.Level0)
{
    const NN_Tensor* tensor = nullptr;
    size_t* offset = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensor_GetOffset(tensor, offset);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nntensor_getoffset_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getoffset_002, testing::ext::TestSize.Level0)
{
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    size_t* offset = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensor_GetOffset(tensor, offset);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nntensor_getoffset_003
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nntensor_getoffset_003, testing::ext::TestSize.Level0)
{
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    std::unique_ptr<NNBackend> hdiDevice = std::make_unique<NNBackend>(device, backendID);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(hdiDevice->CreateTensor(tensorDesc));
    size_t offset = 1;
    OH_NN_ReturnCode ret = OH_NNTensor_GetOffset(tensor, &offset);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nnexecutor_getputputshape_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getputputshape_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* executor = nullptr;
    uint32_t outputIndex = 0;
    int32_t* shape = nullptr;
    uint32_t* shapeLength = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetOutputShape(executor, outputIndex, &shape, shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nnexecutor_getputputshape_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getputputshape_002, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    uint32_t outputIndex = 0;
    int32_t* shape = nullptr;
    uint32_t* shapeLength = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetOutputShape(nnExecutor, outputIndex, &shape, shapeLength);
    delete executor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nnexecutor_getputputshape_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getinputcount_001, testing::ext::TestSize.Level0)
{
    const OH_NNExecutor* executor = nullptr;
    size_t* inputCount = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetInputCount(executor, inputCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nnexecutor_getinputcount_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getinputcount_002, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    size_t* inputCount = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetInputCount(nnExecutor, inputCount);
    delete executor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nnexecutor_getoutputcount_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getoutputcount_001, testing::ext::TestSize.Level0)
{
    const OH_NNExecutor* executor = nullptr;
    size_t* outputCount = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetOutputCount(executor, outputCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nnexecutor_getoutputcount_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getoutputcount_002, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    size_t* outputCount = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetOutputCount(nnExecutor, outputCount);
    delete executor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nnexecutor_createinputtensordesc_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_createinputtensordesc_001, testing::ext::TestSize.Level0)
{
    const OH_NNExecutor* executor = nullptr;
    size_t index = 1;
    NN_TensorDesc* ret = OH_NNExecutor_CreateInputTensorDesc(executor, index);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nnexecutor_createinputtensordesc_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_createouttensordesc_001, testing::ext::TestSize.Level0)
{
    const OH_NNExecutor* executor = nullptr;
    size_t index = 1;
    NN_TensorDesc* ret = OH_NNExecutor_CreateOutputTensorDesc(executor, index);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nnexecutor_getoutputcount_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getinputdimRange_001, testing::ext::TestSize.Level0)
{
    const OH_NNExecutor* executor = nullptr;
    size_t index = 1;
    size_t* minInputDims = nullptr;
    size_t* maxInputDims = nullptr;
    size_t* shapeLength = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetInputDimRange(executor, index, &minInputDims, &maxInputDims, shapeLength);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nnexecutor_getoutputcount_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getinputdimRange_002, testing::ext::TestSize.Level0)
{
    size_t index = 1;
    size_t* minInputDims = nullptr;
    size_t* maxInputDims = nullptr;
    size_t* shapeLength = nullptr;
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
    m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    OH_NN_ReturnCode ret = OH_NNExecutor_GetInputDimRange(nnExecutor, index,
    &minInputDims, &maxInputDims, shapeLength);
    delete executor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

 /*
 * @tc.name: nnt_nnexecutor_getinputdimRange_003
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_getinputdimRange_003, testing::ext::TestSize.Level0)
{
    size_t index = 1;
    size_t mindims = 1;
    size_t* minInputDims = &mindims;
    size_t* maxInputDims = nullptr;
    size_t* shapeLength = nullptr;
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
    m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    OH_NN_ReturnCode ret = OH_NNExecutor_GetInputDimRange(nnExecutor, index,
    &minInputDims, &maxInputDims, shapeLength);
    delete executor;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nnexecutor_setonrundone_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_setonrundone_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* executor = nullptr;
    NN_OnRunDone rundone = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOnRunDone(executor, rundone);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nnexecutor_setonrundone_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_setonrundone_002, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_OnRunDone rundone = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOnRunDone(nnExecutor, rundone);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nnexecutor_setonservicedied_001
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_setonservicedied_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* executor = nullptr;
    NN_OnServiceDied servicedied = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOnServiceDied(executor, servicedied);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_nnexecutor_setonservicedied_002
 * @tc.desc: Verify the NN_Tensor is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_setonservicedied_002, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_OnServiceDied servicedied = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOnServiceDied(nnExecutor, servicedied);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runsync_001
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runsync_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* executor = nullptr;
    NN_Tensor* inputTensor[] = {nullptr};
    size_t inputCount = 0;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunSync(executor, inputTensor, inputCount, outputTensor, outputcount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_executor_runsync_002
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runsync_002, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[] = {nullptr};
    size_t inputCount = 0;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunSync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runsync_003
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runsync_003, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[sizetensor];
    size_t inputCount = 0;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunSync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runsync_004
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runsync_004, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[sizetensor];
    size_t inputCount = 1;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunSync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runsync_005
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runsync_005, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[sizetensor];
    size_t inputCount = 1;
    NN_Tensor* outputTensor[sizetensor];
    size_t outputcount = 0;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunSync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runasync_001
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runasync_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* executor = nullptr;
    NN_Tensor* inputTensor[] = {nullptr};
    size_t inputCount = 0;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    int32_t timeout = 1;
    void* userdata = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunAsync(executor, inputTensor, inputCount, outputTensor, outputcount,
        timeout, userdata);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_executor_runasync_002
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runasync_002, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[] = {nullptr};
    size_t inputCount = 0;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    int32_t timeout = 1;
    void* userdata = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunAsync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount,
        timeout, userdata);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runasync_003
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runasync_003, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[sizetensor];
    size_t inputCount = 0;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    int32_t timeout = 1;
    void* userdata = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunAsync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount,
        timeout, userdata);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runasync_004
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runasync_004, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[sizetensor];
    size_t inputCount = 0;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    int32_t timeout = 1;
    void* userdata = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunAsync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount,
        timeout, userdata);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runasync_005
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runasync_005, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[sizetensor];
    size_t inputCount = 1;
    NN_Tensor* outputTensor[] = {nullptr};
    size_t outputcount = 0;
    int32_t timeout = 1;
    void* userdata = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunAsync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount,
        timeout, userdata);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runasync_006
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runasync_006, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[sizetensor];
    size_t inputCount = 1;
    NN_Tensor* outputTensor[sizetensor];
    size_t outputcount = 0;
    int32_t timeout = 1;
    void* userdata = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunAsync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount,
        timeout, userdata);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_executor_runasync_007
 * @tc.desc: Verify the ExecutorConfig is nullptr of the OH_NNTensorDesc_SetShape function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_executor_runasync_007, testing::ext::TestSize.Level0)
{
    size_t m_backendID {0};
    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);
    NN_Tensor* inputTensor[sizetensor];
    size_t inputCount = 1;
    NN_Tensor* outputTensor[sizetensor];
    size_t outputcount = 1;
    int32_t timeout = 1;
    void* userdata = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_RunAsync(nnExecutor, inputTensor, inputCount, outputTensor, outputcount,
        timeout, userdata);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: nnt_nnexecutor_construct_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_construct_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* nnCompilation = nullptr;
    OH_NNExecutor* ret = OH_NNExecutor_Construct(nnCompilation);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nnexecutor_construct_002
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_construct_002, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation *>(compilation);
    EXPECT_NE(nnCompilation, nullptr);
    BackendManager& backendManager = BackendManager::GetInstance();
    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator4;

    BackendRegistrar backendregistrar(backendName, creator);
    backendManager.RemoveBackend(backendName);
    backendManager.RegisterBackend(backendName, creator);
    OH_NNExecutor* ret = OH_NNExecutor_Construct(nnCompilation);
    delete compilation;
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: nnt_nnexecutor_construct_003
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_nnexecutor_construct_003, testing::ext::TestSize.Level0)
{
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation *>(compilation);
    EXPECT_NE(nnCompilation, nullptr);
    BackendManager& backendManager = BackendManager::GetInstance();
    std::string backendName = "mock";
    std::function<std::shared_ptr<Backend>()> creator = Creator4;

    backendManager.RegisterBackend(backendName, creator);
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();
    size_t backid = 1;

    NNCompiler nnCompiler(device, backid);
    compilation->compiler = &nnCompiler;
    OH_NNExecutor* ret = OH_NNExecutor_Construct(nnCompilation);
    delete compilation;
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: compilation_destroy_001
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_destroy_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* nncompilation = nullptr;
    OH_NNCompilation_Destroy(&nncompilation);
    EXPECT_EQ(nullptr, nncompilation);
}

/*
 * @tc.name: compilation_destroy_002
 * @tc.desc: Verify the normal model of the OH_NNCompilation_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_destroy_002, testing::ext::TestSize.Level0)
{
    InnerModel* innerModel = new InnerModel();
    EXPECT_NE(nullptr, innerModel);

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNCompilation_Destroy(&nnCompilation);
    EXPECT_EQ(nullptr, nnCompilation);
}

/*
 * @tc.name: executor_destroy_001
 * @tc.desc: Verify the compilation is nullptr of the OH_NNCompilation_ExportCacheToBuffer function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, executor_destroy_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* nnExecutor = nullptr;
    OH_NNExecutor_Destroy(&nnExecutor);
    EXPECT_EQ(nullptr, nnExecutor);
}
} // Unittest
} // namespace NeuralNetworkRuntime
} // namespace OHOS