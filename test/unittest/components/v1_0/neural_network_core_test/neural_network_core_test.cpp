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

#include "common/utils.h"
#include "neural_network_core_test.h"
#include "compilation.h"
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
 * @tc.name: compilation_constructforcache_001
 * @tc.desc: Verify the nnCompilation is no nullptr of the OH_NNCompilation_ConstructForCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, compilation_constructforcache_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* ret = OH_NNCompilation_ConstructForCache();
    Compilation *compilation = new (std::nothrow) Compilation();
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
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
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
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
    OH_NN_DataType* datatype = nullptr;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetDataType(tensorDesc, datatype);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: nnt_tensordesc_getdatatype_003
 * @tc.desc: Verify the NN_TensorDesc is nullptr of the OH_NNTensorDesc_GetDataType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkCoreTest, nnt_tensordesc_getdatatype_003, testing::ext::TestSize.Level0)
{
    NN_TensorDesc* tensorDesc = nullptr;
    OH_NN_DataType datatype = OH_NN_INT32;
    OH_NN_ReturnCode ret = OH_NNTensorDesc_GetDataType(tensorDesc, &datatype);
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
} // Unittest
} // namespace NeuralNetworkRuntime
} // namespace OHOS