/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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
#include <fstream>
#include <unistd.h>

#include "nncore_utils.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Test;
namespace OHOS::NeuralNetworkCore {
class CompilationTest : public testing::Test {
public:
    void SetUp()
    {
    }
    void TearDown()
    {
    }
    void GenCacheFile(const std::string& cachePath)
    {
        OH_NNCompilation *compilation = nullptr;
        OH_NNModel *model = nullptr;
        ConstructCompilation(&compilation, &model);
        OHNNCompileParam compileParam{
            .cacheDir = cachePath,
            .cacheVersion = CACHEVERSION,
        };
        ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
        ASSERT_TRUE(CheckPath(cachePath + CACHE_FILE) == PathType::FILE);
        ASSERT_TRUE(CheckPath(cachePath + CACHE_INFO_FILE) == PathType::FILE);
        OH_NNModel_Destroy(&model);
        OH_NNCompilation_Destroy(&compilation);
    }
    void SaveSupportModel()
    {
        OH_NNModel *model = nullptr;
        ConstructAddModel(&model);
        std::ofstream ofs(SUPPORTMODELPATH, std::ios::out | std::ios::binary);
        if (ofs) {
            ofs.write(reinterpret_cast<char*>(model), sizeof(reinterpret_cast<char*>(model)));
            ofs.close();
        }
        OH_NNModel_Destroy(&model);
    }

protected:
    OHNNCompileParam m_compileParam;
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
};

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Construct_Compilation_For_Cache_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_Construct_Compilation_For_Cache_0100
 * @tc.desc   创建compilation，检查返回值为空，设置正确的cache路径，build成功，推理成功
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Construct_Compilation_For_Cache_0100,
         Function | MediumTest | Level1)
{
    pid_t pid = getpid();
    std::string cachePath = CACHE_DIR + "core_build01_" + std::to_string(pid);
    CreateFolder(cachePath);
    GenCacheFile(cachePath);
    OH_NNCompilation *compilation = OH_NNCompilation_ConstructForCache();
    ASSERT_NE(nullptr, compilation);

    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetCache(compilation, cachePath.c_str(), CACHEVERSION));
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(compilation));
    OH_NNCompilation_Destroy(&compilation);
    DeleteFolder(cachePath);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Construct_Compilation_For_Cache_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_Construct_Compilation_For_Cache_0200
 * @tc.desc   创建compilation，检查返回值非空，不设置cache，build失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Construct_Compilation_For_Cache_0200,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = OH_NNCompilation_ConstructForCache();
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(compilation));
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0100
 * @tc.desc   创建compilation，增加config，传入compilation为空，返回错误
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0100,
         Function | MediumTest | Level1)
{
    const char *configName = "test";
    const void *configValue = reinterpret_cast<const void*>(10);
    const size_t configValueSize = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(nullptr, configName, configValue, configValueSize);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0200
 * @tc.desc   创建compilation，增加config，传入configNames为空指针，返回错误
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0200,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);

    const void *configValue = reinterpret_cast<const void*>(10);
    const size_t configValueSize = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(compilation, nullptr, configValue, configValueSize);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ret);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0300
 * @tc.desc   创建compilation，增加config，传入configNames为空字符串，报错
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0300,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);

    const char *configName = "";
    int num = 10;
    const void *configValue = &num;
    const size_t configValueSize = sizeof(num);

    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(compilation, configName, configValue, configValueSize);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ret);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0400
 * @tc.number SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0400
 * @tc.desc   创建compilation，增加config，传入configValues为空，报错
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0400,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);

    const char *configName = "test";
    const size_t configValueSize = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(compilation, configName, nullptr, configValueSize);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ret);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0500
 * @tc.number SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0500
 * @tc.desc   创建compilation，增加config，传入configValueSize为0
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_AddExtension_Config_To_Compilation_0500,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);

    const char *configName = "test";
    const void *configValue = reinterpret_cast<const void*>(10);
    const size_t configValueSize = 0;
    OH_NN_ReturnCode ret = OH_NNCompilation_AddExtensionConfig(compilation, configName, configValue, configValueSize);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ret);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_OfflineModel_File_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_OfflineModel_File_0100
 * @tc.desc   传入filepath为空指针，返回不支持
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_OfflineModel_File_0100,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = OH_NNCompilation_ConstructWithOfflineModelFile(nullptr);
    ASSERT_EQ(nullptr, compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_OfflineModel_File_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_OfflineModel_File_0200
 * @tc.desc   传入合法文件，返回不支持
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_OfflineModel_File_0200,
         Function | MediumTest | Level1)
{
    SaveSupportModel();
    OH_NNCompilation *compilation = OH_NNCompilation_ConstructWithOfflineModelFile(SUPPORTMODELPATH.c_str());
    ASSERT_NE(nullptr, compilation);

    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_FAILED, OH_NNCompilation_Build(compilation));
    DeleteFile(SUPPORTMODELPATH);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_Offline_ModelBuffer_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_Offline_ModelBuffer_0100
 * @tc.desc   传入modelData为空指针，返回错误
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_Offline_ModelBuffer_0100,
         Function | MediumTest | Level1)
{
    int modelSize = 0;
    const void *buffer = nullptr;
    OH_NNCompilation *compilation = OH_NNCompilation_ConstructWithOfflineModelBuffer(buffer, modelSize);
    ASSERT_EQ(nullptr, compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_Offline_ModelBuffer_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_Offline_ModelBuffer_0200
 * @tc.desc   传入modelData为合法离线模型buffer，返回不支持
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Construct_Compilation_With_Offline_ModelBuffer_0200,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation =
        OH_NNCompilation_ConstructWithOfflineModelBuffer(reinterpret_cast<const void*>(TEST_BUFFER), 28);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_FAILED, OH_NNCompilation_Build(compilation));
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Export_Compilation_Cache_To_Buffer_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_Export_Compilation_Cache_To_Buffer_0100
 * @tc.desc   传入空指针返回失败
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Export_Compilation_Cache_To_Buffer_0100,
         Function | MediumTest | Level1)
{
    const char *any = "123456789";
    const void *buffer = reinterpret_cast<const void*>(any);
    size_t length = 10;
    size_t *modelSize = &length;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(nullptr, buffer, length, modelSize);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Export_Compilation_Cache_To_Buffer_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_Export_Compilation_Cache_To_Buffer_0200
 * @tc.desc   参数正确，nnrt模型返回不支持
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Export_Compilation_Cache_To_Buffer_0200,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(compilation));

    const char *any = "123456789";
    const void *buffer = reinterpret_cast<const void*>(any);
    size_t length = 10;
    size_t *modelSize = &length;
    OH_NN_ReturnCode ret = OH_NNCompilation_ExportCacheToBuffer(compilation, buffer, length, modelSize);
    ASSERT_EQ(OH_NN_UNSUPPORTED, ret);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0100
 * @tc.number SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0100
 * @tc.desc   buffer为空，返回错误
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0100,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);

    const void *buffer = nullptr;
    size_t modelSize = MODEL_SIZE;
    OH_NN_ReturnCode ret = OH_NNCompilation_ImportCacheFromBuffer(compilation, buffer, modelSize);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ret);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0200
 * @tc.number SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0200
 * @tc.desc   modelSize为0，返回错误
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0200,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);
    const char *any = "123456789";
    const void *buffer = reinterpret_cast<const void*>(any);
    size_t modelSize = ZERO;
    OH_NN_ReturnCode ret = OH_NNCompilation_ImportCacheFromBuffer(compilation, buffer, modelSize);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, ret);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0300
 * @tc.number SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0300
 * @tc.desc   参数正确，返回不支持
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, SUB_AI_NNRt_Core_Func_North_Import_Compilation_Cache_From_Buffer_0300,
         Function | MediumTest | Level1)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);

    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetCache(compilation, CACHE_DIR.c_str(), CACHEVERSION));
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_EXTREME));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_HIGH));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_EnableFloat16(compilation, false));

    const char *any = "123456789";
    const void *buffer = reinterpret_cast<const void*>(any);
    size_t modelSize = MODEL_SIZE;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_ImportCacheFromBuffer(compilation, buffer, modelSize));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(compilation));
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_001
 * @tc.number nnrt_config_set_focus_mode_001
 * @tc.desc   OH_NN_MEMORY_ERROR断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_001,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_MEMORY_ERROR;
    ASSERT_EQ(OH_NN_MEMORY_ERROR, focusMode);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_002
 * @tc.number nnrt_config_set_focus_mode_002
 * @tc.desc   OH_NN_NULL_PTR断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_002,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_NULL_PTR;
    ASSERT_EQ(OH_NN_NULL_PTR, focusMode);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_003
 * @tc.number nnrt_config_set_focus_mode_003
 * @tc.desc   OH_NN_INVALID_FILE断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_003,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_INVALID_FILE;
    ASSERT_EQ(OH_NN_INVALID_FILE, focusMode);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_004
 * @tc.number nnrt_config_set_focus_mode_004
 * @tc.desc   OH_NN_UNAVALIDABLE_DEVICE断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_004,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_UNAVALIDABLE_DEVICE;
    ASSERT_EQ(OH_NN_UNAVALIDABLE_DEVICE, focusMode);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_005
 * @tc.number nnrt_config_set_focus_mode_005
 * @tc.desc   OH_NN_INVALID_PATH断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_005,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_INVALID_PATH;
    ASSERT_EQ(OH_NN_INVALID_PATH, focusMode);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_006
 * @tc.number nnrt_config_set_focus_mode_006
 * @tc.desc   OH_NN_TIMEOUT断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_006,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_TIMEOUT;
    ASSERT_EQ(OH_NN_TIMEOUT, focusMode);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_007
 * @tc.number nnrt_config_set_focus_mode_007
 * @tc.desc   OH_NN_CONNECTION_EXCEPTION断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_007,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_CONNECTION_EXCEPTION;
    ASSERT_EQ(OH_NN_CONNECTION_EXCEPTION, focusMode);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_008
 * @tc.number nnrt_config_set_focus_mode_008
 * @tc.desc   OH_NN_SAVE_CACHE_EXCEPTION断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_008,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_SAVE_CACHE_EXCEPTION;
    ASSERT_EQ(OH_NN_SAVE_CACHE_EXCEPTION, focusMode);
}

/**
 * @tc.name   nnrt_config_set_focus_mode_009
 * @tc.number nnrt_config_set_focus_mode_009
 * @tc.desc   OH_NN_UNAVAILABLE_DEVICE断言
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(CompilationTest, nnrt_config_set_focus_mode_009,
         Function | MediumTest | Level1)
{
    OH_NN_ReturnCode focusMode = OH_NN_UNAVAILABLE_DEVICE;
    ASSERT_EQ(OH_NN_UNAVAILABLE_DEVICE, focusMode);
}

}