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
#include <cmath>
#include <cstdio>
#include <vector>
#include <thread>
#include <fstream>

#include "nnrt_utils.h"
#include "model.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
using namespace OHOS::NeuralNetworkRuntime::Test;
using namespace OHOS::HDI::Nnrt::V1_0;

namespace {

class CompileTest : public testing::Test {
public:
    void SetUp()
    {
        CreateFolder(CACHE_DIR);
    }
    void TearDown()
    {
        DeleteFolder(CACHE_DIR);
    }
    void GenCacheFile()
    {
        OH_NNModel *model = OH_NNModel_Construct();
        ASSERT_NE(nullptr, model);
        ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
        OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
        ASSERT_NE(nullptr, compilation);
        OHNNCompileParam compileParam{
            .cacheDir = "./cache",
            .cacheVersion = 10,
        };
        ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
        Free(model, compilation);
        ASSERT_TRUE(CheckPath(CACHE_PATH) == PathType::FILE);
        ASSERT_TRUE(CheckPath(CACHE_INFO_PATH) == PathType::FILE);
    }
    void DestroyCache()
    {
        std::ifstream ifs(CACHE_PATH.c_str(), std::ios::in | std::ios::binary);
        char* ptr{nullptr};
        int cacheSize = ifs.tellg();
        int invalidCacheSize = cacheSize * 0.9;
        ifs.read(ptr, cacheSize);
        ifs.close();
        std::ofstream ofs(CACHE_PATH.c_str(), std::ios::out | std::ios::binary);
        ofs.write(ptr, invalidCacheSize);
        ofs.close();
    }

protected:
    OHNNCompileParam m_compileParam;
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
};

void CompileModel(OH_NNCompilation *compilation, const OHNNCompileParam &compileParam)
{
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
}

} // namespace

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Create_0100
 * @tc.name   : 创建编译实例，model为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Create_0100, Function | MediumTest | Level3)
{
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(nullptr);
    ASSERT_EQ(nullptr, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Create_0200
 * @tc.name   : 创建编译实例，model未完成构图
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Create_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Create_0300
 * @tc.name   : 创建编译实例，model已完成构图，存在算子不支持
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Create_0300, Function | MediumTest | Level2)
{
    OHOS::sptr<V1_0::MockIDevice> device = V1_0::MockIDevice::GetInstance();
    std::vector<bool> isSupported = {true, false};
    device->SetOperationsSupported(isSupported);

    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);

    AddTopKModel addTopKModel;
    OHNNGraphArgsMulti graphArgsMulti = addTopKModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildMultiOpGraph(model, graphArgsMulti));

    const size_t *devicesID{nullptr};
    const bool *realSupported{nullptr};
    uint32_t opCount;
    uint32_t devicesCount;
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNDevice_GetAllDevicesID(&devicesID, &devicesCount));
    size_t targetDevice = devicesID[0];

    OH_NN_ReturnCode ret = OH_NNModel_GetAvailableOperations(model, targetDevice, &realSupported, &opCount);
    ASSERT_EQ(OH_NN_SUCCESS, ret);
    for (int i = 0; i < opCount; i++) {
        EXPECT_EQ(realSupported[i], isSupported[i]);
    }
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_FAILED, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
    device->SetOperationsSupported({true});
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetDevice_0100
 * @tc.name   : 设置device，compilation为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetDevice_0100, Function | MediumTest | Level3)
{
    const size_t *devicesID{nullptr};
    uint32_t devicesCount{0};
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNDevice_GetAllDevicesID(&devicesID, &devicesCount));

    size_t targetDevice = devicesID[0]; // Use the first device in system test.
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_SetDevice(nullptr, targetDevice));
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetDevice_0200
 * @tc.name   : 设置device，deviceID不存在
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetDevice_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(compilation, 100000));
    ASSERT_EQ(OH_NN_FAILED, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetDevice_0300
 * @tc.name   : 设置device，deviceID存在
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetDevice_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam;
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetCache_0100
 * @tc.name   : 设置cache路径及版本，compilation为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetCache_0100, Function | MediumTest | Level3)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_SetCache(nullptr, "./", 0));
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetCache_0200
 * @tc.name   : 设置cache路径及版本，cacheDir为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetCache_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_SetCache(compilation, nullptr, 0));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetCache_0300
 * @tc.name   : device不支持，设置cache路径及版本
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetCache_0300, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    // set model cache unavailabel
    OHOS::sptr<V1_0::MockIDevice> device = V1_0::MockIDevice::GetInstance();
    device->SetModelCacheSupported(false);

    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetCache(compilation, "./cache", 10));
    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
    device->SetModelCacheSupported(true);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetCache_0400
 * @tc.name   : 设置不存在cache路径
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetCache_0400, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam{.cacheDir = "./test"};
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetCache_0500
 * @tc.name   : 设置cache路径，cache破坏，重新生成cache
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetCache_0500, Function | MediumTest | Level2)
{
    // generate cache file in cache diretory
    GenCacheFile();
    // destroy cache file to invalid size
    DestroyCache();
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam{
        .cacheDir = "./cache",
        .cacheVersion = 10,
    };
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetCache_0600
 * @tc.name   : 设置version，小于cache版本号
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetCache_0600, Function | MediumTest | Level2)
{
    GenCacheFile();
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam{
        .cacheDir = "./cache",
        .cacheVersion = 9,
    };
    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetCache_0700
 * @tc.name   : 设置version，等于cache版本号
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetCache_0700, Function | MediumTest | Level2)
{
    GenCacheFile();
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam{
        .cacheDir = "./cache",
        .cacheVersion = 10,
    };
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetCache_0800
 * @tc.name   : 设置version，大于cache版本号
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetCache_0800, Function | MediumTest | Level2)
{
    GenCacheFile();
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam{
        .cacheDir = "./cache",
        .cacheVersion = 11,
    };
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0100
 * @tc.name   : 设置priority，compilation为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0100, Function | MediumTest | Level3)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_SetPerformanceMode(nullptr, OH_NN_PERFORMANCE_MEDIUM));
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0200
 * @tc.name   : device不支持，设置performance
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_Mock_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam{
        .performanceMode = OH_NN_PERFORMANCE_LOW,
    };
    OHOS::sptr<V1_0::MockIDevice> device = V1_0::MockIDevice::GetInstance();
    device->SetPerformanceSupported(false);
    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
    device->SetPerformanceSupported(true);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0300
 * @tc.name   : 设置performanceMode为NONE
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0300, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_NONE));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0400
 * @tc.name   : 设置performanceMode为LOW
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0400, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_LOW));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0500
 * @tc.name   : 设置performanceMode为MEDIUM
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0500, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_MEDIUM));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0600
 * @tc.name   : 设置performanceMode为HIGH
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0600, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_HIGH));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0700
 * @tc.name   : 设置performanceMode为EXTREME
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0700, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_EXTREME));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0800
 * @tc.name   : 设置performanceMode为NONE-1
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0800, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNCompilation_SetPerformanceMode(compilation,
                                                  static_cast<OH_NN_PerformanceMode>(OH_NN_PERFORMANCE_NONE - 1)));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0900
 * @tc.name   : 设置performanceMode为EXTREME+1
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPerformanceMode_0900, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNCompilation_SetPerformanceMode(compilation,
                                                  static_cast<OH_NN_PerformanceMode>(OH_NN_PERFORMANCE_EXTREME + 1)));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPriority_0100
 * @tc.name   : 设置priority，compilation为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPriority_0100, Function | MediumTest | Level3)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_SetPriority(nullptr, OH_NN_PRIORITY_MEDIUM));
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPriority_0200
 * @tc.name   : device不支持，设置priority
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPriority_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    // set device not supported
    OHOS::sptr<V1_0::MockIDevice> device = V1_0::MockIDevice::GetInstance();
    device->SetPrioritySupported(false);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_LOW));
    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
    device->SetPrioritySupported(true);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPriority_0300
 * @tc.name   : 设置priority为NONE
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPriority_0300, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_NONE));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPriority_0400
 * @tc.name   : 设置priority为LOW
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPriority_0400, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_LOW));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPriority_0500
 * @tc.name   : 设置priority为MEDIUM
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPriority_0500, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_MEDIUM));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPriority_0600
 * @tc.name   : 设置priority为LOW
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPriority_0600, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_HIGH));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPriority_0700
 * @tc.name   : 设置priority为NONE-1
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPriority_0700, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNCompilation_SetPriority(compilation, static_cast<OH_NN_Priority>(OH_NN_PRIORITY_NONE - 1)));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_SetPriority_0800
 * @tc.name   : 设置priority为HIGH+1
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_SetPriority_0800, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS,
              OH_NNCompilation_SetPriority(compilation, static_cast<OH_NN_Priority>(OH_NN_PRIORITY_HIGH + 1)));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_EnableFloat16_0100
 * @tc.name   : 设置enableFloat16，compilation为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_EnableFloat16_0100, Function | MediumTest | Level3)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_EnableFloat16(nullptr, OH_NN_PERFORMANCE_MEDIUM));
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_EnableFloat16_0200
 * @tc.name   : device支持，设置fp16推理为false
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_EnableFloat16_0200, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_EnableFloat16(compilation, false));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_EnableFloat16_0300
 * @tc.name   : device不支持，设置fp16推理为false
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_EnableFloat16_0300, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    // set fp16 unavailable
    OHOS::sptr<V1_0::MockIDevice> device = V1_0::MockIDevice::GetInstance();
    device->SetFP16Supported(false);

    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_EnableFloat16(compilation, true));
    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
    device->SetFP16Supported(true);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_EnableFloat16_0400
 * @tc.name   : device不支持，设置fp16推理为true
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_EnableFloat16_0400, Function | MediumTest | Level3)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    // set fp16 unavailable
    OHOS::sptr<V1_0::MockIDevice> device = V1_0::MockIDevice::GetInstance();
    device->SetFP16Supported(false);

    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_EnableFloat16(compilation, true));
    ASSERT_EQ(OH_NN_OPERATION_FORBIDDEN, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
    device->SetFP16Supported(true);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Build_0100
 * @tc.name   : 编译模型，compilation为空指针
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Build_0100, Function | MediumTest | Level3)
{
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(nullptr));
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Build_0200
 * @tc.name   : 编译模型，未设置device，默认设备，返回成功
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Build_0200, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Build_0300
 * @tc.name   : 编译模型，仅设置device，默认配置测试
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Build_0300, Function | MediumTest | Level1)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam;
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Build_0400
 * @tc.name   : 设置缓存路径及版本，编译模型导出缓存
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Build_0400, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam{
        .cacheDir = "./cache",
        .cacheVersion = 10,
    };
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Destroy_0100
 * @tc.name   : 释放编译实例，compilation为nullptr
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Destroy_0100, Function | MediumTest | Level3)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNCompilation_Destroy(&compilation);
    ASSERT_EQ(nullptr, compilation);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Destroy_0200
 * @tc.name   : 释放编译实例，未调用模型编译
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Destroy_0200, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    ASSERT_EQ(OH_NN_SUCCESS, SetDevice(compilation));
    OH_NNCompilation_Destroy(&compilation);
    ASSERT_EQ(nullptr, compilation);
    Free(model);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Destroy_0300
 * @tc.name   : 模型已编译，释放编译实例
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Destroy_0300, Function | MediumTest | Level0)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam;
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    Free(model, compilation);
}

/**
 * @tc.number : SUB_AI_NNR_Func_North_Compilation_Combine_0100
 * @tc.name   : 多线程并发模型编译，编译成功
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNR_Func_North_Compilation_Combine_0100, Function | MediumTest | Level2)
{
    OH_NNModel *model1 = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model1);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model1, graphArgs));

    OH_NNModel *model2 = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model2);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model2, graphArgs));

    OH_NNCompilation *compilation1 = OH_NNCompilation_Construct(model1);
    ASSERT_NE(nullptr, compilation1);
    OH_NNCompilation *compilation2 = OH_NNCompilation_Construct(model2);
    ASSERT_NE(nullptr, compilation2);

    std::thread th1(CompileModel, compilation1, m_compileParam);
    std::thread th2(CompileModel, compilation2, m_compileParam);
    th1.join();
    th2.join();
    Free(model1, compilation1);
    Free(model2, compilation2);
}

/**
 * @tc.number : SUB_AI_NNRt_Func_North_Compilation_Combine_0200
 * @tc.name   : 已编译模型，重复编译
 * @tc.desc   : [C- SOFTWARE -0200]
 */
HWTEST_F(CompileTest, SUB_AI_NNRt_Func_North_Compilation_Combine_0200, Function | MediumTest | Level2)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));
    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);
    OHNNCompileParam compileParam;
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    ASSERT_EQ(OH_NN_INVALID_PARAMETER, OH_NNCompilation_Build(compilation));
    Free(model, compilation);
}