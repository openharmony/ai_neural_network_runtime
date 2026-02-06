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
#include <string>
#include <vector>
#include <gtest/gtest.h>

#include "neural_network_runtime/neural_network_runtime.h"
#include "log.h"
#include "nncore_utils.h"

using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime::Test;

class EndToEndTest : public testing::Test {
public:
    void SetUp()
    {
    }
    void TearDown()
    {
    }
};

void BuildModel(OH_NNModel **model)
{
    *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(*model, graphArgs));
}

void BuildDynamicModel(OH_NNModel **model)
{
    *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    AvgPoolDynamicModel avgModel;
    OHNNGraphArgs graphArgs = avgModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(*model, graphArgs));
}

void BuildModelWithQuantParams(OH_NNModel **model)
{
    *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraphWithQuantParams(*model, graphArgs));
}

OH_NNCompilation* ConstructCompilation(OH_NNModel* model, size_t deviceId, bool isUseCache = true)
{
    OH_NNCompilation* compilation = nullptr;
    if (model == nullptr) {
        compilation = OH_NNCompilation_ConstructForCache();
        if (compilation == nullptr) {
            LOGE("End2EndTest::OH_NNCompilation_ConstructForCache failed.");
            return nullptr;
        }
    } else {
        compilation = OH_NNCompilation_Construct(model);
        if (compilation == nullptr) {
            LOGE("End2EndTest::OH_NNCompilation_Construct failed.");
            return nullptr;
        }
    }

    OH_NN_ReturnCode returnCode = OH_NNCompilation_SetDevice(compilation, deviceId);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::OH_NNCompilation_SetDevice failed.");
        return nullptr;
    }

    if (isUseCache) {
        returnCode = OH_NNCompilation_SetCache(compilation, CACHE_DIR.c_str(), 1);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNCompilation_SetCache failed.");
            return nullptr;
        }
    }

    returnCode = OH_NNCompilation_SetPerformanceMode(compilation, OH_NN_PERFORMANCE_EXTREME);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::OH_NNCompilation_SetPerformanceMode failed.");
        return nullptr;
    }

    returnCode = OH_NNCompilation_SetPriority(compilation, OH_NN_PRIORITY_HIGH);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::OH_NNCompilation_SetPriority failed.");
        return nullptr;
    }

    returnCode = OH_NNCompilation_EnableFloat16(compilation, false);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::OH_NNCompilation_EnableFloat16 failed.");
        return nullptr;
    }

    // 执行编译
    returnCode = OH_NNCompilation_Build(compilation);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::OH_NNCompilation_Build failed.");
        return nullptr;
    }
    return compilation;
}

OH_NN_ReturnCode SetInputData(NN_Tensor* inputTensor[], size_t inputSize)
{
    OH_NN_DataType dataType(OH_NN_FLOAT32);
    OH_NN_ReturnCode ret{OH_NN_FAILED};
    size_t elementNum = 0;
    for (size_t i = 0; i < inputSize; ++i) {
        auto data = OH_NNTensor_GetDataBuffer(inputTensor[i]);
        if (data == nullptr) {
            LOGE("End2EndTest::OH_NNTensor_GetDataBuffer failed.");
            return OH_NN_FAILED;
        }
        auto desc = OH_NNTensor_GetTensorDesc(inputTensor[i]);
        if (desc == nullptr) {
            LOGE("End2EndTest::OH_NNTensor_GetTensorDesc failed.");
            return OH_NN_FAILED;
        }
        ret = OH_NNTensorDesc_GetDataType(desc, &dataType);
        if (ret != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNTensorDesc_GetDataType failed.");
            return ret;
        }
        ret = OH_NNTensorDesc_GetElementCount(desc, &elementNum);
        if (ret != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNTensorDesc_GetElementCount failed.");
            return ret;
        }
        switch (dataType) {
            case OH_NN_FLOAT32: {
                float* floatValue = reinterpret_cast<float*>(data);
                for (size_t j = 0; j < elementNum; ++j) {
                    floatValue[j] = static_cast<float>(j);
                }
                break;
            }
            case OH_NN_INT32: {
                int* intValue = reinterpret_cast<int*>(data);
                for (size_t j = 0; j < elementNum; ++j) {
                    intValue[j] = static_cast<int>(j);
                }
                break;
            }
            default:
                return OH_NN_FAILED;
        }
    }
    return OH_NN_SUCCESS;
}
OH_NN_ReturnCode GetInputAndOutputTensorDesc(OH_NNExecutor *executor,
                                             size_t *inputCount, std::vector<NN_TensorDesc*>& inputTensorDescs,
                                             size_t *outputCount, std::vector<NN_TensorDesc*>& outputTensorDescs)
{
    OH_NN_ReturnCode returnCode = OH_NNExecutor_GetInputCount(executor, inputCount);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::OH_NNExecutor_GetInputCount failed.");
        return returnCode;
    }
    NN_TensorDesc* tensorDescTmp = nullptr;
    for (size_t i = 0; i < *inputCount; ++i) {
        tensorDescTmp = OH_NNExecutor_CreateInputTensorDesc(executor, i);
        if (tensorDescTmp == nullptr) {
            LOGE("End2EndTest::OH_NNExecutor_CreateInputTensorDesc failed.");
            return OH_NN_FAILED;
        }
        inputTensorDescs.emplace_back(tensorDescTmp);
    }
    returnCode = OH_NNExecutor_GetOutputCount(executor, outputCount);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::OH_NNExecutor_GetOutputCount failed.");
        return returnCode;
    }
    for (size_t i = 0; i < *outputCount; ++i) {
        tensorDescTmp = OH_NNExecutor_CreateOutputTensorDesc(executor, i);
        if (tensorDescTmp == nullptr) {
            LOGE("End2EndTest::OH_NNExecutor_CreateOutputTensorDesc failed.");
            return OH_NN_FAILED;
        }
        outputTensorDescs.emplace_back(tensorDescTmp);
    }

    return returnCode;
}

OH_NN_ReturnCode GetInputDimAndSetShape(OH_NNExecutor *executor, std::vector<NN_TensorDesc*>& inputTensorDescs,
                                        std::vector<NN_TensorDesc*>& outputTensorDescs, bool isDynamic)
{
    if (isDynamic) {
        size_t *minInputDims = nullptr;
        size_t *maxInputDims = nullptr;
        size_t shapeLength = ZERO;
        for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
            if (OH_NN_SUCCESS != OH_NNExecutor_GetInputDimRange(executor, i, &minInputDims,
                                                                &maxInputDims, &shapeLength)) {
                LOGE("End2EndTest::OH_NNExecutor_GetInputDimRange failed.");
                return OH_NN_FAILED;
            }
            std::vector<int32_t> minInputDimsT;
            for (size_t j = 0; j < shapeLength; ++j) {
                minInputDimsT.emplace_back(static_cast<int32_t>(minInputDims[j]));
            }
            if (OH_NN_SUCCESS != OH_NNTensorDesc_SetShape(inputTensorDescs[i], minInputDimsT.data(), shapeLength)) {
                LOGE("End2EndTest::OH_NNTensorDesc_SetShape failed.");
                return OH_NN_FAILED;
            }
        }
        std::vector<int32_t> outputShape{1, 2, 2, 1};
        for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
            if (OH_NN_SUCCESS != OH_NNTensorDesc_SetShape(outputTensorDescs[i],
                                                          outputShape.data(), outputShape.size())) {
                LOGE("End2EndTest::OH_NNTensorDesc_SetShape failed.");
                return OH_NN_FAILED;
            }
        }
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode CreateTensorAndDestroyTensorDesc(NN_Tensor* tensors[], size_t count,
                                                  std::vector<NN_TensorDesc*>& tensorDescs, size_t deviceId)
{
    NN_Tensor* tensor = nullptr;
    for (size_t i = 0; i < count; ++i) {
        tensor = nullptr;
        tensor = OH_NNTensor_Create(deviceId, tensorDescs[i]);
        if (tensor == nullptr) {
            LOGE("End2EndTest::OH_NNTensor_Create failed.");
            return OH_NN_FAILED;
        }
        tensors[i] = tensor;
    }
    for (size_t i = 0; i < count; ++i) {
        if (OH_NN_SUCCESS != OH_NNTensorDesc_Destroy(&tensorDescs[i])) {
            LOGE("End2EndTest::OH_NNTensorDesc_Destroy failed.");
            return OH_NN_FAILED;
        }
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DestroyInputAndOutputTensor(NN_Tensor** inputTensors, size_t inputCount,
                                             NN_Tensor** outputTensors, size_t outputCount)
{
    OH_NN_ReturnCode returnCode = OH_NN_SUCCESS;
    for (size_t i = 0; i < inputCount; ++i) {
        returnCode = OH_NNTensor_Destroy(&inputTensors[i]);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNTensor_Destroy failed.");
            return OH_NN_FAILED;
        }
    }
    for (size_t i = 0; i < outputCount; ++i) {
        returnCode = OH_NNTensor_Destroy(&outputTensors[i]);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNTensor_Destroy failed.");
            return OH_NN_FAILED;
        }
    }
    return OH_NN_SUCCESS;
}

OH_NNExecutor* RunExecutor(OH_NNCompilation* compilation, size_t deviceId, bool isDynamic = false)
{
    OH_NNExecutor *executor = OH_NNExecutor_Construct(compilation);
    if (executor == nullptr) {
        LOGE("End2EndTest::OH_NNExecutor_Construct failed.");
        return nullptr;
    }
    size_t inputCount = 0;
    std::vector<NN_TensorDesc*> inputTensorDescs;
    size_t outputCount = 0;
    std::vector<NN_TensorDesc*> outputTensorDescs;
    OH_NN_ReturnCode returnCode = GetInputAndOutputTensorDesc(executor, &inputCount, inputTensorDescs,
                                                              &outputCount, outputTensorDescs);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::GetInputAndOutputTensorDesc failed.");
        return nullptr;
    }
    returnCode = GetInputDimAndSetShape(executor, inputTensorDescs, outputTensorDescs, isDynamic);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::GetInputDimAndSetShape failed.");
        return nullptr;
    }
    NN_Tensor* inputTensors[inputCount];
    OH_NN_ReturnCode returnCodeTmp = CreateTensorAndDestroyTensorDesc(inputTensors, inputCount,
                                                                      inputTensorDescs, deviceId);
    NN_Tensor* outputTensors[outputCount];
    returnCode = CreateTensorAndDestroyTensorDesc(outputTensors, outputCount, outputTensorDescs, deviceId);
    if (returnCode != OH_NN_SUCCESS || returnCodeTmp != OH_NN_SUCCESS) {
        LOGE("End2EndTest::Tensors create failed.");
        return nullptr;
    }
    returnCode = SetInputData(inputTensors, inputCount);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::SetInputData failed.");
        return nullptr;
    }
    returnCode = OH_NNExecutor_RunSync(executor, inputTensors, inputCount, outputTensors, outputCount);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::OH_NNExecutor_RunSync failed.");
        return nullptr;
    }
    returnCode = DestroyInputAndOutputTensor(inputTensors, inputCount, outputTensors, outputCount);
    if (returnCode != OH_NN_SUCCESS) {
        LOGE("End2EndTest::DestroyInputAndOutputTensor failed.");
        return nullptr;
    }
    return executor;
}

/**
 * @tc.name   sub_AI_NNRt_Core_Func_North_EndToEnd_0100
 * @tc.number sub_AI_NNRt_Core_Func_North_EndToEnd_0100
 * @tc.desc   定长模型编译端到端测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(EndToEndTest, sub_AI_NNRt_Core_Func_North_EndToEnd_0100, Function | MediumTest | Level1)
{
    size_t deviceId = 0;
    ASSERT_EQ(OH_NN_SUCCESS, GetDeviceID(&deviceId));

    OH_NNModel* model = nullptr;
    BuildModel(&model);

    OH_NNCompilation* compilation = ConstructCompilation(model, deviceId);
    ASSERT_NE(nullptr, compilation);

    OH_NNModel_Destroy(&model);
    OH_NNExecutor* executor = RunExecutor(compilation, deviceId);
    ASSERT_NE(nullptr, executor);
    OH_NNCompilation_Destroy(&compilation);
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name   sub_AI_NNRt_Core_Func_North_EndToEnd_0200
 * @tc.number sub_AI_NNRt_Core_Func_North_EndToEnd_0200
 * @tc.desc   变长模型编译端到端测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(EndToEndTest, sub_AI_NNRt_Core_Func_North_EndToEnd_0200, Function | MediumTest | Level1)
{
    size_t deviceId = 0;
    ASSERT_EQ(OH_NN_SUCCESS, GetDeviceID(&deviceId));
    OH_NNModel* model = nullptr;
    BuildDynamicModel(&model);

    OH_NNCompilation* compilation = ConstructCompilation(model, deviceId);
    ASSERT_NE(nullptr, compilation);
    OH_NNModel_Destroy(&model);
    OH_NNExecutor* executor = RunExecutor(compilation, deviceId, true);
    ASSERT_NE(nullptr, executor);
    OH_NNCompilation_Destroy(&compilation);
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name   sub_AI_NNRt_Core_Func_North_EndToEnd_0300
 * @tc.number sub_AI_NNRt_Core_Func_North_EndToEnd_0300
 * @tc.desc   定长模型编译带量化参数端到端测试
 * @tc.type   FUNCTION
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL1
 */
HWTEST_F(EndToEndTest, sub_AI_NNRt_Core_Func_North_EndToEnd_0300, Function | MediumTest | Level1)
{
    size_t deviceId = 0;
    ASSERT_EQ(OH_NN_SUCCESS, GetDeviceID(&deviceId));

    OH_NNModel* model = nullptr;
    BuildModelWithQuantParams(&model);

    OH_NNCompilation* compilation = ConstructCompilation(model, deviceId);
    ASSERT_NE(nullptr, compilation);
    OH_NNModel_Destroy(&model);
    OH_NNExecutor* executor = RunExecutor(compilation, deviceId);
    ASSERT_NE(nullptr, executor);
    OH_NNCompilation_Destroy(&compilation);
    OH_NNExecutor_Destroy(&executor);
}

/**
 * @tc.name   sub_AI_NNRt_Core_Func_North_Reliability_0100
 * @tc.number sub_AI_NNRt_Core_Func_North_Reliability_0100
 * @tc.desc   定长模型编译长稳测试
 * @tc.type   RELIABILITY
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(EndToEndTest, sub_AI_NNRt_Core_Func_North_Reliability_0100, Reliability | MediumTest | Level2)
{
    size_t deviceId = 0;
    ASSERT_EQ(OH_NN_SUCCESS, GetDeviceID(&deviceId));

    OH_NNModel* model = nullptr;
    BuildModel(&model);
    for (int i = 0; i < STRESS_COUNT; i++) {
        OH_NNCompilation* compilation = ConstructCompilation(model, deviceId);
        ASSERT_NE(nullptr, compilation);
        OH_NNExecutor* executor = RunExecutor(compilation, deviceId);
        ASSERT_NE(nullptr, executor);
        OH_NNCompilation_Destroy(&compilation);
        OH_NNExecutor_Destroy(&executor);
        if (i % PRINT_FREQ == 0) {
            printf("[NnrtTest] Reliability_test_001 times: %d/%d\n", i, STRESS_COUNT);
        }
    }
    OH_NNModel_Destroy(&model);
}

/**
 * @tc.name   sub_AI_NNRt_Core_Func_North_Reliability_0200
 * @tc.number sub_AI_NNRt_Core_Func_North_Reliability_0200
 * @tc.desc   变长模型编译端到端测试
 * @tc.type   RELIABILITY
 * @tc.size   MEDIUMTEST
 * @tc.level  LEVEL2
 */
HWTEST_F(EndToEndTest, sub_AI_NNRt_Core_Func_North_Reliability_0200, Reliability | MediumTest | Level2)
{
    size_t deviceId = 0;
    ASSERT_EQ(OH_NN_SUCCESS, GetDeviceID(&deviceId));

    OH_NNModel* model = nullptr;
    BuildDynamicModel(&model);
    for (int i = 0; i < STRESS_COUNT; i++) {
        OH_NNCompilation* compilation = ConstructCompilation(model, deviceId, false);
        ASSERT_NE(nullptr, compilation);
        OH_NNExecutor* executor = RunExecutor(compilation, deviceId, true);
        ASSERT_NE(nullptr, executor);
        OH_NNCompilation_Destroy(&compilation);
        OH_NNExecutor_Destroy(&executor);
        if (i % PRINT_FREQ == 0) {
            printf("[NnrtTest] Reliability_test_002 times: %d/%d\n", i, STRESS_COUNT);
        }
    }
    OH_NNModel_Destroy(&model);
}
