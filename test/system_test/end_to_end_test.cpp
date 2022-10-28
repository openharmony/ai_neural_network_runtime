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

#include "end_to_end_test.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "securec.h"

#include "common/log.h"
#include "interfaces/kits/c/neural_network_runtime.h"

namespace fs = std::filesystem;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace SystemTest {
const float INPUT_ONE = 1.23;
const float INPUT_TWO = 2.34;
const float EXPECTED_OUTPUT = 5.91;
const int8_t EXPECTED_QUANT_OUTPUT = 10;
const float EPSILON = 1e-4;
const uint32_t NO_DEVICE_COUNT = 0;
const int32_t ELEMENT_COUNT = 12;
const uint32_t ADDEND_DATA_LENGTH = ELEMENT_COUNT * sizeof(float);
const std::string CACHE_DIR = "/data/local/tmp/nnrt_st_cache";
const uint32_t CACHE_VERSION = 1;
const int REPEAT_TIMES = 100;

// End2EndTest build a model with two connected add operations.
OH_NN_ReturnCode End2EndTest::BuildModel(const std::vector<CppTensor>& tensors)
{
    m_model = OH_NNModel_Construct();
    if (m_model == nullptr) {
        LOGE("End2EndTest::BuildModel failed, error happens when creating OH_NNModel.");
        return OH_NN_MEMORY_ERROR;
    }

    OH_NN_ReturnCode status = AddTensors(tensors);
    if (status != OH_NN_SUCCESS) {
        LOGE("End2EndTest::BuildModel failed, error happens when adding tensors.");
        return status;
    }

    status = AddOperation(OH_NN_OPS_ADD, {2}, {0, 1}, {3});
    if (status != OH_NN_SUCCESS) {
        LOGE("End2EndTest::BuildModel failed, error happends when adding first Add operation into the model.");
        return status;
    }

    status = AddOperation(OH_NN_OPS_ADD, {2}, {3, 1}, {4});
    if (status != OH_NN_SUCCESS) {
        LOGE("End2EndTest::BuildModel failed, error happends when adding second Add operation into the model.");
        return status;
    }

    status = SpecifyInputAndOutput({0, 1}, {4});
    if (status != OH_NN_SUCCESS) {
        LOGE("End2EndTest::BuildModel failed, error happends when specifying the inputs and outputs.");
        return status;
    }

    status = OH_NNModel_Finish(m_model);
    if (status != OH_NN_SUCCESS) {
        LOGE("End2EndTest::BuildModel failed, error happends during constructing the model.");
        return status;
    }

    return status;
}

OH_NN_ReturnCode End2EndTest::IsExpectedOutput(const float* outputBuffer)
{
    if (outputBuffer == nullptr) {
        LOGE("End2EndTest::IsExpectedOutput failed, pass nullptr to outputBuffer.");
        return OH_NN_INVALID_PARAMETER;
    }

    for (int i = 0; i < ELEMENT_COUNT; i++) {
        LOGI("Comparing inference output with expected value, output index: %d, output value: %f, "
             "expected value: %f.", i, outputBuffer[i], EXPECTED_OUTPUT);
        if (std::abs(outputBuffer[i] - EXPECTED_OUTPUT) > EPSILON) {
            return OH_NN_FAILED;
        }
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode End2EndTest::IsExpectedOutput(const OH_NN_Memory* outputMemory)
{
    if (outputMemory == nullptr) {
        LOGE("End2EndTest::IsExpectedOutput failed, pass nullptr to outputMemory.");
        return OH_NN_INVALID_PARAMETER;
    }

    if (outputMemory->length == 0) {
        LOGE("End2EndTest::IsExpectedOutput failed, outputMemory is empty.");
        return OH_NN_FAILED;
    }

    float* output = static_cast<float*>(const_cast<void*>(outputMemory->data));
    return IsExpectedOutput(output);
}

/*
 * @tc.name: end_to_end_test_001
 * @tc.desc: Test End-to-End operation of Neural Network Runtime.
 * @tc.type: FUNC
 */
HWTEST_F(End2EndTest, end_to_end_test_001, testing::ext::TestSize.Level1)
{
    // Prepare tensors
    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    CppTensor addend1{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor immediateTensor{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor output{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, immediateTensor, output};

    ASSERT_EQ(OH_NN_SUCCESS, BuildModel(tensors));

    m_compilation = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, m_compilation);
    OH_NNModel_Destroy(&m_model);
    ASSERT_EQ(nullptr, m_model);

    ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
    ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
    size_t targetDevice = m_devices[0]; // Use the first device in system test.
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

    m_executor = OH_NNExecutor_Construct(m_compilation);
    ASSERT_NE(nullptr, m_executor);
    OH_NNCompilation_Destroy(&m_compilation);
    ASSERT_EQ(nullptr, m_compilation);

    // Set value of firstAddend
    std::vector<float> firstAddendValue(ELEMENT_COUNT, INPUT_ONE);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(0, {3, 2, 2}, (void*)firstAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set value of secondAddend
    std::vector<float> secondAddendValue(ELEMENT_COUNT, INPUT_TWO);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(1, {3, 2, 2},  (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set output buffer of output
    float outputBuffer[ELEMENT_COUNT];
    ASSERT_EQ(OH_NN_SUCCESS, SetOutput(0, (void*)outputBuffer, ADDEND_DATA_LENGTH));

    // Run inference and assert output value
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
    ASSERT_EQ(OH_NN_SUCCESS, IsExpectedOutput(outputBuffer));

    OH_NNExecutor_Destroy(&m_executor);
    ASSERT_EQ(nullptr, m_executor);
}

/*
 * @tc.name: end_to_end_test_002
 * @tc.desc: Test End-to-End operation of Neural Network Runtime using OH_NN_Memory
 * @tc.type: FUNC
 */
HWTEST_F(End2EndTest, end_to_end_test_002, testing::ext::TestSize.Level1)
{
    // Prepare tensors
    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    CppTensor addend1{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor immediateTensor{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor output{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, immediateTensor, output};

    ASSERT_EQ(OH_NN_SUCCESS, BuildModel(tensors));

    m_compilation = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, m_compilation);
    OH_NNModel_Destroy(&m_model);
    ASSERT_EQ(nullptr, m_model);

    ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
    ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
    size_t targetDevice = m_devices[0]; // Use the first device in system test.
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

    m_executor = OH_NNExecutor_Construct(m_compilation);
    ASSERT_NE(nullptr, m_executor);
    OH_NNCompilation_Destroy(&m_compilation);
    ASSERT_EQ(nullptr, m_compilation);

    // Set value of firstAddend
    std::vector<float> firstAddendValue(ELEMENT_COUNT, INPUT_ONE);
    OH_NN_Memory* firstAddendMemory;
    ASSERT_EQ(OH_NN_SUCCESS,
        SetInputFromMemory(0, {3, 2, 2}, (void*)firstAddendValue.data(), ADDEND_DATA_LENGTH, &firstAddendMemory));

    // Set value of secondAddend
    std::vector<float> secondAddendValue(ELEMENT_COUNT, INPUT_TWO);
    OH_NN_Memory* secondAddendMemory;
    ASSERT_EQ(OH_NN_SUCCESS,
        SetInputFromMemory(1, {3, 2, 2}, (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH, &secondAddendMemory));

    // Set output buffer of output
    OH_NN_Memory* outputMemory;
    ASSERT_EQ(OH_NN_SUCCESS, SetOutputFromMemory(0, ADDEND_DATA_LENGTH, &outputMemory));

    // Run inference and assert output value
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
    ASSERT_EQ(OH_NN_SUCCESS, IsExpectedOutput(outputMemory));

    OH_NNExecutor_DestroyInputMemory(m_executor, 0, &firstAddendMemory);
    ASSERT_EQ(nullptr, firstAddendMemory);
    OH_NNExecutor_DestroyInputMemory(m_executor, 1, &secondAddendMemory);
    ASSERT_EQ(nullptr, secondAddendMemory);
    OH_NNExecutor_DestroyOutputMemory(m_executor, 0, &outputMemory);
    ASSERT_EQ(nullptr, outputMemory);

    OH_NNExecutor_Destroy(&m_executor);
    ASSERT_EQ(nullptr, m_executor);
}

/*
 * @tc.name: end_to_end_test_003
 * @tc.desc: Test End-to-End operation of Neural Network Runtime with dynamic inputs.
 * @tc.type: FUNC
 */
HWTEST_F(End2EndTest, end_to_end_test_003, testing::ext::TestSize.Level1)
{
    // Prepare tensors
    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    std::vector<float> value(ELEMENT_COUNT, INPUT_ONE);
    CppTensor addend1{OH_NN_FLOAT32, {3, 2, 2}, (void*)value.data(), ADDEND_DATA_LENGTH, quantParam, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor immediateTensor{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor output{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, immediateTensor, output};

    m_model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, m_model);
    ASSERT_EQ(OH_NN_SUCCESS, AddTensors(tensors));
    ASSERT_EQ(OH_NN_SUCCESS, AddOperation(OH_NN_OPS_ADD, {2}, {0, 1}, {3}));
    ASSERT_EQ(OH_NN_SUCCESS, AddOperation(OH_NN_OPS_ADD, {2}, {3, 1}, {4}));
    ASSERT_EQ(OH_NN_SUCCESS, SpecifyInputAndOutput({1}, {4}));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_Finish(m_model));

    m_compilation = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, m_compilation);
    OH_NNModel_Destroy(&m_model);
    ASSERT_EQ(nullptr, m_model);

    ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
    ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
    size_t targetDevice = m_devices[0]; // Use the first device in system test.
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

    m_executor = OH_NNExecutor_Construct(m_compilation);
    ASSERT_NE(nullptr, m_executor);
    OH_NNCompilation_Destroy(&m_compilation);
    ASSERT_EQ(nullptr, m_compilation);

    // Set value of secondAddend
    std::vector<float> secondAddendValue(ELEMENT_COUNT, INPUT_TWO);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(0, {3, 2, 2},  (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set output buffer of output
    float outputBuffer[ELEMENT_COUNT];
    ASSERT_EQ(OH_NN_SUCCESS, SetOutput(0, (void*)outputBuffer, ADDEND_DATA_LENGTH));

    // Run inference and assert output value
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
    ASSERT_EQ(OH_NN_SUCCESS, IsExpectedOutput(outputBuffer));

    OH_NNExecutor_Destroy(&m_executor);
    ASSERT_EQ(nullptr, m_executor);
}

/*
 * @tc.name: end_to_end_test_004
 * @tc.desc: Test End-to-End operation of Neural Network Runtime.
 * @tc.type: FUNC
 */
HWTEST_F(End2EndTest, end_to_end_test_004, testing::ext::TestSize.Level1)
{
    // Prepare tensors
    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    CppTensor addend1{OH_NN_FLOAT32, {-1, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor immediateTensor{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor output{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, immediateTensor, output};

    ASSERT_EQ(OH_NN_SUCCESS, BuildModel(tensors));

    m_compilation = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, m_compilation);
    OH_NNModel_Destroy(&m_model);
    ASSERT_EQ(nullptr, m_model);

    ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
    ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
    size_t targetDevice = m_devices[0]; // Use the first device in system test.
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

    m_executor = OH_NNExecutor_Construct(m_compilation);
    ASSERT_NE(nullptr, m_executor);
    OH_NNCompilation_Destroy(&m_compilation);
    ASSERT_EQ(nullptr, m_compilation);

    // Set value of firstAddend
    std::vector<float> firstAddendValue(ELEMENT_COUNT, INPUT_ONE);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(0, {3, 2, 2}, (void*)firstAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set value of secondAddend
    std::vector<float> secondAddendValue(ELEMENT_COUNT, INPUT_TWO);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(1, {3, 2, 2},  (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set output buffer of output
    float outputBuffer[ELEMENT_COUNT];
    ASSERT_EQ(OH_NN_SUCCESS, SetOutput(0, (void*)outputBuffer, ADDEND_DATA_LENGTH));

    // Run inference and assert output value
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
    ASSERT_EQ(OH_NN_SUCCESS, IsExpectedOutput(outputBuffer));

    OH_NNExecutor_Destroy(&m_executor);
    ASSERT_EQ(nullptr, m_executor);
}

/*
 * @tc.name: end_to_end_test_005
 * @tc.desc: Test End-to-End execution with cache setting and loading.
 * @tc.type: FUNC
 */
HWTEST_F(End2EndTest, end_to_end_test_005, testing::ext::TestSize.Level1)
{
    // Prepare tensors
    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    CppTensor addend1{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor immediateTensor{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor output{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, immediateTensor, output};
    ASSERT_EQ(OH_NN_SUCCESS, BuildModel(tensors));

    ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
    ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
    size_t targetDevice = m_devices[0]; // Use the first device in system test.

    // Used to export cache.
    OH_NNCompilation* compilationCacheExporter = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, compilationCacheExporter);

    const fs::path cachePath{CACHE_DIR};
    ASSERT_EQ(false, fs::exists(cachePath));
    ASSERT_EQ(true, fs::create_directory(cachePath));

    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(compilationCacheExporter, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetCache(compilationCacheExporter, CACHE_DIR.c_str(), CACHE_VERSION));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(compilationCacheExporter));
    ASSERT_EQ(false, fs::is_empty(cachePath));
    OH_NNCompilation_Destroy(&compilationCacheExporter);
    ASSERT_EQ(nullptr, compilationCacheExporter);

    // This compilation loads cache.
    m_compilation = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, m_compilation);
    OH_NNModel_Destroy(&m_model);
    ASSERT_EQ(nullptr, m_model);

    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetCache(m_compilation, CACHE_DIR.c_str(), CACHE_VERSION));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

    m_executor = OH_NNExecutor_Construct(m_compilation);
    ASSERT_NE(nullptr, m_executor);
    OH_NNCompilation_Destroy(&m_compilation);
    ASSERT_EQ(nullptr, m_compilation);

    // Set value of firstAddend
    std::vector<float> firstAddendValue(ELEMENT_COUNT, INPUT_ONE);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(0, {3, 2, 2}, (void*)firstAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set value of secondAddend
    std::vector<float> secondAddendValue(ELEMENT_COUNT, INPUT_TWO);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(1, {3, 2, 2},  (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set output buffer of output
    float outputBuffer[ELEMENT_COUNT];
    ASSERT_EQ(OH_NN_SUCCESS, SetOutput(0, (void*)outputBuffer, ADDEND_DATA_LENGTH));

    // Run inference and assert output value
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
    ASSERT_EQ(OH_NN_SUCCESS, IsExpectedOutput(outputBuffer));

    OH_NNExecutor_Destroy(&m_executor);
    ASSERT_EQ(nullptr, m_executor);

    // If cache directory and files and delete, remove_all() should return a value larger than 1.
    // The actual value depends on the implementation of NNRt service.
    ASSERT_GT(fs::remove_all(cachePath), (std::uintmax_t)1);
}

/*
 * @tc.name: end_to_end_test_006
 * @tc.desc: Test End-to-End execution mixing SetInput and SetInputFromMemory functions.
 * @tc.type: FUNC
 */
HWTEST_F(End2EndTest, end_to_end_test_006, testing::ext::TestSize.Level1)
{
    // Prepare tensors
    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    CppTensor addend1{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor immediateTensor{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor output{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, immediateTensor, output};

    ASSERT_EQ(OH_NN_SUCCESS, BuildModel(tensors));

    ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
    ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
    size_t targetDevice = m_devices[0]; // Use the first device in system test.

    // This compilation loads cache.
    m_compilation = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, m_compilation);
    OH_NNModel_Destroy(&m_model);
    ASSERT_EQ(nullptr, m_model);

    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

    m_executor = OH_NNExecutor_Construct(m_compilation);
    ASSERT_NE(nullptr, m_executor);
    OH_NNCompilation_Destroy(&m_compilation);
    ASSERT_EQ(nullptr, m_compilation);

    // Set value of firstAddend
    std::vector<float> firstAddendValue(ELEMENT_COUNT, INPUT_ONE);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(0, {3, 2, 2}, (void*)firstAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set value of secondAddend
    std::vector<float> secondAddendValue(ELEMENT_COUNT, INPUT_TWO);
    OH_NN_Memory* secondAddendMemory;
    ASSERT_EQ(OH_NN_SUCCESS,
        SetInputFromMemory(1, {3, 2, 2}, (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH, &secondAddendMemory));

    // Set output buffer of output
    OH_NN_Memory* outputMemory;
    ASSERT_EQ(OH_NN_SUCCESS, SetOutputFromMemory(0, ADDEND_DATA_LENGTH, &outputMemory));

    // Run inference and assert output value
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
    ASSERT_EQ(OH_NN_SUCCESS, IsExpectedOutput(outputMemory));

    OH_NNExecutor_DestroyInputMemory(m_executor, 1, &secondAddendMemory);
    ASSERT_EQ(nullptr, secondAddendMemory);
    OH_NNExecutor_DestroyOutputMemory(m_executor, 0, &outputMemory);
    ASSERT_EQ(nullptr, outputMemory);

    OH_NNExecutor_Destroy(&m_executor);
    ASSERT_EQ(nullptr, m_executor);
}

/*
 * @tc.name: end_to_end_test_007
 * @tc.desc: Test End-to-End operation of Neural Network Runtime with quantization.
 * @tc.type: FUNC
 */
HWTEST_F(End2EndTest, end_to_end_test_007, testing::ext::TestSize.Level1)
{
    // Prepare tensors
    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    CppQuantParam quantParam1{{8}, {0.2}, {0}};
    CppQuantParam quantParam2{{8}, {0.4}, {0}};
    CppTensor addend1{OH_NN_INT8, {3, 2, 2}, nullptr, 0, quantParam1, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_INT8, {3, 2, 2}, nullptr, 0, quantParam1, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor immediateTensor{OH_NN_INT8, {3, 2, 2}, nullptr, 0, quantParam1, OH_NN_TENSOR};
    CppTensor output{OH_NN_INT8, {3, 2, 2}, nullptr, 0, quantParam2, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, immediateTensor, output};

    ASSERT_EQ(OH_NN_SUCCESS, BuildModel(tensors));

    m_compilation = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, m_compilation);
    OH_NNModel_Destroy(&m_model);
    ASSERT_EQ(nullptr, m_model);

    ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
    ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
    size_t targetDevice = m_devices[0]; // Use the first device in system test.
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

    m_executor = OH_NNExecutor_Construct(m_compilation);
    ASSERT_NE(nullptr, m_executor);
    OH_NNCompilation_Destroy(&m_compilation);
    ASSERT_EQ(nullptr, m_compilation);

    // Set value of firstAddend
    std::vector<int8_t> firstAddendValue(ELEMENT_COUNT, 4);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(0, {3, 2, 2}, (void*)firstAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set value of secondAddend
    std::vector<int8_t> secondAddendValue(ELEMENT_COUNT, 8);
    ASSERT_EQ(OH_NN_SUCCESS, SetInput(1, {3, 2, 2},  (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH));

    // Set output buffer of output
    int8_t outputBuffer[ELEMENT_COUNT];
    ASSERT_EQ(OH_NN_SUCCESS, SetOutput(0, (void*)outputBuffer, ADDEND_DATA_LENGTH));

    // Run inference and assert output value
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        printf("Comparing output with expected value, output index: %d, output value: %d, expected value: %d.",
             i, static_cast<int>(outputBuffer[i]), static_cast<int>(EXPECTED_QUANT_OUTPUT));
        ASSERT_EQ(outputBuffer[i], EXPECTED_QUANT_OUTPUT);
    }

    OH_NNExecutor_Destroy(&m_executor);
    ASSERT_EQ(nullptr, m_executor);
}

/*
 * @tc.name: end_to_end_test_008
 * @tc.desc: Test End-to-End operation of Neural Network Runtime by calling OH_NNExecutor_Run multiple times.
 * @tc.type: FUNC
 */
HWTEST_F(End2EndTest, end_to_end_test_008, testing::ext::TestSize.Level1)
{
    // Prepare tensors
    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    CppTensor addend1{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor immediateTensor{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor output{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, immediateTensor, output};

    ASSERT_EQ(OH_NN_SUCCESS, BuildModel(tensors));

    m_compilation = OH_NNCompilation_Construct(m_model);
    ASSERT_NE(nullptr, m_compilation);
    OH_NNModel_Destroy(&m_model);
    ASSERT_EQ(nullptr, m_model);

    ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
    ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
    size_t targetDevice = m_devices[0]; // Use the first device in system test.
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
    ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

    m_executor = OH_NNExecutor_Construct(m_compilation);
    ASSERT_NE(nullptr, m_executor);
    OH_NNCompilation_Destroy(&m_compilation);
    ASSERT_EQ(nullptr, m_compilation);

    std::vector<float> firstAddendValue(ELEMENT_COUNT, INPUT_ONE);
    std::vector<float> secondAddendValue(ELEMENT_COUNT, INPUT_TWO);
    float outputBuffer[ELEMENT_COUNT];

    // Test inference multiple times.
    for (int i = 0; i < REPEAT_TIMES; i++) {

        // Set value of firstAddend
        ASSERT_EQ(OH_NN_SUCCESS, SetInput(0, {3, 2, 2}, (void*)firstAddendValue.data(), ADDEND_DATA_LENGTH));

        // Set value of secondAddend
        ASSERT_EQ(OH_NN_SUCCESS, SetInput(1, {3, 2, 2},  (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH));

        // Set output buffer of output
        ASSERT_EQ(OH_NN_SUCCESS, SetOutput(0, (void*)outputBuffer, ADDEND_DATA_LENGTH));

        // Run inference and assert output value
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
        ASSERT_EQ(OH_NN_SUCCESS, IsExpectedOutput(outputBuffer));
    }

    OH_NNExecutor_Destroy(&m_executor);
    ASSERT_EQ(nullptr, m_executor);
}
} // namespace SystemTest
} // NeuralNetworkRuntime
} // OHOS