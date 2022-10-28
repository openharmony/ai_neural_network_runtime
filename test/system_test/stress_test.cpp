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

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

#include "securec.h"

#include "test/system_test/common/nnrt_test.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace SystemTest {
constexpr int TMP_LENGTH = 32;
constexpr int PATH_LENGTH = 255;
constexpr int STRESS_COUNT = 10000000;
const float EPSILON = 1e-4;
const uint32_t NO_DEVICE_COUNT = 0;
const uint32_t ADDEND_DATA_LENGTH = 12 * sizeof(float);
const std::string VMRSS = "VmSize:";

class StressTest : public NNRtTest {
public:
    StressTest() = default;
};

std::string GetVMRSS(pid_t pid)
{
    std::string fileName{"/proc/"};
    fileName += std::to_string(pid) + "/status";
    std::ifstream ifs(fileName, std::ios::binary);
    if (!ifs.is_open()) {
        std::cout << "Failed to open " << fileName << std::endl;
        return "";
    }

    std::string vmRss;
    // Extract physical memory use from process status.
    while (!ifs.eof()) {
        getline(ifs, vmRss);
        // Compare the first seven characters, which is "VmSize:".
        if (vmRss.compare(0, 7, VMRSS) == 0) {
            break;
        }
    }
    ifs.close();

    time_t t = time(nullptr);
    char tmp[TMP_LENGTH] {' '};
    strftime(&(tmp[1]), TMP_LENGTH * sizeof(char), "%Y-%m-%d %H:%M:%S", localtime(&t));

    return vmRss + tmp;
}

void PrintVMRSS(pid_t pid)
{
    char path[PATH_LENGTH];
    if (!getcwd(path, PATH_LENGTH)) {
        std::cout << "Failed to get current path" << std::endl;
        return;
    }
    std::string pathStr = path;
    std::string pathFull = pathStr + "/RealtimeVMRSS_" + std::to_string(pid) + ".txt";

    std::ofstream out(pathFull, std::ios::app);
    if (!out.is_open()) {
        std::cout << "Some error occurs" << std::endl;
        return;
    }

    while (true) {
        std::string rss = GetVMRSS(pid);
        if (rss.empty()) {
            std::cout << "Some error occurs" << std::endl;
            out.close();
            return;
        }

        out << rss << std::endl;
        sleep(1);
    }
}

/*
 * @tc.name: stress_test_001
 * @tc.desc: Check memory leak by repeatly implement end-to-end execution.
 * @tc.type: FUNC
 */
HWTEST_F(StressTest, stress_test_001, testing::ext::TestSize.Level1)
{
    std::cout << "Start RunDoubleConvStressTest test cast." << std::endl;

    pid_t pidOfStressTest = getpid();
    std::thread thread(PrintVMRSS, pidOfStressTest);

    size_t targetDevice{0};

    int8_t activationValue{0};
    CppQuantParam quantParam{{}, {}, {}};
    CppTensor addend1{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor addend2{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    CppTensor activation{OH_NN_INT8, {}, (void*)(&activationValue), 1, quantParam, OH_NN_ADD_ACTIVATIONTYPE};
    CppTensor output{OH_NN_FLOAT32, {3, 2, 2}, nullptr, 0, quantParam, OH_NN_TENSOR};
    std::vector<CppTensor> tensors{addend1, addend2, activation, output};

    std::vector<float> firstAddendValue(12, 1.23);
    std::vector<float> secondAddendValue(12, 2.34);
    float outputBuffer[12];
    std::vector<float> expectedOutput(12, 3.57);

    for (int i = 0; i < STRESS_COUNT; i++) {
        tensors = {addend1, addend2, activation, output};

        m_model = OH_NNModel_Construct();
        ASSERT_NE(nullptr, m_model);
        ASSERT_EQ(OH_NN_SUCCESS, AddTensors(tensors));
        ASSERT_EQ(OH_NN_SUCCESS, AddOperation(OH_NN_OPS_ADD, {2}, {0, 1}, {3}));
        ASSERT_EQ(OH_NN_SUCCESS, SpecifyInputAndOutput({0, 1}, {3}));
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNModel_Finish(m_model));

        m_compilation = OH_NNCompilation_Construct(m_model);
        ASSERT_NE(nullptr, m_compilation);
        OH_NNModel_Destroy(&m_model);
        ASSERT_EQ(nullptr, m_model);

        ASSERT_EQ(OH_NN_SUCCESS, GetDevices());
        ASSERT_GT(m_devices.size(), NO_DEVICE_COUNT); // Expect available accelerator.
        targetDevice = m_devices[0]; // Use the first device in system test.
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_SetDevice(m_compilation, targetDevice));
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNCompilation_Build(m_compilation));

        m_executor = OH_NNExecutor_Construct(m_compilation);
        ASSERT_NE(nullptr, m_executor);
        OH_NNCompilation_Destroy(&m_compilation);
        ASSERT_EQ(nullptr, m_compilation);

        // Set value of firstAddend
        ASSERT_EQ(OH_NN_SUCCESS, SetInput(0, {3, 2, 2}, (void*)firstAddendValue.data(), ADDEND_DATA_LENGTH));

        // Set value of secondAddend
        ASSERT_EQ(OH_NN_SUCCESS, SetInput(1, {3, 2, 2},  (void*)secondAddendValue.data(), ADDEND_DATA_LENGTH));

        // Set output buffer of output
        ASSERT_EQ(OH_NN_SUCCESS, SetOutput(0, (void*)outputBuffer, ADDEND_DATA_LENGTH));

        // Run inference and assert output value
        ASSERT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(m_executor));
        for (int j = 0; j < 12; j++) {
            ASSERT_LE(std::abs(outputBuffer[j]-expectedOutput[j]), EPSILON);
        }

        OH_NNExecutor_Destroy(&m_executor);
        ASSERT_EQ(nullptr, m_executor);

        m_tensors.clear();
        m_quantParams.clear();
        m_nodes.clear();
        m_inputs.clear();
        m_outputs.clear();
        m_devices.clear();

        if (i % 1000 == 0) {
            std::cout << "Execute " << i << "times." << std::endl;
        }
    }
    thread.join();
}
} // namespace SystemTest
} // NeuralNetworkRuntime
} // OHOS