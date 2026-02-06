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
#ifndef NNRT_UTILS_H
#define NNRT_UTILS_H

#include <dirent.h>
#include <sys/stat.h>
#include <gtest/gtest.h>

#include "neural_network_runtime/neural_network_runtime.h"
#include "log.h"
#include "mock_idevice.h"
#include "const.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Test {
namespace V2_0 = OHOS::HDI::Nnrt::V2_0;
struct OHNNOperandTest {
    OH_NN_DataType dataType;
    OH_NN_TensorType type;
    std::vector<int32_t> shape;
    void *data{nullptr};
    int32_t length{0};
    const OH_NN_QuantParam *quantParam = nullptr;
};

struct OHNNGraphArgs {
    OH_NN_OperationType operationType;
    std::vector<OHNNOperandTest> operands;
    std::vector<uint32_t> paramIndices;
    std::vector<uint32_t> inputIndices;
    std::vector<uint32_t> outputIndices;
    bool build = true;
    bool specifyIO = true;
    bool addOperation = true;
};

struct OHNNGraphArgsMulti {
    std::vector<OH_NN_OperationType> operationTypes;
    std::vector<std::vector<OHNNOperandTest>> operands;
    std::vector<std::vector<uint32_t>> paramIndices;
    std::vector<std::vector<uint32_t>> inputIndices;
    std::vector<std::vector<uint32_t>> outputIndices;
    std::vector<uint32_t> graphInput;
    std::vector<uint32_t> graphOutput;
};

struct OHNNCompileParam {
    int32_t deviceId = 0;
    std::string cacheDir;
    uint32_t cacheVersion = 0;
    OH_NN_PerformanceMode performanceMode = OH_NN_PERFORMANCE_NONE;
    OH_NN_Priority priority = OH_NN_PRIORITY_NONE;
    bool enableFp16 = false;
};

int BuildSingleOpGraph(OH_NNModel *model, const OHNNGraphArgs &graphArgs);

int ExecutorWithMemory(OH_NNExecutor *executor, const OHNNGraphArgs &graphArgs, OH_NN_Memory *OHNNMemory[],
    float* expect);

void Free(OH_NNModel *model = nullptr, OH_NNCompilation *compilation = nullptr, OH_NNExecutor *executor = nullptr);

int CompileGraphMock(OH_NNCompilation *compilation, const OHNNCompileParam &compileParam);

int ExecuteGraphMock(OH_NNExecutor *executor, const OHNNGraphArgs &graphArgs, float* expect);

OH_NN_ReturnCode SetDevice(OH_NNCompilation *compilation);
int BuildMultiOpGraph(OH_NNModel *model, const OHNNGraphArgsMulti &graphArgs);
OH_NN_UInt32Array GetUInt32Array(std::vector<uint32_t> indices);

bool CheckOutput(const float* output, const float* expect);

enum class PathType { FILE, DIR, UNKNOWN, NOT_FOUND };
PathType CheckPath(const std::string &path);
bool DeleteFile(const std::string &path);
void CopyFile(const std::string &srcPath, const std::string &dstPath);
std::string ConcatPath(const std::string &str1, const std::string &str2);
void DeleteFolder(const std::string &path);
bool CreateFolder(const std::string &path);

} // namespace Test
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NNRT_UTILS_H