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
#include "nncore_const.h"

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
    OH_NN_Format format = OH_NN_FORMAT_NONE;
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

struct AddModel {
    // ADD MODEL
    float inputValue0[4] = {0, 1, 2, 3};
    float inputValue1[4] = {0, 1, 2, 3};
    int8_t activationValue = OH_NN_FUSED_NONE;
    float outputValue[4] = {0};
    float expectValue[4] = {0, 2, 4, 6};

    OHNNOperandTest input0 = {OH_NN_FLOAT32, OH_NN_TENSOR, TENSOR_SHAPE, inputValue0, ADD_DATA_LENGTH};
    OHNNOperandTest input1 = {OH_NN_FLOAT32, OH_NN_TENSOR, TENSOR_SHAPE, inputValue1, ADD_DATA_LENGTH};
    OHNNOperandTest activation = {OH_NN_INT8, OH_NN_ADD_ACTIVATIONTYPE, {}, &activationValue, sizeof(int8_t)};
    OHNNOperandTest output = {OH_NN_FLOAT32, OH_NN_TENSOR, TENSOR_SHAPE, outputValue, ADD_DATA_LENGTH};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_ADD,
                               .operands = {input0, input1, activation, output},
                               .paramIndices = {2},
                               .inputIndices = {0, 1},
                               .outputIndices = {3}};
};

struct AvgPoolDynamicModel {
    // ADD MODEL
    float inputValue0[4] = {0, 1, 2, 3};
    float inputValue1[4] = {0, 1, 2, 3};
    int8_t activationValue = OH_NN_FUSED_NONE;
    float outputValue[4] = {0};
    float expectValue[4] = {0, 2, 4, 6};

    OHNNOperandTest input0 = {OH_NN_FLOAT32, OH_NN_TENSOR, {-1, -1, -1, -1}, inputValue0, ADD_DATA_LENGTH};
    OHNNOperandTest input1 = {OH_NN_FLOAT32, OH_NN_TENSOR, {-1, -1, -1, -1}, inputValue1, ADD_DATA_LENGTH};
    OHNNOperandTest activation = {OH_NN_INT8, OH_NN_ADD_ACTIVATIONTYPE, {}, &activationValue, sizeof(int8_t)};
    OHNNOperandTest output = {OH_NN_FLOAT32, OH_NN_TENSOR, {-1, -1, -1, -1}, outputValue, ADD_DATA_LENGTH};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_ADD,
                               .operands = {input0, input1, activation, output},
                               .paramIndices = {2},
                               .inputIndices = {0, 1},
                               .outputIndices = {3}};
};

struct TopKModel {
    // TopK Model
    float valueX[6] = {0, 1, 2, 3, 4, 5};
    int8_t valueK = 2;
    bool valueSorted = true;
    float valueOutput1[2];
    int32_t valueOutput2[2];

    OHNNOperandTest x = {OH_NN_FLOAT32, OH_NN_TENSOR, {1, 6}, valueX, 6 * sizeof(float)};
    OHNNOperandTest k = {OH_NN_INT8, OH_NN_TENSOR, {}, &valueK, sizeof(int8_t)};
    OHNNOperandTest sorted = {OH_NN_BOOL, OH_NN_TOP_K_SORTED, {}, &valueSorted, sizeof(bool)};
    OHNNOperandTest output1 = {OH_NN_FLOAT32, OH_NN_TENSOR, {1, 2}, valueOutput1, 2 * sizeof(float)};
    OHNNOperandTest output2 = {OH_NN_INT32, OH_NN_TENSOR, {1, 2}, valueOutput2, 2 * sizeof(int32_t)};

    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_TOP_K,
                               .operands = {x, k, sorted, output1, output2},
                               .paramIndices = {2},
                               .inputIndices = {0, 1},
                               .outputIndices = {3, 4}};
};

class AddTopKModel {
    // Build two ops Model
private:
    AddModel addModel;
    TopKModel topKModel;

public:
    OHNNGraphArgsMulti graphArgs = {
        .operationTypes = {OH_NN_OPS_ADD, OH_NN_OPS_TOP_K},
        .operands = {{addModel.input0, addModel.input1, addModel.activation, addModel.output},
            {topKModel.k, topKModel.sorted, topKModel.output1, topKModel.output2}},
        .paramIndices = {{2}, {5}},
        .inputIndices = {{0, 1}, {3, 4}},
        .outputIndices = {{3}, {6, 7}},
        .graphInput = {0, 1, 4},
        .graphOutput = {6, 7}};
};

NN_TensorDesc* createTensorDesc(const int32_t* shape, size_t shapeNum, OH_NN_DataType dataType, OH_NN_Format format);
int MultiModelBuildEndStep(OH_NNModel *model, const OHNNGraphArgsMulti &graphArgs);
int SingleModelBuildEndStep(OH_NNModel *model, const OHNNGraphArgs &graphArgs);
int BuildSingleOpGraph(OH_NNModel *model, const OHNNGraphArgs &graphArgs);
int BuildSingleOpGraphWithQuantParams(OH_NNModel *model, const OHNNGraphArgs &graphArgs);
void Free(OH_NNModel *model = nullptr, OH_NNCompilation *compilation = nullptr, OH_NNExecutor *executor = nullptr);
int CompileGraphMock(OH_NNCompilation *compilation, const OHNNCompileParam &compileParam);
OH_NN_ReturnCode SetDevice(OH_NNCompilation *compilation);
int BuildMultiOpGraph(OH_NNModel *model, const OHNNGraphArgsMulti &graphArgs);
OH_NN_UInt32Array GetUInt32Array(std::vector<uint32_t> indices);
bool CheckOutput(const float* output, const float* expect);
OH_NN_ReturnCode GetDeviceID(size_t *deviceId);

//文件相关
enum class PathType { FILE, DIR, UNKNOWN, NOT_FOUND };
PathType CheckPath(const std::string &path);
bool DeleteFile(const std::string &path);
void CopyFile(const std::string &srcPath, const std::string &dstPath);
std::string ConcatPath(const std::string &str1, const std::string &str2);
void DeleteFolder(const std::string &path);
bool CreateFolder(const std::string &path);

//nncore创建相关
void ConstructAddModel(OH_NNModel **model);
void ConstructCompilation(OH_NNCompilation **compilation, OH_NNModel **model);
void CreateExecutor(OH_NNExecutor **executor);
void CreateDynamicExecutor(OH_NNExecutor **executor);
void GetExecutorInputOutputTensorDesc(OH_NNExecutor* executor,
                                      std::vector<NN_TensorDesc*>& inputTensorDescs, size_t& inputCount,
                                      std::vector<NN_TensorDesc*>& outputTensorDescs, size_t& outputCount);
void GetExecutorInputOutputTensorByDesc(OH_NNExecutor* executor,
    std::vector<NN_Tensor*>& inputTensors, const std::vector<NN_TensorDesc*>& inputTensorDescs,
    std::vector<NN_Tensor*>& outputTensors, const std::vector<NN_TensorDesc*>& outputTensorDescs);
void GetExecutorInputOutputTensor(OH_NNExecutor* executor, std::vector<NN_Tensor*>& inputTensors, size_t& inputCount,
                                  std::vector<NN_Tensor*>& outputTensors, size_t& outputCount);
OH_NN_ReturnCode DestroyTensorDesc(std::vector<NN_TensorDesc*>& inputTensorDescs,
                                   std::vector<NN_TensorDesc*>& outputTensorDescs);
OH_NN_ReturnCode DestroyTensor(std::vector<NN_Tensor*>& inputTensors, std::vector<NN_Tensor*>& outputTensors);
} // namespace Test
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NNRT_UTILS_H