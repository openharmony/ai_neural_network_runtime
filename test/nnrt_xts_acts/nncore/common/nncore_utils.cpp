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
#include <fstream>

#include "nncore_utils.h"
#include "nncore_const.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Test {
OH_NN_UInt32Array TransformUInt32Array(const std::vector<uint32_t>& vector)
{
    uint32_t* data = (vector.empty()) ? nullptr : const_cast<uint32_t*>(vector.data());
    return {data, vector.size()};
}

NN_TensorDesc* createTensorDesc(const int32_t* shape, size_t shapeNum, OH_NN_DataType dataType, OH_NN_Format format)
{
    NN_TensorDesc* tensorDescTmp = OH_NNTensorDesc_Create();
    if (tensorDescTmp == nullptr) {
        LOGE("[NNRtTest]OH_NNTensorDesc_Create failed!");
        return nullptr;
    }

    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetDataType(tensorDescTmp, dataType);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNRtTest]OH_NNTensorDesc_SetDataType failed!ret = %d\n", ret);
        return nullptr;
    }

    if (shape != nullptr) {
        ret = OH_NNTensorDesc_SetShape(tensorDescTmp, shape, shapeNum);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest]OH_NNTensorDesc_SetShape failed!ret = %d\n", ret);
            return nullptr;
        }
    }

    ret = OH_NNTensorDesc_SetFormat(tensorDescTmp, format);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNRtTest]OH_NNTensorDesc_SetShape failed!ret = %d\n", ret);
        return nullptr;
    }

    return tensorDescTmp;
}

int MultiModelBuildEndStep(OH_NNModel *model, const OHNNGraphArgsMulti &graphArgs)
{
    int ret = 0;
    auto graphInputs = TransformUInt32Array(graphArgs.graphInput);
    auto graphOutputs = TransformUInt32Array(graphArgs.graphOutput);
    ret = OH_NNModel_SpecifyInputsAndOutputs(model, &graphInputs, &graphOutputs);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNRtTest] OH_NNModel_SpecifyInputsAndOutputs failed! ret=%d\n", ret);
        return ret;
    }
    ret = OH_NNModel_Finish(model);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNRtTest] OH_NNModel_Finish failed! ret=%d\n", ret);
        return ret;
    }
    return ret;
}

int BuildMultiOpGraph(OH_NNModel *model, const OHNNGraphArgsMulti &graphArgs)
{
    int ret = 0;
    int opCnt = 0;
    for (size_t j = 0; j < graphArgs.operationTypes.size(); j++) {
        for (size_t i = 0; i < graphArgs.operands[j].size(); i++) {
            const OHNNOperandTest &operandTem = graphArgs.operands[j][i];
            NN_TensorDesc* tensorDesc = createTensorDesc(operandTem.shape.data(),
                                                         (uint32_t) operandTem.shape.size(),
                                                         operandTem.dataType, operandTem.format);
            ret = OH_NNModel_AddTensorToModel(model, tensorDesc);
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NNRtTest] OH_NNModel_AddTensor failed! ret=%d\n", ret);
                return ret;
            }
            ret = OH_NNModel_SetTensorType(model, i, operandTem.type);
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NNRtTest] OH_NNBackend_SetModelTensorType failed! ret=%d\n", ret);
                return ret;
            }
            if (std::find(graphArgs.paramIndices[j].begin(), graphArgs.paramIndices[j].end(), opCnt) !=
                graphArgs.paramIndices[j].end()) {
                ret = OH_NNModel_SetTensorData(model, opCnt, operandTem.data, operandTem.length);
                opCnt += 1;
            }
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NNRtTest] OH_NNModel_SetTensorData failed! ret=%d\n", ret);
                return ret;
            }
            OH_NNTensorDesc_Destroy(&tensorDesc);
        }
        auto paramIndices = TransformUInt32Array(graphArgs.paramIndices[j]);
        auto inputIndices = TransformUInt32Array(graphArgs.inputIndices[j]);
        auto outputIndices = TransformUInt32Array(graphArgs.outputIndices[j]);
        ret = OH_NNModel_AddOperation(model, graphArgs.operationTypes[j], &paramIndices, &inputIndices,
        &outputIndices);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNModel_AddOperation failed! ret=%d\n", ret);
            return ret;
        }
    }
    ret = MultiModelBuildEndStep(model, graphArgs);
    return ret;
}

int SingleModelBuildEndStep(OH_NNModel *model, const OHNNGraphArgs &graphArgs)
{
    int ret = 0;
    auto paramIndices = TransformUInt32Array(graphArgs.paramIndices);
    auto inputIndices = TransformUInt32Array(graphArgs.inputIndices);
    auto outputIndices = TransformUInt32Array(graphArgs.outputIndices);
    if (graphArgs.addOperation) {
        ret = OH_NNModel_AddOperation(model, graphArgs.operationType, &paramIndices, &inputIndices,
                                      &outputIndices);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNModel_AddOperation failed! ret=%d\n", ret);
            return ret;
        }
    }
    if (graphArgs.specifyIO) {
        ret = OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNModel_SpecifyInputsAndOutputs failed! ret=%d\n", ret);
            return ret;
        }
    }
    if (graphArgs.build) {
        ret = OH_NNModel_Finish(model);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNModel_Finish failed! ret=%d\n", ret);
            return ret;
        }
    }
    return ret;
}

int BuildSingleOpGraph(OH_NNModel *model, const OHNNGraphArgs &graphArgs)
{
    int ret = 0;
    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        NN_TensorDesc* tensorDesc = createTensorDesc(operandTem.shape.data(),
                                                     (uint32_t) operandTem.shape.size(),
                                                     operandTem.dataType, operandTem.format);
        ret = OH_NNModel_AddTensorToModel(model, tensorDesc);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNModel_AddTensor failed! ret=%d\n", ret);
            return ret;
        }
        ret = OH_NNModel_SetTensorType(model, i, operandTem.type);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNBackend_SetModelTensorType failed! ret=%d\n", ret);
            return ret;
        }
        if (std::find(graphArgs.paramIndices.begin(), graphArgs.paramIndices.end(), i) !=
            graphArgs.paramIndices.end()) {
            ret = OH_NNModel_SetTensorData(model, i, operandTem.data, operandTem.length);
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NNRtTest] OH_NNModel_SetTensorData failed! ret=%d\n", ret);
                return ret;
            }
        }
        OH_NNTensorDesc_Destroy(&tensorDesc);
    }
    ret = SingleModelBuildEndStep(model, graphArgs);
    return ret;
}

int BuildSingleOpGraphWithQuantParams(OH_NNModel *model, const OHNNGraphArgs &graphArgs)
{
    int ret = 0;
    for (size_t i = 0; i < graphArgs.operands.size(); i++) {
        const OHNNOperandTest &operandTem = graphArgs.operands[i];
        NN_TensorDesc* tensorDesc = createTensorDesc(operandTem.shape.data(),
                                                     (uint32_t) operandTem.shape.size(),
                                                     operandTem.dataType, operandTem.format);
        ret = OH_NNModel_AddTensorToModel(model, tensorDesc);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNModel_AddTensor failed! ret=%d\n", ret);
            return ret;
        }

        NN_QuantParam* quantParam = OH_NNQuantParam_Create();
        double scales = 0.2;
        int32_t zeroPoints = 0;
        uint32_t numBits = 8;
        ret = OH_NNQuantParam_SetScales(quantParam, &scales, 1);
        ret = OH_NNQuantParam_SetZeroPoints(quantParam, &zeroPoints, 1);
        ret = OH_NNQuantParam_SetNumBits(quantParam, &numBits, 1);
        ret = OH_NNModel_SetTensorQuantParams(model, i, quantParam);
        ret = OH_NNQuantParam_Destroy(&quantParam);
        ret = OH_NNModel_SetTensorType(model, i, operandTem.type);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNBackend_SetModelTensorType failed! ret=%d\n", ret);
            return ret;
        }
        if (std::find(graphArgs.paramIndices.begin(), graphArgs.paramIndices.end(), i) !=
            graphArgs.paramIndices.end()) {
            ret = OH_NNModel_SetTensorData(model, i, operandTem.data, operandTem.length);
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NNRtTest] OH_NNModel_SetTensorData failed! ret=%d\n", ret);
                return ret;
            }
        }
        OH_NNTensorDesc_Destroy(&tensorDesc);
    }
    ret = SingleModelBuildEndStep(model, graphArgs);
    return ret;
}

OH_NN_ReturnCode GetDeviceID(size_t *deviceId)
{
    OH_NN_ReturnCode ret = OH_NN_FAILED;
    const size_t *devicesID{nullptr};
    uint32_t devicesCount{0};
    ret = OH_NNDevice_GetAllDevicesID(&devicesID, &devicesCount);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNRtTest] OH_NNDevice_GetAllDevicesID failed! ret=%d\n", ret);
        return ret;
    }
    if (devicesCount <= NO_DEVICE_COUNT) {
        LOGE("[NNRtTest] devicesCount <= 0  devicesCount=%d\n", devicesCount);
        return OH_NN_FAILED;
    }

    const char *name = nullptr;
    std::string deviceName{"Device-CPU_TestVendor_v2_0"};
    for (uint32_t i = 0; i < devicesCount; i++) {
        name = nullptr;
        ret = OH_NNDevice_GetName(devicesID[i], &name);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNDevice_GetName failed! ret=%d\n", ret);
            return ret;
        }

        std::string sName(name);
        if (deviceName == sName) {
            *deviceId = devicesID[i];
            return OH_NN_SUCCESS;
        }
    }
    return OH_NN_FAILED;
}

OH_NN_ReturnCode SetDevice(OH_NNCompilation *compilation)
{
    OH_NN_ReturnCode ret = OH_NN_FAILED;
    const size_t *devicesID{nullptr};
    uint32_t devicesCount{0};
    ret = OH_NNDevice_GetAllDevicesID(&devicesID, &devicesCount);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNRtTest] OH_NNDevice_GetAllDevicesID failed! ret=%d\n", ret);
        return ret;
    }
    if (devicesCount <= NO_DEVICE_COUNT) {
        LOGE("[NNRtTest] devicesCount <= 0  devicesCount=%d\n", devicesCount);
        return OH_NN_FAILED;
    }

    const char *name = nullptr;
    std::string deviceName{"Device-CPU_TestVendor_v2_0"};
    for (uint32_t i = 0; i < devicesCount; i++) {
        name = nullptr;
        ret = OH_NNDevice_GetName(devicesID[i], &name);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNDevice_GetName failed! ret=%d\n", ret);
            return ret;
        }

        std::string sName(name);
        if (deviceName == sName) {
            ret = OH_NNCompilation_SetDevice(compilation, devicesID[i]);
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NNRtTest] OH_NNCompilation_SetDevice failed! ret=%d\n", ret);
                return ret;
            }
            return OH_NN_SUCCESS;
        }
    }
    return OH_NN_FAILED;
}

int CompileGraphMock(OH_NNCompilation *compilation, const OHNNCompileParam &compileParam)
{
    int ret = 0;
    ret = SetDevice(compilation);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NNRtTest] OH_NNCompilation_SetDevice failed! ret=%d\n", ret);
        return ret;
    }
    // set cache
    if (!compileParam.cacheDir.empty()) {
        ret = OH_NNCompilation_SetCache(compilation, compileParam.cacheDir.c_str(),
        compileParam.cacheVersion);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNCompilation_SetCache failed! ret=%d\n", ret);
            return ret;
        }
    }
    // set performance
    if (compileParam.performanceMode != OH_NN_PERFORMANCE_NONE) {
        ret = OH_NNCompilation_SetPerformanceMode(compilation, compileParam.performanceMode);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNCompilation_SetPerformanceMode failed! ret=%d\n", ret);
            return ret;
        }
    }
    // set priority
    if (compileParam.priority != OH_NN_PRIORITY_NONE) {
        ret = OH_NNCompilation_SetPriority(compilation, compileParam.priority);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNCompilation_SetPriority failed! ret=%d\n", ret);
            return ret;
        }
    }
    // enable fp16
    if (compileParam.enableFp16) {
        ret = OH_NNCompilation_EnableFloat16(compilation, compileParam.enableFp16);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNCompilation_EnableFloat16 failed! ret=%d\n", ret);
            return ret;
        }
    }
    // build
    ret = OH_NNCompilation_Build(compilation);
    return ret;
}

void Free(OH_NNModel *model, OH_NNCompilation *compilation, OH_NNExecutor *executor)
{
    if (model != nullptr) {
        OH_NNModel_Destroy(&model);
        ASSERT_EQ(nullptr, model);
    }
    if (compilation != nullptr) {
        OH_NNCompilation_Destroy(&compilation);
        ASSERT_EQ(nullptr, compilation);
    }
    if (executor != nullptr) {
        OH_NNExecutor_Destroy(&executor);
        ASSERT_EQ(nullptr, executor);
    }
}

PathType CheckPath(const std::string &path)
{
    if (path.empty()) {
        LOGI("CheckPath: path is null");
        return PathType::NOT_FOUND;
    }
    struct stat buf{};
    if (stat(path.c_str(), &buf) == 0) {
        if (buf.st_mode & S_IFDIR) {
            return PathType::DIR;
        } else if (buf.st_mode & S_IFREG) {
            return PathType::FILE;
        } else {
            return PathType::UNKNOWN;
        }
    }
    LOGI("%s not found", path.c_str());
    return PathType::NOT_FOUND;
}

bool DeleteFile(const std::string &path)
{
    if (path.empty()) {
        LOGI("DeleteFile: path is null");
        return false;
    }
    if (CheckPath(path) == PathType::NOT_FOUND) {
        LOGI("not found: %s", path.c_str());
        return true;
    }
    if (remove(path.c_str()) == 0) {
        LOGI("deleted: %s", path.c_str());
        return true;
    }
    LOGI("delete failed: %s", path.c_str());
    return false;
}

void CopyFile(const std::string &srcPath, const std::string &dstPath)
{
    std::ifstream src(srcPath, std::ios::binary);
    std::ofstream dst(dstPath, std::ios::binary);

    dst << src.rdbuf();
}

std::string ConcatPath(const std::string &str1, const std::string &str2)
{
    // boundary
    if (str2.empty()) {
        return str1;
    }
    if (str1.empty()) {
        return str2;
    }
    // concat
    char end = str1[str1.size() - 1];
    if (end == '\\' or end == '/') {
        return str1 + str2;
    } else {
        return str1 + '/' + str2;
    }
}

void DeleteFolder(const std::string &path)
{
    if (path.empty()) {
        LOGI("DeletePath: path is null");
        return;
    }

    DIR *dir = opendir(path.c_str());
    // check is dir ?
    if (dir == nullptr) {
        LOGE("[NNRtTest] Can not open dir. Check path or permission! path: %s", path.c_str());
        return;
    }
    struct dirent *file;
    // read all the files in dir
    std::vector <std::string> pathList;
    while ((file = readdir(dir)) != nullptr) {
        // skip "." and ".."
        if (strcmp(file->d_name, ".") == 0 || strcmp(file->d_name, "..") == 0) {
            continue;
        }
        if (file->d_type == DT_DIR) {
            std::string filePath = path + "/" + file->d_name;
            DeleteFolder(filePath); // 递归执行
        } else {
            pathList.emplace_back(ConcatPath(path, file->d_name));
        }
    }
    closedir(dir);
    pathList.emplace_back(path);
    LOGI("[Common] Delete folder %s", path.c_str());
    for (auto &i : pathList) {
        DeleteFile(i);
    }
}

bool CreateFolder(const std::string &path)
{
    if (path.empty()) {
        LOGI("CreateFolder: path is empty");
        return false;
    }
    LOGI("CreateFolder:%s", path.c_str());
    mode_t mode = 0700;
    for (size_t i = 1; i < path.size() - 1; i++) {
        if (path[i] != '/') {
            continue;
        }
        PathType ret = CheckPath(path.substr(0, i));
        switch (ret) {
            case PathType::DIR:
                continue;
            case PathType::NOT_FOUND:
                LOGI("mkdir: %s", path.substr(0, i).c_str());
                mkdir(path.substr(0, i).c_str(), mode);
                break;
            default:
                LOGI("error: %s", path.substr(0, i).c_str());
                return false;
        }
    }
    mkdir(path.c_str(), mode);
    return CheckPath(path) == PathType::DIR;
}

bool CheckOutput(const float* output, const float* expect)
{
    if (output == nullptr || expect == nullptr) {
        LOGE("[NNRtTest] output or expect is nullptr\n");
        return false;
    }
    for (int i = 0; i < ELEMENT_COUNT; i++) {
        if (std::abs(float(output[i]) - float(expect[i])) > 1e-8) {
            for (int j = 0; j < ELEMENT_COUNT; j++) {
                LOGE("[NNRtTest] output %d not match: expect:%f, actual:%f\n", j, float(expect[j]), float(output[j]));
            }
            return false;
        }
    }
    return true;
}

//创建定长模型
void ConstructAddModel(OH_NNModel **model)
{
    *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    AddModel addModel;
    OHNNGraphArgs graphArgs = addModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(*model, graphArgs));
}

//定长模型创建compilation
void ConstructCompilation(OH_NNCompilation **compilation, OH_NNModel **model)
{
    ConstructAddModel(model);
    *compilation = OH_NNCompilation_Construct(*model);
    ASSERT_NE(nullptr, *compilation);
}

//通过定长compilation创建executor
void CreateExecutor(OH_NNExecutor **executor)
{
    OH_NNCompilation *compilation = nullptr;
    OH_NNModel *model = nullptr;
    ConstructCompilation(&compilation, &model);
    OHNNCompileParam compileParam{
        .performanceMode = OH_NN_PERFORMANCE_HIGH,
        .priority = OH_NN_PRIORITY_HIGH,
    };
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, *executor);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

void CreateDynamicExecutor(OH_NNExecutor **executor)
{
    OH_NNModel *model = OH_NNModel_Construct();
    ASSERT_NE(nullptr, model);
    AvgPoolDynamicModel avgModel;
    OHNNGraphArgs graphArgs = avgModel.graphArgs;
    ASSERT_EQ(OH_NN_SUCCESS, BuildSingleOpGraph(model, graphArgs));

    OH_NNCompilation *compilation = OH_NNCompilation_Construct(model);
    ASSERT_NE(nullptr, compilation);

    OHNNCompileParam compileParam{
        .performanceMode = OH_NN_PERFORMANCE_HIGH,
        .priority = OH_NN_PRIORITY_HIGH,
    };
    ASSERT_EQ(OH_NN_SUCCESS, CompileGraphMock(compilation, compileParam));
    *executor = OH_NNExecutor_Construct(compilation);
    ASSERT_NE(nullptr, *executor);
    OH_NNModel_Destroy(&model);
    OH_NNCompilation_Destroy(&compilation);
}

void GetExecutorInputOutputTensorDesc(OH_NNExecutor* executor,
                                      std::vector<NN_TensorDesc*>& inputTensorDescs, size_t& inputCount,
                                      std::vector<NN_TensorDesc*>& outputTensorDescs, size_t& outputCount)
{
    OH_NN_ReturnCode ret = OH_NNExecutor_GetInputCount(executor, &inputCount);
    ASSERT_EQ(OH_NN_SUCCESS, ret);
    NN_TensorDesc* tensorDescTmp = nullptr;
    for (size_t i = 0; i < inputCount; ++i) {
        tensorDescTmp = OH_NNExecutor_CreateInputTensorDesc(executor, i);
        ASSERT_NE(nullptr, tensorDescTmp);
        inputTensorDescs.emplace_back(tensorDescTmp);
    }

    ret = OH_NNExecutor_GetOutputCount(executor, &outputCount);
    ASSERT_EQ(OH_NN_SUCCESS, ret);
    for (size_t i = 0; i < outputCount; ++i) {
        tensorDescTmp = OH_NNExecutor_CreateOutputTensorDesc(executor, i);
        ASSERT_NE(nullptr, tensorDescTmp);
        outputTensorDescs.emplace_back(tensorDescTmp);
    }
}

void GetExecutorInputOutputTensorByDesc(OH_NNExecutor* executor,
    std::vector<NN_Tensor*>& inputTensors, const std::vector<NN_TensorDesc*>& inputTensorDescs,
    std::vector<NN_Tensor*>& outputTensors, const std::vector<NN_TensorDesc*>& outputTensorDescs)
{
    size_t deviceID = 0;
    if (OH_NN_SUCCESS != GetDeviceID(&deviceID)) {
        LOGE("Get deviceid failed.");
        return;
    }
    NN_Tensor* tensor = nullptr;
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        tensor = nullptr;
        tensor = OH_NNTensor_Create(deviceID, inputTensorDescs[i]);
        ASSERT_NE(nullptr, tensor);
        inputTensors.emplace_back(tensor);
    }

    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        tensor = nullptr;
        tensor = OH_NNTensor_Create(deviceID, outputTensorDescs[i]);
        ASSERT_NE(nullptr, tensor);
        outputTensors.emplace_back(tensor);
    }
}

void GetExecutorInputOutputTensor(OH_NNExecutor* executor, std::vector<NN_Tensor*>& inputTensors, size_t& inputCount,
                                  std::vector<NN_Tensor*>& outputTensors, size_t& outputCount)
{
    std::vector<NN_TensorDesc*> inputTensorDescs;
    std::vector<NN_TensorDesc*> outputTensorDescs;
    OH_NN_ReturnCode ret = OH_NNExecutor_GetInputCount(executor, &inputCount);
    ASSERT_EQ(OH_NN_SUCCESS, ret);
    NN_TensorDesc* tensorDescTmp = nullptr;
    for (size_t i = 0; i < inputCount; ++i) {
        tensorDescTmp = OH_NNExecutor_CreateInputTensorDesc(executor, i);
        ASSERT_NE(nullptr, tensorDescTmp);
        inputTensorDescs.emplace_back(tensorDescTmp);
    }

    ret = OH_NNExecutor_GetOutputCount(executor, &outputCount);
    ASSERT_EQ(OH_NN_SUCCESS, ret);
    for (size_t i = 0; i < outputCount; ++i) {
        tensorDescTmp = OH_NNExecutor_CreateOutputTensorDesc(executor, i);
        ASSERT_NE(nullptr, tensorDescTmp);
        outputTensorDescs.emplace_back(tensorDescTmp);
    }

    size_t deviceID = 0;
    if (OH_NN_SUCCESS != GetDeviceID(&deviceID)) {
        LOGE("Get deviceid failed.");
        return;
    }
    NN_Tensor* tensor = nullptr;
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        tensor = nullptr;
        tensor = OH_NNTensor_Create(deviceID, inputTensorDescs[i]);
        ASSERT_NE(nullptr, tensor);
        inputTensors.emplace_back(tensor);
    }

    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        tensor = nullptr;
        tensor = OH_NNTensor_Create(deviceID, outputTensorDescs[i]);
        ASSERT_NE(nullptr, tensor);
        outputTensors.emplace_back(tensor);
    }

    DestroyTensorDesc(inputTensorDescs, outputTensorDescs);
}

OH_NN_ReturnCode DestroyTensorDesc(
    std::vector<NN_TensorDesc*>& inputTensorDescs, std::vector<NN_TensorDesc*>& outputTensorDescs)
{
    // 销毁输入输出tensordesc
    OH_NN_ReturnCode returnCode {OH_NN_FAILED};
    for (size_t i = 0; i < inputTensorDescs.size(); ++i) {
        returnCode = OH_NNTensorDesc_Destroy(&inputTensorDescs[i]);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNTensorDesc_Destroy failed.");
            return returnCode;
        }
    }
    for (size_t i = 0; i < outputTensorDescs.size(); ++i) {
        returnCode = OH_NNTensorDesc_Destroy(&outputTensorDescs[i]);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNTensorDesc_Destroy failed.");
            return returnCode;
        }
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode DestroyTensor(
    std::vector<NN_Tensor*>& inputTensors, std::vector<NN_Tensor*>& outputTensors)
{
    // 清理输入输出Tensor
    OH_NN_ReturnCode returnCode {OH_NN_FAILED};
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        returnCode = OH_NNTensor_Destroy(&inputTensors[i]);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNTensor_Destroy failed.");
            return returnCode;
        }
    }
    for (size_t i = 0; i < outputTensors.size(); ++i) {
        returnCode = OH_NNTensor_Destroy(&outputTensors[i]);
        if (returnCode != OH_NN_SUCCESS) {
            LOGE("End2EndTest::OH_NNTensor_Destroy failed.");
            return returnCode;
        }
    }

    return OH_NN_SUCCESS;
}
} // namespace Test
} // namespace NeuralNetworkRuntime
} // namespace OHOS