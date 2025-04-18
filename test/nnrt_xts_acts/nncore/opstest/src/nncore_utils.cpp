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
#include <fstream>

#include "nncore_utils.h"

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
            LOGE("[NNRtTest] OH_NNModel_AddOperation failed! ret=%{public}d\n", ret);
            return ret;
        }
    }

    if (graphArgs.specifyIO) {
        ret = OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NNRtTest] OH_NNModel_SpecifyInputsAndOutputs failed! ret=%{public}d\n", ret);
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
                LOGE("[NNRtTest] OH_NNModel_SetTensorData failed! ret=%{public}d\n", ret);
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
    std::string deviceName{"Device-CPU_TestVendor_v2_1"};
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

int CompileGraphMock(OH_NNCompilation *compilation, const OHNNCompileParam &compileParam)
{
    int ret = 0;

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

void FreeTensorDescVec(std::vector<NN_TensorDesc*> tensorDescVec)
{
    if (!tensorDescVec.empty()) {
        for (auto tensorDesc : tensorDescVec) {
            OH_NNTensorDesc_Destroy(&tensorDesc);
            ASSERT_EQ(nullptr, tensorDesc);
        }
    }
}
} // namespace Test
} // namespace NeuralNetworkRuntime
} // namespace OHOS