/*
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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

#include <vector>
#include "nnrtops_fuzzer.h"
#include "../data.h"
#include "../../../common/log.h"

#include "neural_network_runtime_type.h"
#include "neural_network_runtime.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
constexpr size_t U32_AT_SIZE = 4;
struct OHNNOperandTest {
    OH_NN_DataType dataType;
    OH_NN_TensorType type;
    std::vector<int32_t> shape;
    void *data {nullptr};
    int32_t length {0};
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

struct Model0 {
    float value = 1;
    OHNNOperandTest input = {OH_NN_FLOAT32, OH_NN_TENSOR, {1}, &value, sizeof(float)};
    OHNNOperandTest output = {OH_NN_FLOAT32, OH_NN_TENSOR, {1}, &value, sizeof(float)};
    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_ADD,
                               .operands = {input, output},
                               .paramIndices = {},
                               .inputIndices = {0},
                               .outputIndices = {1}};
};

OH_NN_UInt32Array TransformUInt32Array(const std::vector<uint32_t>& vector)
{
    uint32_t* data = (vector.empty()) ? nullptr : const_cast<uint32_t*>(vector.data());
    return {data, vector.size()};
}

NN_TensorDesc* createTensorDesc(const int32_t* shape, size_t shapeNum, OH_NN_DataType dataType, OH_NN_Format format)
{
    NN_TensorDesc* tensorDescTmp = OH_NNTensorDesc_Create();
    if (tensorDescTmp == nullptr) {
        LOGE("[NnrtOpsFuzzTest]OH_NNTensorDesc_Create failed!");
        return nullptr;
    }

    OH_NN_ReturnCode ret = OH_NNTensorDesc_SetDataType(tensorDescTmp, dataType);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NnrtOpsFuzzTest]OH_NNTensorDesc_SetDataType failed!ret = %d\n", ret);
        return nullptr;
    }

    if (shape != nullptr) {
        ret = OH_NNTensorDesc_SetShape(tensorDescTmp, shape, shapeNum);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NnrtOpsFuzzTest]OH_NNTensorDesc_SetShape failed!ret = %d\n", ret);
            return nullptr;
        }
    }

    ret = OH_NNTensorDesc_SetFormat(tensorDescTmp, format);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[NnrtOpsFuzzTest]OH_NNTensorDesc_SetShape failed!ret = %d\n", ret);
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
            LOGE("[NnrtOpsFuzzTest] OH_NNModel_AddOperation failed! ret=%{public}d\n", ret);
            return ret;
        }
    }

    if (graphArgs.specifyIO) {
        ret = OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NnrtOpsFuzzTest] OH_NNModel_SpecifyInputsAndOutputs failed! ret=%{public}d\n", ret);
            return ret;
        }
    }

    if (graphArgs.build) {
        ret = OH_NNModel_Finish(model);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NnrtOpsFuzzTest] OH_NNModel_Finish failed! ret=%d\n", ret);
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
            LOGE("[NnrtOpsFuzzTest] OH_NNModel_AddTensor failed! ret=%d\n", ret);
            return ret;
        }

        ret = OH_NNModel_SetTensorType(model, i, operandTem.type);
        if (ret != OH_NN_SUCCESS) {
            LOGE("[NnrtOpsFuzzTest] OH_NNBackend_SetModelTensorType failed! ret=%d\n", ret);
            return ret;
        }

        if (std::find(graphArgs.paramIndices.begin(), graphArgs.paramIndices.end(), i) !=
            graphArgs.paramIndices.end()) {
            ret = OH_NNModel_SetTensorData(model, i, operandTem.data, operandTem.length);
            if (ret != OH_NN_SUCCESS) {
                LOGE("[NnrtOpsFuzzTest] OH_NNModel_SetTensorData failed! ret=%{public}d\n", ret);
                return ret;
            }
        }
        OH_NNTensorDesc_Destroy(&tensorDesc);
    }
    ret = SingleModelBuildEndStep(model, graphArgs);
    return ret;
}

int buildModel0(uint32_t opsType)
{
    OH_NNModel *model = OH_NNModel_Construct();
    if (model == nullptr) {
        return -1;
    }

    Model0 model0;
    OHNNGraphArgs graphArgs = model0.graphArgs;
    graphArgs.operationType = static_cast<OH_NN_OperationType>(opsType);

    if (BuildSingleOpGraph(model, graphArgs) != OH_NN_SUCCESS) {
        OH_NNModel_Destroy(&model);
        return -1;
    }
    OH_NNModel_Destroy(&model);
    return 0;
}

bool HdiNnrtOpsFuzzTest(const uint8_t* data, size_t size)
{
    Data dataFuzz(data, size);
    uint32_t opsType = dataFuzz.GetData<uint32_t>()
        % (OH_NN_OPS_GATHER_ND - OH_NN_OPS_ADD + 1);
    buildModel0(opsType);
    return true;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS

/* Fuzzer entry point */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (data == nullptr) {
        LOGE("Pass data is nullptr.");
        return 0;
    }

    if (size < OHOS::NeuralNetworkRuntime::U32_AT_SIZE) {
        LOGE("Pass size is too small.");
        return 0;
    }

    OHOS::NeuralNetworkRuntime::HdiNnrtOpsFuzzTest(data, size);
    return 0;
}