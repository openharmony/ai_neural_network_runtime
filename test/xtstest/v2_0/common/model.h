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
#ifndef MODEL_H
#define MODEL_H

#include "const.h"
#include "nnrt_utils.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Test {

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
    // AVG POOL MODEL
    float inputValue[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    uint64_t kernelValue[2] = {2, 2};
    uint64_t strideValue[2] = {1, 1};
    int8_t padValue = 1;
    int8_t activationValue = OH_NN_FUSED_NONE;
    float outputValue[4] = {0};
    float expectValue[4] = {2, 3, 5, 6};

    OHNNOperandTest dynamicInput = {OH_NN_FLOAT32, OH_NN_TENSOR, {-1, -1, -1, -1}, inputValue, AVG_INPUT_LENGTH};
    OHNNOperandTest kernel = {OH_NN_INT64, OH_NN_AVG_POOL_KERNEL_SIZE, {2}, kernelValue, sizeof(kernelValue)};
    OHNNOperandTest strides = {OH_NN_INT64, OH_NN_AVG_POOL_STRIDE, {2}, strideValue, sizeof(strideValue)};
    OHNNOperandTest padMode = {OH_NN_INT8, OH_NN_AVG_POOL_PAD_MODE, {}, &padValue, sizeof(padValue)};
    OHNNOperandTest activation = {OH_NN_INT8, OH_NN_AVG_POOL_ACTIVATION_TYPE, {}, &activationValue, sizeof(int8_t)};
    OHNNOperandTest output = {OH_NN_FLOAT32, OH_NN_TENSOR, {-1, -1, -1, -1}, outputValue, sizeof(outputValue)};

    OHNNGraphArgs graphArgs = {.operationType = OH_NN_OPS_AVG_POOL,
                               .operands = {dynamicInput, kernel, strides, padMode, activation, output},
                               .paramIndices = {1, 2, 3, 4},
                               .inputIndices = {0},
                               .outputIndices = {5}};
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

} // namespace Test
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // MODEL_H