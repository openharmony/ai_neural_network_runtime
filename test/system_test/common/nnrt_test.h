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

#ifndef NEURAL_NETWORK_RUNTIME_SYSTEM_TEST_NNRT_TEST
#define NEURAL_NETWORK_RUNTIME_SYSTEM_TEST_NNRT_TEST

#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

#include "interfaces/kits/c/neural_network_runtime.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace SystemTest {
struct CppQuantParam {
    std::vector<uint32_t> numBits;
    std::vector<double> scale;
    std::vector<int32_t> zeroPoint;
};

struct CppTensor {
    OH_NN_DataType dataType{OH_NN_UNKNOWN};
    std::vector<int32_t> dimensions;
    void* data{nullptr};
    size_t dataLength{0};
    CppQuantParam quantParam;
    OH_NN_TensorType type{OH_NN_TENSOR};
};

struct Node {
    OH_NN_OperationType opType;
    std::vector<uint32_t> inputs;
    std::vector<uint32_t> outputs;
    std::vector<uint32_t> params;
};

class NNRtTest : public testing::Test {
public:
    virtual OH_NN_ReturnCode AddTensors(const std::vector<CppTensor>& cppTensors);
    virtual OH_NN_ReturnCode AddOperation(OH_NN_OperationType opType,
                                          const std::vector<uint32_t>& paramIndices,
                                          const std::vector<uint32_t>& inputIndices,
                                          const std::vector<uint32_t>& outputIndices);
    virtual OH_NN_ReturnCode SpecifyInputAndOutput(const std::vector<uint32_t>& inputIndices,
                                                   const std::vector<uint32_t>& outputIndices);
    virtual OH_NN_ReturnCode SetInput(uint32_t index,
                                      const std::vector<int32_t>& dimensions,
                                      const void* buffer,
                                      size_t length);
    virtual OH_NN_ReturnCode SetOutput(uint32_t index, void* buffer, size_t length);
    virtual OH_NN_ReturnCode SetInputFromMemory(uint32_t index,
                                                const std::vector<int32_t>& dimensions,
                                                const void* buffer,
                                                size_t length,
                                                OH_NN_Memory** pMemory);
    virtual OH_NN_ReturnCode SetOutputFromMemory(uint32_t index, size_t length, OH_NN_Memory** pMemory);
    virtual OH_NN_ReturnCode GetDevices();

protected:
    OH_NNModel* m_model{nullptr};
    OH_NNCompilation* m_compilation{nullptr};
    OH_NNExecutor* m_executor{nullptr};

    std::vector<OH_NN_Tensor> m_tensors;
    std::vector<std::unique_ptr<OH_NN_QuantParam>> m_quantParams;
    std::vector<Node> m_nodes;
    std::vector<uint32_t> m_inputs;
    std::vector<uint32_t> m_outputs;
    std::vector<size_t> m_devices;
};
} // namespace SystemTest
} // NeuralNetworkRuntime
} // OHOS

#endif // NEURAL_NETWORK_RUNTIME_SYSTEM_TEST_NNRT_TEST