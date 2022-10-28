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

#include "ops_test.h"

using namespace OHOS::NeuralNetworkRuntime::Ops;
using namespace std;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
void OpsTest::SaveInputTensor(const std::vector<uint32_t>& inputsIndex, OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam)
{
    m_inputsIndex = inputsIndex;
    for (size_t i = 0; i < inputsIndex.size(); ++i) {
        std::shared_ptr<NNTensor> inputTensor;
        inputTensor = TransToNNTensor(dataType, dim, quantParam, OH_NN_TENSOR);
        m_allTensors.emplace_back(inputTensor);
    }
}

void OpsTest::SaveOutputTensor(const std::vector<uint32_t>& outputsIndex, OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam)
{
    m_outputsIndex = outputsIndex;
    for (size_t i = 0; i < outputsIndex.size(); ++i) {
        std::shared_ptr<NNTensor> outputTensor;
        outputTensor = TransToNNTensor(dataType, dim, quantParam, OH_NN_TENSOR);
        m_allTensors.emplace_back(outputTensor);
    }
}

void OpsTest::SetKernelSize(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t kernelsNum{2};
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* kernelSizeValue = new (std::nothrow) int64_t[kernelsNum]{1, 1};
    EXPECT_NE(nullptr, kernelSizeValue);
    tensor->SetBuffer(kernelSizeValue, sizeof(int64_t) * kernelsNum);
    m_allTensors.emplace_back(tensor);
}

void OpsTest::SetStride(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t strideNum{2};
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* strideValue = new (std::nothrow) int64_t[strideNum]{1, 1};
    EXPECT_NE(nullptr, strideValue);
    tensor->SetBuffer(strideValue, sizeof(int64_t) * strideNum);
    m_allTensors.emplace_back(tensor);
}

void OpsTest::SetActivation(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int8_t* activationValue = new (std::nothrow) int8_t(0);
    EXPECT_NE(nullptr, activationValue);
    tensor->SetBuffer(activationValue, sizeof(int8_t));
    m_allTensors.emplace_back(tensor);
}

void OpsTest::SetDilation(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    int32_t dilationNum = 2;
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* dilationValue = new (std::nothrow) int64_t[2]{1, 1};
    EXPECT_NE(nullptr, dilationValue);
    tensor->SetBuffer(dilationValue, dilationNum * sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
}

void OpsTest::SetGroup(OH_NN_DataType dataType,
    const std::vector<int32_t> &dim,  const OH_NN_QuantParam* quantParam, OH_NN_TensorType type)
{
    std::shared_ptr<NNTensor> tensor = TransToNNTensor(dataType, dim, quantParam, type);
    int64_t* groupValue = new (std::nothrow) int64_t(0);
    EXPECT_NE(nullptr, groupValue);
    tensor->SetBuffer(groupValue, sizeof(int64_t));
    m_allTensors.emplace_back(tensor);
}

} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
