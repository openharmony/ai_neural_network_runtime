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

#ifndef NEURAL_NETWORK_RUNTIME_NN_TENSOR_H
#define NEURAL_NETWORK_RUNTIME_NN_TENSOR_H

#include <string>
#include <vector>

#include "cpp_type.h"
#include "interfaces/kits/c/neural_network_runtime.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
using LiteGraphTensorPtr = std::unique_ptr<void, void(*)(void*)>;

void DestroyLiteGraphTensor(void* tensor);

class NNTensor {
public:
    NNTensor() = default;
    ~NNTensor();
    NNTensor(NNTensor&& tensor) noexcept;
    NNTensor& operator=(NNTensor&& tensor) noexcept;
    // Copy construction and assignment is not allowed in case of double-free of m_buffer
    NNTensor(const NNTensor& tensor) = delete;
    NNTensor& operator=(const NNTensor& tensor) = delete;

    OH_NN_ReturnCode BuildFromOHNNTensor(const OH_NN_Tensor& nnTensor);
    OH_NN_ReturnCode Build(OH_NN_DataType dataType,
                           const std::vector<int32_t>& dimensions,
                           const std::vector<QuantParam>& quantParam,
                           OH_NN_TensorType type);
    void IdentifyOpParameter();

    void SetName(const std::string& name);
    void SetBuffer(const void* buffer, size_t length);
    OH_NN_ReturnCode SetDimensions(const std::vector<int32_t>& dimensions);

    std::string GetName() const;
    OH_NN_TensorType GetType() const;
    void* GetBuffer() const;
    // Return complete buffer length
    size_t GetBufferLength() const;
    // Return actual data length, since the data can be store in a larger buffer
    size_t GetDataLength() const;
    OH_NN_DataType GetDataType() const;
    uint32_t GetElementCount() const;
    std::vector<int32_t> GetDimensions() const;
    OH_NN_Format GetFormat() const;
    std::vector<QuantParam> GetQuantParam() const;
    LiteGraphTensorPtr ConvertToLiteGraphTensor() const;
    void ConvertToIOTensor(IOTensor& tensor) const;

    bool IsDynamicShape() const;
    bool IsQuantTensor() const;
    bool IsScalar() const;
    bool IsOpParameter() const;
    bool CompareAttribute(const NNTensor& tensor) const;

private:
    // Used in BuildFromOHNNTensor()
    OH_NN_ReturnCode ParseQuantParams(const OH_NN_QuantParam* quantParams);
    OH_NN_ReturnCode ParseDimensions(const OH_NN_Tensor& nnTensor);
    // Used in Build()
    OH_NN_ReturnCode ParseQuantParams(const std::vector<QuantParam>& quantParams);
    OH_NN_ReturnCode ParseDimensions(const std::vector<int32_t>& dimensions);

private:
    OH_NN_TensorType m_type {OH_NN_TENSOR};
    OH_NN_DataType m_dataType {OH_NN_FLOAT32};
    OH_NN_Format m_format {OH_NN_FORMAT_NHWC};
    std::string m_name;
    std::vector<int32_t> m_dimensions;
    std::vector<QuantParam> m_quantParams;
    uint32_t m_elementCount {0};
    bool m_isDynamicShape {false};
    bool m_isOpParameter {false};
    void* m_buffer {nullptr};
    size_t m_bufferLength {0};
    size_t m_dataLength {0};
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif // NEURAL_NETWORK_RUNTIME_NN_TENSOR_H