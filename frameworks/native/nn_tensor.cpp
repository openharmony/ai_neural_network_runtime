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

#include <algorithm>
#include <cstdlib>
#include <new>

#include "nn_tensor.h"
#include "validation.h"
#include "transform.h"
#include "common/log.h"
#include "mindir.h"
#include "mindir_types.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
const uint32_t SUPPORT_NUM_BIT = 8; // Currently support 8-bit quantization only
const uint32_t INVALID_NUM_BIT = 0;

void DestroyLiteGraphTensor(void* tensor)
{
    mindspore::lite::MindIR_Tensor_Destroy(&tensor);
}

NNTensor::~NNTensor()
{
    if (m_buffer != nullptr) {
        delete [] reinterpret_cast<char*>(m_buffer);
    }
}

NNTensor::NNTensor(NNTensor&& tensor) noexcept
{
    *this = std::move(tensor);
}

NNTensor& NNTensor::operator=(NNTensor&& tensor) noexcept
{
    if (this == &tensor) {
        return *this;
    }

    m_type = tensor.m_type;
    m_dataType = tensor.m_dataType;
    m_format = tensor.m_format;
    m_name = std::move(tensor.m_name);
    m_dimensions = std::move(tensor.m_dimensions);
    m_quantParams = std::move(tensor.m_quantParams);
    m_elementCount = tensor.m_elementCount;
    m_isDynamicShape = tensor.m_isDynamicShape;
    m_isOpParameter = tensor.m_isOpParameter;
    m_buffer = tensor.m_buffer;
    m_bufferLength = tensor.m_bufferLength;
    m_dataLength = tensor.m_dataLength;

    tensor.m_buffer = nullptr;
    tensor.m_bufferLength = 0;
    tensor.m_dataLength = 0;

    return *this;
}

OH_NN_ReturnCode NNTensor::Build(OH_NN_DataType dataType,
                                 const std::vector<int32_t>& dimensions,
                                 const std::vector<QuantParam>& quantParam,
                                 OH_NN_TensorType type)
{
    m_type = type;

    if (!Validation::ValidateTensorDataType(dataType)) {
        LOGE("Build failed, passed invalid data type.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_dataType = dataType;

    OH_NN_ReturnCode ret = ParseDimensions(dimensions);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Build failed, passed invalid dimensions.");
        return ret;
    }

    ret = ParseQuantParams(quantParam);
    if (ret != OH_NN_SUCCESS) {
        LOGE("Build failed, please check quantParam.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNTensor::BuildFromOHNNTensor(const OH_NN_Tensor& nnTensor)
{
    m_type = nnTensor.type;

    if (!Validation::ValidateTensorDataType(nnTensor.dataType)) {
        LOGE("BuildFromOHNNTensor failed, passed invalid data type: %d.", nnTensor.dataType);
        return OH_NN_INVALID_PARAMETER;
    }
    m_dataType = nnTensor.dataType;

    if (!Validation::ValidateTensorType(nnTensor.type)) {
        LOGE("BuildFromOHNNTensor failed, passed invalid nnTensor type: %d.", nnTensor.type);
        return OH_NN_INVALID_PARAMETER;
    }

    OH_NN_ReturnCode ret = ParseDimensions(nnTensor);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildFromOHNNTensor failed, passed invalid nnTensor dimensions.");
        return ret;
    }

    ret = ParseQuantParams(nnTensor.quantParam);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildFromOHNNTensor failed, please check quantParam in nnTensor.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNTensor::ParseDimensions(const std::vector<int32_t>& dimensions)
{
    // Temporary variable to check overflow.
    uint64_t absoluteDim {0};
    uint64_t elementCount {1};
    uint64_t dataLength {static_cast<uint64_t>(GetTypeSize(m_dataType))};
    m_isDynamicShape = false;
    for (int32_t dim : dimensions) {
        if (dim < -1 || dim == 0) {
            LOGE("ParseDimension failed, dimension of OH_NN_Tensor cannot be 0 or less than -1, receive %d.", dim);
            return OH_NN_INVALID_PARAMETER;
        }

        m_isDynamicShape = m_isDynamicShape || (dim == -1);
        absoluteDim = static_cast<uint64_t>(abs(dim));
        elementCount *= absoluteDim;
        dataLength *= absoluteDim;

        if (dataLength > UINT32_MAX) {
            LOGE("ParseDimension failed, expected data length of tensor exceed limit %u.", UINT32_MAX);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    if (m_isDynamicShape) {
        // If tensor has dynamic shape, m_elementCount and m_dataLength take 0.
        m_elementCount = 0;
        m_dataLength = 0;
    } else {
        m_elementCount = static_cast<uint32_t>(elementCount);
        m_dataLength = static_cast<size_t>(dataLength);
    }

    m_dimensions = std::move(dimensions);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNTensor::ParseDimensions(const OH_NN_Tensor& nnTensor)
{
    OH_NN_ReturnCode ret = Validation::ValidateArray(nnTensor.dimensions, nnTensor.dimensionCount);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildFromOHNNTensor failed, please check dimension and dimensionCount in NNTensor.");
        return ret;
    }
    std::vector<int32_t> dimensions = ConstructVectorFromArray(nnTensor.dimensions, nnTensor.dimensionCount);

    ret = ParseDimensions(dimensions);
    if (ret != OH_NN_SUCCESS) {
        LOGE("BuildFromOHNNTensor failed, passed invalid dimension info.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNTensor::ParseQuantParams(const OH_NN_QuantParam* quantParam)
{
    if (quantParam == nullptr) {
        return OH_NN_SUCCESS;
    }

    if ((quantParam->numBits == nullptr) || (quantParam->scale == nullptr) || (quantParam->zeroPoint == nullptr)) {
        LOGE("ParseQuantParams failed, scale or zeroPoint is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }

    std::vector<QuantParam> tmpQuantParam;
    uint32_t numBits{0};
    double scale{0.0};
    int32_t zeroPoint{0};
    for (uint32_t i = 0; i < quantParam->quantCount; i++) {
        numBits = quantParam->numBits[i];
        scale = quantParam->scale[i];
        zeroPoint = quantParam->zeroPoint[i];
        tmpQuantParam.emplace_back((QuantParam){numBits, scale, zeroPoint});
    }

    OH_NN_ReturnCode ret = ParseQuantParams(tmpQuantParam);
    if (ret != OH_NN_SUCCESS) {
        LOGE("ParseQuantParams failed, please numBits in NNTensor.");
        return ret;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode NNTensor::ParseQuantParams(const std::vector<QuantParam>& quantParams)
{
    for (const QuantParam& param : quantParams) {
        // Only support 8-bit quantization in NNR version 1.0
        if ((param.numBits != SUPPORT_NUM_BIT) || (param.numBits == INVALID_NUM_BIT)) {
            LOGE("ParseQuantParams failed, get invalid numBits %d.", param.numBits);
            return OH_NN_INVALID_PARAMETER;
        }
    }

    m_quantParams = quantParams;
    return OH_NN_SUCCESS;
}

void NNTensor::IdentifyOpParameter()
{
    m_isOpParameter = true;
}

void NNTensor::SetName(const std::string& name)
{
    m_name = name;
}

// Buffer set inside NNTensor will be released during deconstruction, make sure the buffer won't be released twice.
void NNTensor::SetBuffer(const void* buffer, size_t length)
{
    // copy pointer instead of memory copying
    m_buffer = const_cast<void*>(buffer);
    m_bufferLength = length;
}

OH_NN_ReturnCode NNTensor::SetDimensions(const std::vector<int32_t>& dimensions)
{
    size_t expectedDimensionCount = m_dimensions.size();
    size_t dimensionCount = dimensions.size();
    if (dimensionCount != expectedDimensionCount) {
        LOGE("Passed dimensions have different dimension counts from NNTensor, expected %zu, but passed %zu.",
             expectedDimensionCount, dimensionCount);
        return OH_NN_INVALID_PARAMETER;
    }

    auto ret = ParseDimensions(dimensions);
    if (ret != OH_NN_SUCCESS) {
        LOGE("SetDimemsions failed, passed invalid dimension info.");
        return ret;
    }

    m_dimensions = dimensions;
    return OH_NN_SUCCESS;
}

OH_NN_TensorType NNTensor::GetType() const
{
    return m_type;
}

std::string NNTensor::GetName() const
{
    return m_name;
}

void* NNTensor::GetBuffer() const
{
    return m_buffer;
}

size_t NNTensor::GetBufferLength() const
{
    return m_bufferLength;
}

size_t NNTensor::GetDataLength() const
{
    return m_dataLength;
}

OH_NN_DataType NNTensor::GetDataType() const
{
    return m_dataType;
}

uint32_t NNTensor::GetElementCount() const
{
    return m_elementCount;
}

std::vector<int32_t> NNTensor::GetDimensions() const
{
    return m_dimensions;
}

OH_NN_Format NNTensor::GetFormat() const
{
    return m_format;
}

std::vector<QuantParam> NNTensor::GetQuantParam() const
{
    return m_quantParams;
}

LiteGraphTensorPtr NNTensor::ConvertToLiteGraphTensor() const
{
    mindspore::lite::DataType dataType = NNToMS::TransformDataType(m_dataType);
    mindspore::lite::Format format = NNToMS::TransformFormat(m_format);
    const uint8_t* buffer = static_cast<const uint8_t*>(m_buffer);
    std::vector<uint8_t> data = ConstructVectorFromArray(buffer, m_dataLength);

    std::vector<mindspore::lite::QuantParam> quantParams;
    mindspore::lite::QuantParam msQuantParam;
    for (const QuantParam& param : m_quantParams) {
        msQuantParam = {param.zeroPoint, param.scale, param.numBits};
        quantParams.emplace_back(std::move(msQuantParam));
    }

    mindspore::lite::TensorPtr tensor = mindspore::lite::MindIR_Tensor_Create(
        m_name, dataType, m_dimensions, format, data, quantParams);
    if (tensor == nullptr) {
        LOGE("ConvertToLiteGraphTensor failed, please check attributes of NNTensor.");
        return {nullptr, DestroyLiteGraphTensor};
    }

    LiteGraphTensorPtr liteGraphTensor(tensor, DestroyLiteGraphTensor);
    return liteGraphTensor;
}

void NNTensor::ConvertToIOTensor(IOTensor& tensor) const
{
    tensor.dataType = m_dataType;
    tensor.format = m_format;
    tensor.dimensions = m_dimensions;
    tensor.data = const_cast<void*>(m_buffer);
    tensor.length = m_bufferLength;
}

bool NNTensor::IsDynamicShape() const
{
    return m_isDynamicShape;
}

bool NNTensor::IsQuantTensor() const
{
    return (m_quantParams.size() > 0);
}

bool NNTensor::IsScalar() const
{
    return (m_dimensions.empty());
}

bool NNTensor::IsOpParameter() const
{
    return m_isOpParameter;
}

bool NNTensor::CompareAttribute(const NNTensor& tensor) const
{
    if (m_dataType != tensor.GetDataType()) {
        LOGI("Tensors have different data type: %d and %d.", m_dataType, tensor.GetDataType());
        return false;
    }

    if (m_format != tensor.GetFormat()) {
        LOGI("Tensors have different format: %d and %d.", m_format, tensor.GetFormat());
        return false;
    }

    const std::vector<int32_t> dimensions = tensor.GetDimensions();
    if (m_dimensions.size() != dimensions.size()) {
        LOGI("Tensors have differents dimension counts: %zu and %zu.", m_dimensions.size(), dimensions.size());
        return false;
    }

    for (auto i = 0; i < dimensions.size(); i++) {
        if (m_dimensions[i] != -1 && m_dimensions[i] != dimensions[i]) {
            LOGI("Tensors have different dimension: dimension index: %u, dimension value: %d and %d.",
                 i, m_dimensions[i], dimensions[i]);
            return false;
        }
    }

    if (m_type != tensor.GetType()) {
        LOGI("Tensors have different type: %d and %d.", m_type, tensor.GetType());
        return false;
    }

    return true;
}
} // NeuralNetworkRuntime
} // OHOS