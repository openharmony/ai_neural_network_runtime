/*
 * Copyright (c) 2023 Huawei Device Co., Ltd.
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

#include "tensor_desc.h"
#include "validation.h"
#include "common/log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
const uint32_t BIT8_TO_BYTE = 1;
const uint32_t BIT16_TO_BYTE = 2;
const uint32_t BIT32_TO_BYTE = 4;
const uint32_t BIT64_TO_BYTE = 8;

uint32_t GetTypeSize(OH_NN_DataType type)
{
    switch (type) {
        case OH_NN_BOOL:
            return sizeof(bool);
        case OH_NN_INT8:
        case OH_NN_UINT8:
            return BIT8_TO_BYTE;
        case OH_NN_INT16:
        case OH_NN_UINT16:
        case OH_NN_FLOAT16:
            return BIT16_TO_BYTE;
        case OH_NN_INT32:
        case OH_NN_UINT32:
        case OH_NN_FLOAT32:
            return BIT32_TO_BYTE;
        case OH_NN_INT64:
        case OH_NN_UINT64:
        case OH_NN_FLOAT64:
            return BIT64_TO_BYTE;
        default:
            return 0;
    }
}

OH_NN_ReturnCode TensorDesc::GetDataType(OH_NN_DataType* dataType) const
{
    if (dataType == nullptr) {
        LOGE("GetDataType failed, dataType is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    *dataType = m_dataType;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode TensorDesc::SetDataType(OH_NN_DataType dataType)
{
    if (!Validation::ValidateTensorDataType(dataType)) {
        LOGE("TensorDesc::SetDataType failed, dataType %{public}d is invalid.", static_cast<int>(dataType));
        return OH_NN_INVALID_PARAMETER;
    }
    m_dataType = dataType;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode TensorDesc::GetFormat(OH_NN_Format* format) const
{
    if (format == nullptr) {
        LOGE("GetFormat failed, format is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    *format = m_format;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode TensorDesc::SetFormat(OH_NN_Format format)
{
    if (!Validation::ValidateTensorFormat(format)) {
        LOGE("TensorDesc::SetFormat failed, format %{public}d is invalid.", static_cast<int>(format));
        return OH_NN_INVALID_PARAMETER;
    }
    m_format = format;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode TensorDesc::GetShape(int32_t** shape, size_t* shapeNum) const
{
    if (shape == nullptr) {
        LOGE("GetShape failed, shape is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (*shape != nullptr) {
        LOGE("GetShape failed, *shape is not nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shapeNum == nullptr) {
        LOGE("GetShape failed, shapeNum is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    *shape = const_cast<int32_t*>(m_shape.data());
    *shapeNum = m_shape.size();
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode TensorDesc::SetShape(const int32_t* shape, size_t shapeNum)
{
    if (shape == nullptr) {
        LOGE("SetShape failed, shape is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (shapeNum == 0) {
        LOGE("SetShape failed, shapeNum is 0.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_shape.clear();
    for (size_t i = 0; i < shapeNum; ++i) {
        m_shape.emplace_back(shape[i]);
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode TensorDesc::GetElementNum(size_t* elementNum) const
{
    if (elementNum == nullptr) {
        LOGE("GetElementNum failed, elementNum is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (m_shape.empty()) {
        LOGE("GetElementNum failed, shape is empty.");
        return OH_NN_INVALID_PARAMETER;
    }
    *elementNum = 1;
    size_t shapeNum = m_shape.size();
    for (size_t i = 0; i < shapeNum; ++i) {
        if (m_shape[i] <= 0) {
            LOGW("GetElementNum return 0 with dynamic shape, shape[%{public}zu] is %{public}d.", i, m_shape[i]);
            *elementNum = 0;
            return OH_NN_DYNAMIC_SHAPE;
        }
        (*elementNum) *= m_shape[i];
    }
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode TensorDesc::GetByteSize(size_t* byteSize) const
{
    if (byteSize == nullptr) {
        LOGE("GetByteSize failed, byteSize is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    *byteSize = 0;
    size_t elementNum = 0;
    auto ret = GetElementNum(&elementNum);
    if (ret == OH_NN_DYNAMIC_SHAPE) {
        return OH_NN_SUCCESS;
    } else if (ret != OH_NN_SUCCESS) {
        LOGE("GetByteSize failed, get element num failed.");
        return ret;
    }

    uint32_t typeSize = GetTypeSize(m_dataType);
    if (typeSize == 0) {
        LOGE("GetByteSize failed, data type is invalid.");
        return OH_NN_INVALID_PARAMETER;
    }

    *byteSize = elementNum * typeSize;

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode TensorDesc::SetName(const char* name)
{
    if (name == nullptr) {
        LOGE("SetName failed, name is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    m_name = name;
    return OH_NN_SUCCESS;
}

// *name will be invalid after TensorDesc is destroyed
OH_NN_ReturnCode TensorDesc::GetName(const char** name) const
{
    if (name == nullptr) {
        LOGE("GetName failed, name is nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    if (*name != nullptr) {
        LOGE("GetName failed, *name is not nullptr.");
        return OH_NN_INVALID_PARAMETER;
    }
    *name = m_name.c_str();
    return OH_NN_SUCCESS;
}
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS