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

#ifndef NEURAL_NETWORK_RUNTIME_TENSOR_DESC_H
#define NEURAL_NETWORK_RUNTIME_TENSOR_DESC_H

#include <string>
#include <vector>
#include "interfaces/kits/c/neural_network_runtime/neural_network_runtime_type.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class TensorDesc {
public:
    TensorDesc() = default;
    ~TensorDesc() = default;

    OH_NN_ReturnCode GetDataType(OH_NN_DataType* dataType) const;
    OH_NN_ReturnCode SetDataType(OH_NN_DataType dataType);

    OH_NN_ReturnCode GetFormat(OH_NN_Format* format) const;
    OH_NN_ReturnCode SetFormat(OH_NN_Format format);

    OH_NN_ReturnCode GetShape(int32_t** shape, size_t* shapeNum) const;
    OH_NN_ReturnCode SetShape(const int32_t* shape, size_t shapeNum);

    OH_NN_ReturnCode GetElementNum(size_t* elementNum) const;
    OH_NN_ReturnCode GetByteSize(size_t* byteSize) const;

    OH_NN_ReturnCode SetName(const char* name);
    OH_NN_ReturnCode GetName(const char** name) const;

private:
    OH_NN_DataType m_dataType {OH_NN_UNKNOWN};
    OH_NN_Format m_format {OH_NN_FORMAT_NONE};
    std::vector<int32_t> m_shape;
    std::string m_name;
};
}  // namespace NeuralNetworkRuntime
}  // namespace OHOS
#endif  // NEURAL_NETWORK_RUNTIME_TENSOR_DESC_H