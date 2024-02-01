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


#ifndef NEURAL_NETWORK_RUNTIME_HDI_RETURNCODE_UTILS_H
#define NEURAL_NETWORK_RUNTIME_HDI_RETURNCODE_UTILS_H

#include <cstring>
#include <unordered_map>
#include <v2_1/nnrt_types.h>
#include "common/log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
inline std::string ConverterRetToString(OHOS::HDI::Nnrt::V2_1::NNRT_ReturnCode returnCode)
{
    static std::unordered_map<OHOS::HDI::Nnrt::V2_1::NNRT_ReturnCode, std::string> nnrtRet2StringMap {
        {V2_1::NNRT_ReturnCode::NNRT_SUCCESS, "NNRT_SUCCESS"},
        {V2_1::NNRT_ReturnCode::NNRT_FAILED, "NNRT_FAILED"},
        {V2_1::NNRT_ReturnCode::NNRT_NULL_PTR, "NNRT_NULL_PTR"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_PARAMETER, "NNRT_INVALID_PARAMETER"},
        {V2_1::NNRT_ReturnCode::NNRT_MEMORY_ERROR, "NNRT_MEMORY_ERROR"},
        {V2_1::NNRT_ReturnCode::NNRT_OUT_OF_MEMORY, "NNRT_OUT_OF_MEMORY"},
        {V2_1::NNRT_ReturnCode::NNRT_OPERATION_FORBIDDEN, "NNRT_OPERATION_FORBIDDEN"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_FILE, "NNRT_INVALID_FILE"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_PATH, "NNRT_INVALID_PATH"},
        {V2_1::NNRT_ReturnCode::NNRT_INSUFFICIENT_BUFFER, "NNRT_INSUFFICIENT_BUFFER"},
        {V2_1::NNRT_ReturnCode::NNRT_NO_CHANGE, "NNRT_NO_CHANGE"},
        {V2_1::NNRT_ReturnCode::NNRT_NOT_SUPPORT, "NNRT_NOT_SUPPORT"},
        {V2_1::NNRT_ReturnCode::NNRT_SERVICE_ERROR, "NNRT_SERVICE_ERROR"},
        {V2_1::NNRT_ReturnCode::NNRT_DEVICE_ERROR, "NNRT_DEVICE_ERROR"},
        {V2_1::NNRT_ReturnCode::NNRT_DEVICE_BUSY, "NNRT_DEVICE_BUSY"},
        {V2_1::NNRT_ReturnCode::NNRT_CANCELLED, "NNRT_CANCELLED"},
        {V2_1::NNRT_ReturnCode::NNRT_PERMISSION_DENIED, "NNRT_PERMISSION_DENIED"},
        {V2_1::NNRT_ReturnCode::NNRT_TIME_OUT, "NNRT_TIME_OUT"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_TENSOR, "NNRT_INVALID_TENSOR"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_NODE, "NNRT_INVALID_NODE"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_INPUT, "NNRT_INVALID_INPUT"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_OUTPUT, "NNRT_INVALID_OUTPUT"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_DATATYPE, "NNRT_INVALID_DATATYPE"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_FORMAT, "NNRT_INVALID_FORMAT"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_TENSOR_NAME, "NNRT_INVALID_TENSOR_NAME"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_SHAPE, "NNRT_INVALID_SHAPE"},
        {V2_1::NNRT_ReturnCode::NNRT_OUT_OF_DIMENTION_RANGES, "NNRT_OUT_OF_DIMENTION_RANGES"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_BUFFER, "NNRT_INVALID_BUFFER"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_BUFFER_SIZE, "NNRT_INVALID_BUFFER_SIZE"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_PERFORMANCE_MODE, "NNRT_INVALID_PERFORMANCE_MODE"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_PRIORITY, "NNRT_INVALID_PRIORITY"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_MODEL, "NNRT_INVALID_MODEL"},
        {V2_1::NNRT_ReturnCode::NNRT_INVALID_MODEL_CACHE, "NNRT_INVALID_MODEL_CACHE"},
        {V2_1::NNRT_ReturnCode::NNRT_UNSUPPORTED_OP, "NNRT_UNSUPPORTED_OP"}
    };

    if (nnrtRet2StringMap.find(returnCode) == nnrtRet2StringMap.end()) {
        return "ConverterRetToString failed, returnCode is invalid.";
    }

    return nnrtRet2StringMap.at(returnCode);
}

template<typename T>
T CheckReturnCode_V2_1(int32_t ret, T funcRet, const std::string& errorInfo)
{
    int32_t success = static_cast<int32_t>(V2_1::NNRT_ReturnCode::NNRT_SUCCESS);
    if (ret < success) {
        LOGE("%{public}s. An error occurred in HDI, errorcode is %{public}d.", errorInfo.c_str(), ret);
    } else if (ret > success) {
        OHOS::HDI::Nnrt::V2_1::NNRT_ReturnCode nnrtRet = static_cast<OHOS::HDI::Nnrt::V2_1::NNRT_ReturnCode>(ret);
        LOGE("%{public}s. Errorcode is %{public}s.", errorInfo.c_str(), ConverterRetToString(nnrtRet).c_str());
    }

    return funcRet;
}
} // namespace NeuralNetworkRuntime
} // OHOS
#endif // NEURAL_NETWORK_RUNTIME_HDI_RETURNCODE_UTILS_H