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

#ifndef OHOS_HDI_NNR_NODE_FUNCTIONS_H
#define OHOS_HDI_NNR_NODE_FUNCTIONS_H

#include <functional>

#include "hdf_base.h"
#include "utils/hdf_log.h"

#include "node_registry.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
template<typename T>
int32_t ParsePrimitive(const std::vector<int8_t>& primitive, T& attr,
    std::function<bool(OHOS::MessageParcel&, T&)> parseFunc)
{
    if (primitive.empty()) {
        HDF_LOGE("Primitive data is empty.");
        return HDF_FAILURE;
    }

    OHOS::MessageParcel parcelData;
    bool ret = parcelData.WriteBuffer(primitive.data(), primitive.size());
    if (!ret) {
        HDF_LOGE("Write data to MessageParcel failed.");
        return HDF_FAILURE;
    }

    ret = parseFunc(parcelData, attr);
    if (!ret) {
        HDF_LOGE("Unmarshalling data failed.");
        return HDF_FAILURE;
    }
    return HDF_SUCCESS;
}

PrimUniquePtr GetAddPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetAvgPoolPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetConcatPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetConv2dPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetFullConnectionPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetMaxPoolFusionPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetMatMulFusionPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetSoftmaxPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetReshapePrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetScaleFusionPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetActivationPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetQuantDTypeCastPrimitive(const std::vector<int8_t>& primitive);
PrimUniquePtr GetMulFusionPrimitive(const std::vector<int8_t>& primitive);
} // namespace V1_0
} // namespace Nnrt
} // namespace HDI
} // namespace OHOS
#endif // OHOS_HDI_NNR_NODE_FUNCTIONS_H