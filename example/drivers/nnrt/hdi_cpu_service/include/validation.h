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

#ifndef OHOS_HDI_NNRT_VALIDATION_H
#define OHOS_HDI_NNRT_VALIDATION_H

#include "v1_0/nnrt_types.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V1_0 {
int32_t ValidatePerformanceMode(PerformanceMode mode);
int32_t ValidatePriority(Priority priority);
int32_t ValidateDataType(DataType dataType);
int32_t ValidateFormat(Format format);
} // namespace V1_0
} // namespace Nnrt
} // namespace HDI
} // namespace OHOS
#endif // OHOS_HDI_NNRT_VALIDATION_H