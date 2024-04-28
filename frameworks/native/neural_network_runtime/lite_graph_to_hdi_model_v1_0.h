/*
 * Copyright (c) 2024 Huawei Device Co., Ltd.
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

#ifndef NEURAL_NETWORK_RUNTIME_LITEGRAPH_TO_HDIMODEL_V1_0_H
#define NEURAL_NETWORK_RUNTIME_LITEGRAPH_TO_HDIMODEL_V1_0_H

#include "mindir.h"
#include "nnrt/v1_0/model_types.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace V1 {
void HDIModel_Destroy(OHOS::HDI::Nnrt::V1_0::Model **model);
OHOS::HDI::Nnrt::V1_0::Model *LiteGraph_To_HDIModel(const mindspore::lite::LiteGraph *liteGraph,
    const OHOS::HDI::Nnrt::V1_0::SharedBuffer &buffer);
} // V1
} // NeuralNetworkRuntime
} // OHOS

#endif // NEURAL_NETWORK_RUNTIME_LITEGRAPH_TO_HDIMODEL_V1_0_H