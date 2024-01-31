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

#include "hdi_prepared_model_v2_1.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace {
HDIPreparedModelV2_1::HDIPreparedModelV2_1(OHOS::sptr<V2_1::IPreparedModel> hdiPreparedModel)
    : m_hdiPreparedModel(hdiPreparedModel)
{
    hdiPreparedModel->GetVersion(m_hdiVersion.first, m_hdiVersion.second);
}
}
} // namespace NeuralNetworkRuntime
} // OHOS
