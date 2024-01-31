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


#ifndef NEURAL_NETWORK_RUNTIME_HDI_PREPARED_MODEL_V2_1_H
#define NEURAL_NETWORK_RUNTIME_HDI_PREPARED_MODEL_V2_1_H

#include <vector>

#include <v2_1/innrt_device.h>
#include <v2_1/iprepared_model.h>
#include <v2_1/nnrt_types.h>

#include "cpp_type.h"
#include "hdi_prepared_model_v2_0.h"
#include "refbase.h"

namespace V2_1 = OHOS::HDI::Nnrt::V2_1;

namespace OHOS {
namespace NeuralNetworkRuntime {
class HDIPreparedModelV2_1 : public HDIPreparedModelV2_0 {
public:
    explicit HDIPreparedModelV2_1(OHOS::sptr<V2_1::IPreparedModel> hdiPreparedModel);

private:
    OHOS::sptr<V2_1::IPreparedModel> m_hdiPreparedModel {nullptr};
};
} // namespace NeuralNetworkRuntime
} // OHOS
#endif // NEURAL_NETWORK_RUNTIME_HDI_PREPARED_MODEL_V2_1_H
