/*
 * Copyright (C) 2023 Huawei Device Co., Ltd.
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
#include "hdinnrtpreparedmodel_fuzzer.h"
#include "../data.h"
#include "../nnrt_prepare_model_impl.h"
#include "../../../common/log.h"

#include <v2_0/prepared_model_stub.h>
#include "message_parcel.h"
#include "message_option.h"
#include "securec.h"

namespace V2_0 = OHOS::HDI::Nnrt::V2_0;

namespace OHOS {
namespace NeuralNetworkRuntime {
constexpr size_t U32_AT_SIZE = 4;
bool HdiNnrtPreparedModelFuzzTest(const uint8_t* data, size_t size)
{
    OHOS::sptr<V2_0::IPreparedModel> preparedModel = new V2_0::NnrtPrepareModelImpl();
    if (preparedModel == nullptr) {
        LOGE("[HdiNnrtPreparedModelFuzzTest]Prepare model make failed.");
        return false;
    }

    Data dataFuzz(data, size);
    uint32_t code = dataFuzz.GetData<uint32_t>()
        % (V2_0::CMD_PREPARED_MODEL_RUN - V2_0::CMD_PREPARED_MODEL_GET_VERSION + 1);
    MessageParcel datas;
    datas.WriteInterfaceToken(V2_0::IPreparedModel::GetDescriptor());
    datas.WriteBuffer(dataFuzz.GetNowData(), dataFuzz.GetNowDataSize());
    datas.RewindRead(0);
    MessageParcel reply;
    MessageOption option;
    OHOS::sptr<V2_0::PreparedModelStub> preparedModelStub = new V2_0::PreparedModelStub(preparedModel);
    if (preparedModelStub == nullptr) {
        LOGE("[HdiNnrtPreparedModelFuzzTest]Nnrt preparemodel stub make failed.");
        return false;
    }
    preparedModelStub->OnRemoteRequest(code, datas, reply, option);
    return true;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS

/* Fuzzer entry point */
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size)
{
    if (data == nullptr) {
        LOGE("Pass data is nullptr.");
        return 0;
    }

    if (size < OHOS::NeuralNetworkRuntime::U32_AT_SIZE) {
        LOGE("Pass size is too small.");
        return 0;
    }

    OHOS::NeuralNetworkRuntime::HdiNnrtPreparedModelFuzzTest(data, size);
    return 0;
}