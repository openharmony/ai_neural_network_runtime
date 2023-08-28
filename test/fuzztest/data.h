#ifndef TEST_FUZZTEST_GETDATA_H
#define TEST_FUZZTEST_GETDATA_H

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
#include <cstdint>

#include "../../common/log.h"
#include "securec.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
class Data {
public:
    Data(const uint8_t* data, size_t size)
    {
        dataFuzz = data;
        dataSize = size;
    }

    template<class T> T GetData()
    {
        T object {};
        size_t objectSize = sizeof(object);
        if (dataFuzz == nullptr || objectSize > dataSize - dataPos) {
            LOGE("[GetData]Data is not enough.");
            return {};
        }
        if (memcpy_s(&object, objectSize, dataFuzz + dataPos, objectSize) != EOK) {
            LOGE("[GetData]Memcpy_s failed.");
            return {};
        }
        dataPos = dataPos + objectSize;
        return object;
    }

    const uint8_t* GetNowData() const
    {
        return dataFuzz + dataPos;
    }

    size_t GetNowDataSize() const
    {
        return dataSize - dataPos;
    }
private:
    const uint8_t* dataFuzz {nullptr};
    size_t dataSize {0};
    size_t dataPos {0};
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif