/*
 * Copyright (c) 2026 Huawei Device Co., Ltd.
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

#ifndef NEURAL_NETWORK_RUNTIME_PARSE_NNRT_CONFIG_INT_H
#define NEURAL_NETWORK_RUNTIME_PARSE_NNRT_CONFIG_INT_H

#include <charconv>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <system_error>

namespace OHOS {
namespace NeuralNetworkRuntime {
/*
 * Parse whole-token decimal integers from NNRT extension config text
 * (SetExtensionConfig callingPid / hiaiModelId char vectors).
 * Reject empty, overflow, leading/trailing junk, '+', hex, and floats.
 * Valid in-range values keep the same numeric result as atoi/strtoul on
 * digit-only input.
 */
inline bool ParseNnrtConfigInt32(std::string_view text, int32_t &out)
{
    if (text.empty()) {
        return false;
    }
    int32_t value = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        return false;
    }
    out = value;
    return true;
}

inline bool ParseNnrtConfigInt32(const char *text, int32_t &out)
{
    if (text == nullptr) {
        return false;
    }
    return ParseNnrtConfigInt32(std::string_view(text), out);
}

inline bool ParseNnrtConfigInt32(const std::string &text, int32_t &out)
{
    return ParseNnrtConfigInt32(std::string_view(text), out);
}

inline bool ParseNnrtConfigUint32(std::string_view text, uint32_t &out)
{
    if (text.empty()) {
        return false;
    }
    uint32_t value = 0;
    auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
        return false;
    }
    out = value;
    return true;
}

inline bool ParseNnrtConfigUint32(const char *text, uint32_t &out)
{
    if (text == nullptr) {
        return false;
    }
    return ParseNnrtConfigUint32(std::string_view(text), out);
}

inline bool ParseNnrtConfigUint32(const std::string &text, uint32_t &out)
{
    return ParseNnrtConfigUint32(std::string_view(text), out);
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_PARSE_NNRT_CONFIG_INT_H
