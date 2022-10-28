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

#ifndef NEURAL_NETWORK_RUNTIME_SCOPED_TRACE_H
#define NEURAL_NETWORK_RUNTIME_SCOPED_TRACE_H

#include <string>
#include "hitrace/trace.h"

#define NNRT_TRACE_NAME(name) ScopedTrace ___tracer(name)
namespace OHOS {
namespace NeuralNetworkRuntime {
class ScopedTrace {
public:
    inline ScopedTrace(const std::string& name)
    {
        m_name = name;
        HiviewDFX::HiTraceId traceId = HiviewDFX::HiTraceChain::GetId();
        if (traceId.IsValid()) {
            HiviewDFX::HiTraceChain::Tracepoint(HITRACE_TP_GENERAL, traceId, "NNRt Trace start: %s", name.c_str());
        }
    }

    inline ~ScopedTrace()
    {
        HiviewDFX::HiTraceId traceId = HiviewDFX::HiTraceChain::GetId();
        if (traceId.IsValid()) {
            HiviewDFX::HiTraceChain::Tracepoint(HITRACE_TP_GENERAL, traceId, "NNRt Trace end: %s", m_name.c_str());
        }
    }

private:
    std::string m_name {};
};
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_SCOPED_TRACE_H
