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

#include <string>
#include <utility>

#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#include "nnrt_delegate.h"
#include "../nnrt/nnrt_implementation.h"

namespace tflite {
namespace tools {
constexpr int32_t DEFAULT_THREADS = 1;
constexpr int32_t DEFAULT_DELEGATE_NUM = -1;
class NnrtDelegateProvider : public DelegateProvider {
public:
    NnrtDelegateProvider()
    {
        default_params_.AddParam("use_nnrt", ToolParam::Create<bool>(false));
        default_params_.AddParam("performance", ToolParam::Create<std::string>(""));
        default_params_.AddParam("priority", ToolParam::Create<std::string>(""));
        default_params_.AddParam("device", ToolParam::Create<std::string>(""));
        default_params_.AddParam("cache_dir", ToolParam::Create<std::string>(""));
        default_params_.AddParam("model_token", ToolParam::Create<std::string>(""));
        default_params_.AddParam("max_delegate_num", ToolParam::Create<int32_t>(DEFAULT_DELEGATE_NUM));
        default_params_.AddParam("enable_fp16", ToolParam::Create<bool>(false));
        default_params_.AddParam("allow_dynamic_dimensions", ToolParam::Create<bool>(false));
    }

    ~NnrtDelegateProvider() {};

    std::vector<Flag> CreateFlags(ToolParams* param) const final;

    void LogParams(const ToolParams& params, bool verbose) const final;

    TfLiteDelegatePtr CreateTfLiteDelegate(const ToolParams& params) const final;

    std::pair<TfLiteDelegatePtr, int32_t> CreateRankedTfLiteDelegate(const ToolParams& params) const final;

    std::string GetName() const final
    {
        return "NNRT";
    }
};

REGISTER_DELEGATE_PROVIDER(NnrtDelegateProvider);

std::vector<Flag> NnrtDelegateProvider::CreateFlags(ToolParams* params) const
{
    std::vector<Flag> flags = {
        CreateFlag<int32_t>("max_delegate_num", params, "Delegate max num limit, max_delegate_num <= 0 means no limit"),
        CreateFlag<bool>("enable_fp16", params, "Whether to Infer model with FP16."),
        CreateFlag<bool>("allow_dynamic_dimensions", params,
            "Whether to allow dynamic dimension sizes without re-compilation."),
        CreateFlag<std::string>("performance", params,
        "Execution performance for nnrt delegate. "
        "choose within [low, medium, high, extreme, default]."),
        CreateFlag<std::string>("priority", params,
        "The model execution priority in nnrt, and it "
        "choose within [default, low, medium, high]."),
        CreateFlag<std::string>("device", params,
        "The name of the nnrt accelerator to use, "
        "choose within [cpu, gpu, apu, nnrt-reference], "
        "nnrt-reference means chosen automatically by nnrt."),
        CreateFlag<std::string>("cache_dir", params, "The directory of load and save cache for delegate"),
        CreateFlag<std::string>("model_token", params, "The file_name of load and save cache for delegate"),
    };
    return flags;
}

void NnrtDelegateProvider::LogParams(const ToolParams& params, bool verbose) const
{
    LOG_TOOL_PARAM(params, bool, "use_nnrt", "Use NNRT", verbose);
    if (!params.Get<bool>("use_nnrt")) {
        return; // no use nnrt, return.
    }

    LOG_TOOL_PARAM(params, std::string, "performance", "NNRT execution performance", verbose);
    LOG_TOOL_PARAM(params, std::string, "priority", "NNRT execution priority", verbose);
    LOG_TOOL_PARAM(params, std::string, "device", "NNRT accelerator name", verbose);
    LOG_TOOL_PARAM(params, std::string, "cache_dir", "NNRT model cache directory", verbose);
    LOG_TOOL_PARAM(params, std::string, "model_token", "NNRT model cache filename", verbose);
    LOG_TOOL_PARAM(params, int32_t, "max_delegate_num", "NNRT delegate max partition", verbose);
    LOG_TOOL_PARAM(params, bool, "enable_fp16", "NNRT allow fp16 inference", verbose);
    LOG_TOOL_PARAM(params, bool, "allow_dynamic_dimensions", "NNRT allow dynamic dimensions", verbose);
}

TfLiteStatus GetExecutionPerformance(const ToolParams& params, NnrtDelegate::Options& options)
{
    std::string stringExecutionPerformance = params.Get<std::string>("performance");
    if (stringExecutionPerformance.empty()) {
        return kTfLiteOk; // no set performance
    }

    OH_NN_PerformanceMode executionPerformance = OH_NN_PERFORMANCE_NONE;
    if (stringExecutionPerformance == "low") {
        executionPerformance = OH_NN_PERFORMANCE_LOW;
    } else if (stringExecutionPerformance == "medium") {
        executionPerformance = OH_NN_PERFORMANCE_MEDIUM;
    } else if (stringExecutionPerformance == "high") {
        executionPerformance = OH_NN_PERFORMANCE_HIGH;
    } else if (stringExecutionPerformance == "extreme") {
        executionPerformance = OH_NN_PERFORMANCE_EXTREME;
    } else if (stringExecutionPerformance == "default") {
        executionPerformance = OH_NN_PERFORMANCE_NONE;
    } else {
        TFLITE_LOG(ERROR) << "The provided value is not a valid nnrt execution performance.";
        return kTfLiteError;
    }
    options.executionPerformance = executionPerformance;

    return kTfLiteOk;
}

TfLiteStatus GetExecutionPriority(const ToolParams& params, NnrtDelegate::Options& options)
{
    std::string stringExecutionPriority = params.Get<std::string>("priority");
    if (stringExecutionPriority.empty()) {
        return kTfLiteOk; // no set priority
    }

    OH_NN_Priority executionPriority = OH_NN_PRIORITY_MEDIUM;
    if (stringExecutionPriority == "low") {
        executionPriority = OH_NN_PRIORITY_LOW;
    } else if (stringExecutionPriority == "medium") {
        executionPriority = OH_NN_PRIORITY_MEDIUM;
    } else if (stringExecutionPriority == "high") {
        executionPriority = OH_NN_PRIORITY_HIGH;
    } else if (stringExecutionPriority == "default") {
        executionPriority = OH_NN_PRIORITY_MEDIUM;
    } else {
        TFLITE_LOG(ERROR) << "The provided value is not a valid nnrt execution priority.";
        return kTfLiteError;
    }
    options.executionPriority = executionPriority;

    return kTfLiteOk;
}

TfLiteStatus MapParams(const ToolParams& params, NnrtDelegate::Options& options)
{
    std::string acceleratorName = params.Get<std::string>("device");
    if (!acceleratorName.empty()) {
        options.acceleratorName = acceleratorName;
    }

    if (params.GetParam("max_delegate_num") != nullptr) {
        options.maxNumberDelegatedPartitions = params.Get<int32_t>("max_delegate_num");
    }

    std::string cacheDir = params.Get<std::string>("cache_dir");
    if (!cacheDir.empty()) {
        options.cacheDir = cacheDir;
    }

    std::string modelToken = params.Get<std::string>("model_token");
    if (!modelToken.empty()) {
        options.modelToken = modelToken;
    }

    if (params.Get<bool>("enable_fp16")) {
        options.enableFp16 = true;
    }

    if (params.Get<bool>("allow_dynamic_dimensions")) {
        options.allowDynamicDimensions = true;
    }

    return kTfLiteOk;
}

TfLiteDelegatePtr NnrtDelegateProvider::CreateTfLiteDelegate(const ToolParams& params) const
{
    TfLiteDelegatePtr delegate(nullptr, [](TfLiteDelegate*) {});
    if (!params.Get<bool>("use_nnrt")) {
        return delegate;
    }

    NnrtDelegate::Options options;
    TFLITE_TOOLS_CHECK(MapParams(params, options) == kTfLiteOk) << "Map params to NNRT Delegate options failed.";
    TFLITE_TOOLS_CHECK(GetExecutionPerformance(params, options) == kTfLiteOk) <<
        "Create TfLite NNRT Delegate failed.";
    TFLITE_TOOLS_CHECK(GetExecutionPriority(params, options) == kTfLiteOk) << "Create TfLite NNRT Delegate failed.";

    const auto* nnrtImpl = NnrtImplementation();
    if (!nnrtImpl->nnrtExists) {
        TFLITE_LOG(WARN) << "NNRT acceleration is unsupported on this platform.";
        return delegate;
    }

    return TfLiteDelegatePtr(new (std::nothrow) NnrtDelegate(nnrtImpl, options),
        [](TfLiteDelegate* delegate) { delete reinterpret_cast<NnrtDelegate*>(delegate); });
}

std::pair<TfLiteDelegatePtr, int32_t> NnrtDelegateProvider::CreateRankedTfLiteDelegate(const ToolParams& params) const
{
    auto ptr = CreateTfLiteDelegate(params);
    LogParams(params, false);
    return std::make_pair(std::move(ptr), params.GetPosition<bool>("use_nnrt"));
}
} // namespace tools
} // namespace tflite