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

#include "nnrt_delegate.h"

#include "tensorflow/lite/util.h"
#include "tensorflow/lite/context_util.h"
#include "tensorflow/lite/minimal_logging.h"

#include "nnrt_utils.h"
#include "nnrt_delegate_kernel.h"

namespace tflite {
const char* g_tfliteNnrtDelegateName = "TfLiteNnrtDelegate";
constexpr int32_t TFLITE_NNRT_DELEGATE_VERSION = 1;

NnrtDelegate::Data::Data(const NnrtApi* nnrt) : nnrt(nnrt) {}

NnrtDelegate::Data::~Data() {}

void NnrtDelegate::NnrtDelegateConstructorImpl(const Options& options)
{
    m_delegateData.acceleratorName = options.acceleratorName;
    m_delegateData.cacheDir = options.cacheDir;
    m_delegateData.modelToken = options.modelToken;
    m_delegateData.enableFp16 = options.enableFp16;
    m_delegateData.executionPriority = options.executionPriority;
    m_delegateData.executionPerformance = options.executionPerformance;
    m_delegateData.allowDynamicDimensions = options.allowDynamicDimensions;
    m_delegateData.maxNumberDelegatedPartitions = options.maxNumberDelegatedPartitions;
    m_delegateData.maxCompilationTimeoutDurationNs = options.maxCompilationTimeoutDurationNs;
    m_delegateData.maxExecutionTimeoutDurationNs = options.maxExecutionTimeoutDurationNs;
    m_delegateData.maxExecutionLoopTimeoutDurationNs = options.maxExecutionLoopTimeoutDurationNs;

    Prepare = DoPrepare;
    CopyFromBufferHandle = DoCopyFromBufferHandle;
    CopyToBufferHandle = DoCopyToBufferHandle;
    FreeBufferHandle = DoFreeBufferHandle;
    data_ = &m_delegateData;

    // NNRT support dynamic shape feature.
    flags |= kTfLiteDelegateFlagsAllowDynamicTensors;
    flags |= kTfLiteDelegateFlagsRequirePropagatedShapes;
}

NnrtDelegate::NnrtDelegate(const NnrtApi* nnrt) : NnrtDelegate(nnrt, Options()) {}

NnrtDelegate::NnrtDelegate(const Options& options) : NnrtDelegate(NnrtImplementation(), options) {}

NnrtDelegate::NnrtDelegate(const NnrtApi* nnrt, const Options& options)
    : TfLiteDelegate(TfLiteDelegateCreate()), m_delegateData(nnrt)
{
    NnrtDelegateConstructorImpl(options);
}

NnrtDelegate::NnrtDelegate() : NnrtDelegate(Options()) {}

TfLiteStatus NnrtDelegate::GetOptions(const TfLiteDelegate* pDelegate, Options& options)
{
    // Caller guarantees that parameters are legal
    auto pDelegateData = static_cast<Data*>(pDelegate->data_);
    options.acceleratorName = pDelegateData->acceleratorName;
    options.cacheDir = pDelegateData->cacheDir;
    options.modelToken = pDelegateData->modelToken;
    options.enableFp16 = pDelegateData->enableFp16;
    options.executionPriority = pDelegateData->executionPriority;
    options.executionPerformance = pDelegateData->executionPerformance;
    options.allowDynamicDimensions = pDelegateData->allowDynamicDimensions;
    options.maxNumberDelegatedPartitions = pDelegateData->maxNumberDelegatedPartitions;
    options.maxCompilationTimeoutDurationNs = pDelegateData->maxCompilationTimeoutDurationNs;
    options.maxExecutionTimeoutDurationNs = pDelegateData->maxExecutionTimeoutDurationNs;
    options.maxExecutionLoopTimeoutDurationNs = pDelegateData->maxExecutionLoopTimeoutDurationNs;
    options.version = pDelegateData->version;

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegate::DoCopyFromBufferHandle(TfLiteContext* context,
    TfLiteDelegate* delegate, TfLiteBufferHandle bufferHandle, TfLiteTensor* tensor)
{
    return kTfLiteError;
}

TfLiteStatus NnrtDelegate::DoCopyToBufferHandle(TfLiteContext* context,
    TfLiteDelegate* delegate, TfLiteBufferHandle bufferHandle, TfLiteTensor* tensor)
{
    return kTfLiteError;
}

void NnrtDelegate::DoFreeBufferHandle(TfLiteContext* context,
    TfLiteDelegate* delegate, TfLiteBufferHandle* handle)
{
    return;
}

TfLiteStatus NnrtDelegate::LimitDelegatedPartitions(int32_t maxPartitions,
    std::vector<TfLiteDelegateParams> partitionParamsArray, std::vector<int32_t>& nodesToDelegate)
{
    int32_t numPartitions = partitionParamsArray.size();
    if ((maxPartitions <= 0) || (numPartitions <= maxPartitions)) { // no limit or not exceed limit
        return kTfLiteOk;
    }

    int32_t numberDelegatedPartitions = std::count_if(
        partitionParamsArray.begin(), partitionParamsArray.end(),
        [nodesToDelegate](const TfLiteDelegateParams& partitionParams) {
            return std::find(nodesToDelegate.begin(), nodesToDelegate.end(),
                partitionParams.nodes_to_replace->data[0]) != nodesToDelegate.end();
        });
    // Adapt maxPartitions to limit delegate paritions, sort and abandon the low-ranking nodes.
    if (numberDelegatedPartitions > maxPartitions) {
        std::sort(partitionParamsArray.begin(), partitionParamsArray.end(),
            [](const TfLiteDelegateParams& left, const TfLiteDelegateParams& right) -> bool {
                return left.nodes_to_replace->size > right.nodes_to_replace->size;
            });

        nodesToDelegate.clear();

        for (int32_t i = 0; i < maxPartitions; ++i) {
            const TfLiteDelegateParams& partitionParams = partitionParamsArray[i];
            nodesToDelegate.insert(nodesToDelegate.end(),
                                   partitionParams.nodes_to_replace->data,
                                   partitionParams.nodes_to_replace->data +
                                   partitionParams.nodes_to_replace->size);
        }
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegate::GetSupportedNodes(TfLiteContext* context,
    TfLiteDelegate* delegate, std::vector<int32_t>& supportedNodes)
{
    // Caller guarantees that parameters are legal
    TfLiteIntArray* executionPlan = nullptr;
    TF_LITE_ENSURE_STATUS(context->GetExecutionPlan(context, &executionPlan));
    TF_LITE_ENSURE_EQ(context, executionPlan != nullptr, true);

    // Check for every node if it is supported
    TfLiteNode* node = nullptr;
    TfLiteRegistration* registration = nullptr;
    for (auto nodeIndex : TfLiteIntArrayView(executionPlan)) {
        node = nullptr;
        registration = nullptr;
        TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(context, nodeIndex, &node, &registration));
        if (NnrtDelegateKernel::Validate(registration->builtin_code)) {
            supportedNodes.emplace_back(nodeIndex);
        } else {
            TFLITE_LOG_PROD(TFLITE_LOG_WARNING,
                "[NNRT-DELEGATE] Get unsupportted node: %d.", registration->builtin_code);
        }
    }

    return kTfLiteOk;
}

void NnrtDelegate::GetDelegateKernelRegistration(TfLiteDelegate* delegate, TfLiteRegistration& nnrtDelegateKernel)
{
    // Caller guarantees that parameters are legal
    nnrtDelegateKernel.profiling_string = nullptr;
    nnrtDelegateKernel.builtin_code = kTfLiteBuiltinDelegate;
    nnrtDelegateKernel.custom_name = g_tfliteNnrtDelegateName;
    nnrtDelegateKernel.version = TFLITE_NNRT_DELEGATE_VERSION;

    nnrtDelegateKernel.init = [](TfLiteContext* context, const char* buffer, size_t length) -> void* {
        if (buffer == nullptr) {
            return nullptr;
        }

        const TfLiteDelegateParams* params = reinterpret_cast<const TfLiteDelegateParams*>(buffer);
        auto* delegateData = static_cast<Data*>(params->delegate->data_);
        NnrtDelegateKernel* state = new (std::nothrow) NnrtDelegateKernel(delegateData->nnrt);
        if (state == nullptr) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to create NnrtDelegateKernel instance.");
            return state;
        }

        TfLiteStatus status = state->Init(context, params);
        if (status != kTfLiteOk) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to init NnrtDelegateKernel.");
            delete state;
            state = nullptr;
        }
        return state;
    };

    nnrtDelegateKernel.free = [](TfLiteContext* context, void* buffer) -> void {
        if (buffer != nullptr) {
            delete static_cast<NnrtDelegateKernel*>(buffer);
            buffer = nullptr;
        }
    };

    nnrtDelegateKernel.prepare = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        if (node == nullptr) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to prepare delegate kernels, the node is nullptr.");
            return kTfLiteError;
        }

        NnrtDelegateKernel* state = reinterpret_cast<NnrtDelegateKernel*>(node->user_data);
        return state->Prepare(context, node);
    };

    nnrtDelegateKernel.invoke = [](TfLiteContext* context, TfLiteNode* node) -> TfLiteStatus {
        if (node == nullptr) {
            TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to invoke delegate kernels, the node is nullptr.");
            return kTfLiteError;
        }

        NnrtDelegateKernel* state = reinterpret_cast<NnrtDelegateKernel*>(node->user_data);
        return state->Invoke(context, node);
    };
}

TfLiteStatus NnrtDelegate::CheckDeviceValid(TfLiteContext* context, TfLiteDelegate* delegate)
{
    // Caller guarantees that parameters are legal
    auto* delegateData = static_cast<Data*>(delegate->data_);
    if (delegateData == nullptr) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Delegate data not be found.");
        return kTfLiteDelegateDataNotFound;
    }

    const NnrtApi* nnrt = delegateData->nnrt;
    if (nnrt == nullptr) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "Failed to get nnrt instance.");
        return kTfLiteError;
    }

    NnrtDelegate::Options delegateOptions;
    TF_LITE_ENSURE_STATUS(NnrtDelegate::GetOptions(delegate, delegateOptions));

    if (tflite::IsUseTargetDevice(delegateOptions)) {
        size_t nnrtDevice;
        TF_LITE_ENSURE_STATUS(GetTargetDevice(context, delegate, nnrt, nnrtDevice));
    }

    return kTfLiteOk;
}

TfLiteStatus NnrtDelegate::DoPrepare(TfLiteContext* context, TfLiteDelegate* delegate)
{
    if ((context == nullptr) || (delegate == nullptr)) {
        TFLITE_LOG_PROD(TFLITE_LOG_ERROR, "[NNRT-DELEGATE] Input TFLite-Context or TFLite-Delegate is nullptr.");
        return kTfLiteError;
    }

    auto* delegateData = static_cast<Data*>(delegate->data_);
    const NnrtApi* nnrt = delegateData->nnrt;

    // Do not delegate nodes_ if NN API is unavailable.
    if (!nnrt->nnrtExists) {
        return kTfLiteOk;
    }

    // Check devices validity
    TF_LITE_ENSURE_STATUS(CheckDeviceValid(context, delegate));

    // Get supportted nodes by tflite.
    // We don't care about all nodes_, we only care about ones in the current plan.
    std::vector<int32_t> supportedNodes;
    GetSupportedNodes(context, delegate, supportedNodes);

    // If there are no delegated nodes, short-circuit node replacement.
    if (supportedNodes.empty()) {
        TFLITE_LOG_PROD(TFLITE_LOG_INFO, "[NNRT-DELEGATE] supportted node list is empty.");
        return kTfLiteOk;
    }

    static TfLiteRegistration nnrtDelegateKernel;
    GetDelegateKernelRegistration(delegate, nnrtDelegateKernel);

    std::vector<int32_t> nodesToDelegate(supportedNodes);
    int32_t numPartitions;
    TfLiteDelegateParams* paramsArray = nullptr;
    auto supportedNodesArray = BuildTfLiteIntArray(supportedNodes);
    TF_LITE_ENSURE_STATUS(context->PreviewDelegatePartitioning(
        context, supportedNodesArray.get(), &paramsArray, &numPartitions));

    NnrtDelegate::Options delegateOptions;
    TF_LITE_ENSURE_STATUS(NnrtDelegate::GetOptions(delegate, delegateOptions));
    const auto partitionParamsArray = std::vector<TfLiteDelegateParams>(paramsArray, paramsArray + numPartitions);
    TF_LITE_ENSURE_STATUS(LimitDelegatedPartitions(
        delegateOptions.maxNumberDelegatedPartitions, partitionParamsArray, nodesToDelegate));

    auto nodesToDelegateArray = BuildTfLiteIntArray(nodesToDelegate);
    if (nodesToDelegateArray->size == 0) {
        TFLITE_LOG_PROD(TFLITE_LOG_INFO, "[NNRT-DELEGATE] No node to delegate.");
        return kTfLiteOk;
    } else {
        // Request TFLite to partition the graph and make kernels
        // for each independent node sub set a new nnrtDelegateKernel.
        return context->ReplaceNodeSubsetsWithDelegateKernels(context,
            nnrtDelegateKernel, nodesToDelegateArray.get(), delegate);
    }
}

// Return a singleton NNRT Delegate that can check ops supported.
TfLiteDelegate* NnrtDelegateSingleton()
{
    static NnrtDelegate delegate;
    return &delegate;
}
} // namespace tflite