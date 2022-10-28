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

#ifndef TENSORFLOW_LITE_DELEGATES_NNRT_DELEGATE_H
#define TENSORFLOW_LITE_DELEGATES_NNRT_DELEGATE_H

#include <string>
#include <vector>

#include "neural_network_runtime.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/delegates/serialization.h"

#include "../nnrt/nnrt_implementation.h"

namespace tflite {
namespace delegate {
namespace nnrt {
class NnrtDelegateKernel;
} // namespace nnrt
} // namespace delegate

using tflite::delegate::nnrt::NnrtDelegateKernel;

// TFliteDelegate to interface with NNRT.
class NnrtDelegate : public TfLiteDelegate {
public:
    struct Options {
        OH_NN_PerformanceMode executionPerformance = OH_NN_PERFORMANCE_NONE;
        std::string acceleratorName;
        std::string cacheDir;
        std::string modelToken;
        OH_NN_Priority executionPriority = OH_NN_PRIORITY_MEDIUM;
        int32_t maxNumberDelegatedPartitions = -1;
        uint64_t maxCompilationTimeoutDurationNs = 0;
        uint64_t maxExecutionTimeoutDurationNs = 0;
        uint64_t maxExecutionLoopTimeoutDurationNs = 0;
        // allow fp32 compuation to be run in fp16.
        bool enableFp16 = false;
        bool allowDynamicDimensions = false;

        uint32_t version {0};
    };

    // Uses default options.
    NnrtDelegate();

    // The ownership of the NNRT instance is left to the caller of the
    // NnrtDelegate constructor; the caller must ensure that the lifetime
    // of the NNRT instance exceeds the lifetime of the NnrtDelegate.
    explicit NnrtDelegate(const NnrtApi* nnrt);

    // The constructor that accepts options from user.
    // This makes a copy of any data that it needs from Options, so
    // the caller can safely deallocate any storage pointed to by
    // the 'const char *' members of Options immediately after calling this.
    explicit NnrtDelegate(const Options& options);

    // Constructor that accepts both an NNRT instance and options.
    // The ownership of the NNRT instance is left to the caller of the
    // NnrtDelegate constructor; the caller must ensure that the lifetime
    // of the NNRT instance exceeds the lifetime of the NnrtDelegate.
    // This constructor makes a copy of any data that it needs from Options, so
    // the caller can safely deallocate any storage pointed to by
    // the 'const char *' members of Options immediately after calling this.
    NnrtDelegate(const NnrtApi* nnrt, const Options& options);

    ~NnrtDelegate() = default;

    // Returns the delegate options.
    // The lifetime of the storage pointed to by the 'const char *' members of the
    // returned Options object is the same as the lifetime of the supplied
    // TfLiteDelegate instance.
    static TfLiteStatus GetOptions(const TfLiteDelegate* pDelegate, Options& options);

private:
    struct Data {
        const NnrtApi* nnrt = nullptr;

        // Preferred Power/perf trade-off.
        OH_NN_PerformanceMode executionPerformance = OH_NN_PERFORMANCE_NONE;

        // Selected NNRT accelerator name.
        std::string acceleratorName;

        // The cache dir for NNRT model.
        std::string cacheDir;

        // The unique token string for NNRT model.
        std::string modelToken;

        // Maximum number of NNRT partition to delegate. Zero or negative means
        // no limit.
        int32_t maxNumberDelegatedPartitions = -1;

        // Specifies the relative priority for executions of the model.
        OH_NN_Priority executionPriority = OH_NN_PRIORITY_MEDIUM;

        // Specifies the maximum expected duration in nanosecond for compiling the
        // model.
        uint64_t maxCompilationTimeoutDurationNs = 0;

        // Specifies the maximum expected duration in nanosecond for executing the
        // model.
        uint64_t maxExecutionTimeoutDurationNs = 0;

        // Specifies the maximum expected duration in nanosecond for WHILE loops in
        // the execution
        uint64_t maxExecutionLoopTimeoutDurationNs = 0;

        // allow fp32 compuation to be run in fp16.
        bool enableFp16 = false;

        // Whether to allow dynamic dimension sizes without re-compilation.
        bool allowDynamicDimensions = false;

        uint32_t version {0};

        explicit Data(const NnrtApi* nnrt);
        ~Data();
    };

    static TfLiteStatus DoPrepare(TfLiteContext* context, TfLiteDelegate* delegate);

    static TfLiteStatus DoCopyFromBufferHandle(TfLiteContext* context,
        TfLiteDelegate* delegate, TfLiteBufferHandle bufferHandle, TfLiteTensor* tensor);

    static TfLiteStatus DoCopyToBufferHandle(TfLiteContext* context,
        TfLiteDelegate* delegate, TfLiteBufferHandle bufferHandle, TfLiteTensor* tensor);

    static void DoFreeBufferHandle(TfLiteContext* context,
        TfLiteDelegate* delegate, TfLiteBufferHandle* handle);

    static TfLiteStatus LimitDelegatedPartitions(int32_t maxPartitions,
        std::vector<TfLiteDelegateParams> partitionParamsArray, std::vector<int32_t>& nodesToDelegate);

    static TfLiteStatus GetSupportedNodes(TfLiteContext* context,
        TfLiteDelegate* delegate, std::vector<int32_t>& supportedNodes);

    static void GetDelegateKernelRegistration(TfLiteDelegate* delegate, TfLiteRegistration& nnrtDelegateKernel);

    static TfLiteStatus CheckDeviceValid(TfLiteContext* context, TfLiteDelegate* delegate);

    void NnrtDelegateConstructorImpl(const Options& options);

private:
    // Delegate data presented through TfLiteDelegate::data_.
    Data m_delegateData;
};

TfLiteDelegate* NnrtDelegateSingleton();
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_NNRT_DELEGATE_H
