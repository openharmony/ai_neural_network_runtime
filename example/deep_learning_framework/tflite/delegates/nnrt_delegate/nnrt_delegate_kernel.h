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

#ifndef TENSORFLOW_LITE_DELEGATES_NNRT_DELEGATE_KERNEL_H
#define TENSORFLOW_LITE_DELEGATES_NNRT_DELEGATE_KERNEL_H


#include "neural_network_runtime.h"
#include "tensorflow/lite/c/common.h"

#include "tensor_mapping.h"
#include "nnrt_op_builder.h"

namespace tflite {
namespace delegate {
namespace nnrt {

// Represents a subgraph in TFLite that will be delegated to NNRt.
// It is abstracted as a single kernel node in the main TFLite graph and
// implements Init/Prepare/Invoke as TFLite kernel nodes.
class NnrtDelegateKernel {
public:
    explicit NnrtDelegateKernel(const NnrtApi* nnrt)
        : m_initialised(false),
          m_compiled(false),
          m_nnrt(nnrt),
          m_nnModel(nullptr),
          m_pNnCompilation(nullptr) {}

    NnrtDelegateKernel() : NnrtDelegateKernel(NnrtImplementation()) {}
    virtual ~NnrtDelegateKernel()
    {
        m_nnrt->OH_NNModel_Destroy(&m_nnModel);
        m_nnrt->OH_NNCompilation_Destroy(&m_pNnCompilation);
        m_nnrt = nullptr;
    }

    // Returns true if the node can be accelerated with NNRT.
    static bool Validate(const int32_t builtinCode);

    // Initialize the kernel (a NN model) and builds the NN Model.
    TfLiteStatus Init(TfLiteContext* context, const TfLiteDelegateParams* params);

    // Creates the NNRT Compilation for the NN model. It assumes that Init has
    // been called and completed successfully.
    TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node);

    // Invoke the NN Model. Expects Init and Prepare to have been completed successfully.
    TfLiteStatus Invoke(TfLiteContext* context, TfLiteNode* node);

private:
    TfLiteStatus Map(int32_t builtinCode, const NnrtOpMappingArgs& mappingArgs, int32_t& nnOpType) const;
    TfLiteStatus AddOpsAndTensors(TfLiteContext* context, const TfLiteIntArray* inputTensors,
        const NnrtDelegate::Options& delegateOptions);
    TfLiteStatus BuildGraph(TfLiteContext* context, const NnrtDelegate::Options& options,
        const TfLiteIntArray* inputTensors, const TfLiteIntArray* outputTensors);
    TfLiteStatus ConvertTensorTypeToNn(TfLiteContext* context, const std::pair<int32_t, int32_t>& indexPair,
        OH_NN_QuantParam* nnQuantParam, OH_NN_Tensor& nnTensor);
    TfLiteStatus SetInputTensors(TfLiteContext* context, TfLiteNode* node, OH_NNExecutor* pNnExecution,
        OH_NN_Tensor& nnTensor);
    TfLiteStatus SetOutputTensors(TfLiteContext* context, TfLiteNode* node, OH_NNExecutor* pNnExecution);
    TfLiteStatus SetNnOptions(TfLiteContext* context, const NnrtDelegate::Options& delegateOptions);

private:
    // True if initialization has been completed successfully
    bool m_initialised;

    // True if compilation has been completed successfully
    bool m_compiled;

    // NN device handle.
    size_t m_nnrtDevice;

    // Access to NNRT.
    const NnrtApi* m_nnrt;

    // NN API state.
    OH_NNModel* m_nnModel;
    OH_NNCompilation* m_pNnCompilation;

    // Node indices that this delegate is responsible for. Indices here
    // indexes into the nodes array in the TfLiteContext.
    std::vector<int32_t> m_delegateNodes;

    // Track indices we use
    TensorMapping m_tensorMapping;
};
} // namespace nnrt
} // namespace delegate
} // namespace tflite

#endif // TENSORFLOW_LITE_DELEGATES_NNRT_DELEGATE_KERNEL_H
