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

#ifndef NEURAL_NETWORK_RUNTIME_LSTM_BUILDER_H
#define NEURAL_NETWORK_RUNTIME_LSTM_BUILDER_H

#include "mindir.h"

#include "ops_builder.h"
#include "ops_registry.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Ops {
class LSTMBuilder : public OpsBuilder {
public:
    typedef OH_NN_ReturnCode(LSTMBuilder::*FuncPtr)(std::shared_ptr<NNTensor>);

    LSTMBuilder();
    ~LSTMBuilder() override;
    OH_NN_ReturnCode Build(const std::vector<uint32_t>& paramsIndex,
                           const std::vector<uint32_t>& inputsIndex,
                           const std::vector<uint32_t>& outputsIndex,
                           const std::vector<std::shared_ptr<NNTensor>>& allTensors) override;
    LiteGraphPrimitvePtr GetPrimitive() override;

private:
    OH_NN_ReturnCode SetBidirectional(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetHasBias(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetInputSize(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetHiddenSize(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetNumLayers(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetNumDirections(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetDropout(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetZoneoutCell(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetZoneoutHidden(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode SetProjSize(std::shared_ptr<NNTensor> tensor);
    OH_NN_ReturnCode ParseParam(const std::vector<uint32_t>& paramsIndex,
                                const std::vector<std::shared_ptr<NNTensor>>& allTensors);

private:
    bool m_bidirectional {false};
    bool m_hasBias {false};
    int64_t m_inputSize {0};
    int64_t m_hiddenSize {0};
    int64_t m_numLayers {0};
    int64_t m_numDirections {0};
    float m_dropout {0.0f};
    float m_zoneoutCell {0.0f};
    float m_zoneoutHidden {0.0f};
    int64_t m_projSize {0};
    std::unordered_map<OH_NN_TensorType, FuncPtr> m_paramMap = {
        {OH_NN_LSTM_BIDIRECTIONAL, &LSTMBuilder::SetBidirectional},
        {OH_NN_LSTM_HAS_BIAS, &LSTMBuilder::SetHasBias},
        {OH_NN_LSTM_INPUT_SIZE, &LSTMBuilder::SetInputSize},
        {OH_NN_LSTM_HIDDEN_SIZE, &LSTMBuilder::SetHiddenSize},
        {OH_NN_LSTM_NUM_LAYERS, &LSTMBuilder::SetNumLayers},
        {OH_NN_LSTM_NUM_DIRECTIONS, &LSTMBuilder::SetNumDirections},
        {OH_NN_LSTM_DROPOUT, &LSTMBuilder::SetDropout},
        {OH_NN_LSTM_ZONEOUT_CELL, &LSTMBuilder::SetZoneoutCell},
        {OH_NN_LSTM_ZONEOUT_HIDDEN, &LSTMBuilder::SetZoneoutHidden},
        {OH_NN_LSTM_PROJ_SIZE, &LSTMBuilder::SetProjSize}
    };
};
} // namespace Ops
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // NEURAL_NETWORK_RUNTIME_LSTM_BUILDER_H