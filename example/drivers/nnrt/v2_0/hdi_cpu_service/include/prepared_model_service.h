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

#ifndef OHOS_HDI_NNRT_V2_0_PREPAREDMODELSERVICE_H
#define OHOS_HDI_NNRT_V2_0_PREPAREDMODELSERVICE_H

#include "v2_0/iprepared_model.h"
#include "include/api/data_type.h"
#include "include/api/context.h"
#include "include/api/types.h"
#include "include/api/model.h"
#include "mindspore_schema/model_generated.h"
#include "ashmem.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V2_0 {
constexpr int DYNAMIC_SHAPE_FLAG = -1;
class PreparedModelService : public IPreparedModel {
public:
    PreparedModelService() = default;

    virtual ~PreparedModelService();

    explicit PreparedModelService(std::shared_ptr<mindspore::Context> context);

    NNRT_ReturnCode Compile(std::shared_ptr<mindspore::schema::MetaGraphT> graph);

    NNRT_ReturnCode Compile(const void* modelBuffer, size_t length);

    int32_t ExportModelCache(std::vector<SharedBuffer>& modelCache) override;

    int32_t Run(const std::vector<IOTensor>& inputs, const std::vector<IOTensor>& outputs,
        std::vector<std::vector<int32_t>>& outputsDims) override;

    int32_t GetInputDimRanges(std::vector<std::vector<uint32_t>>& minInputDims,
        std::vector<std::vector<uint32_t>>& maxInputDims) override;

private:
    NNRT_ReturnCode SetInputs(const std::vector<IOTensor>& inputs);
    NNRT_ReturnCode SetOutputs(const std::vector<IOTensor>& outputs);
    NNRT_ReturnCode GetMSInputsAndOutputs();
    NNRT_ReturnCode CompareTensor(const IOTensor& tensor, const mindspore::MSTensor& msTensor);
    sptr<Ashmem> ParseBuffer(const SharedBuffer& buffer);
    NNRT_ReturnCode UpdateOutput(const std::vector<IOTensor>& outputs,
        std::vector<std::vector<int32_t>>& outputsDims, bool& isOutputBufferEnough);
    void ResetInputAndOutput();

private:
    std::shared_ptr<mindspore::schema::MetaGraphT> m_graph {nullptr};
    std::shared_ptr<mindspore::Context> m_context {nullptr};
    flatbuffers::FlatBufferBuilder m_builder;
    std::shared_ptr<mindspore::Model> m_model {nullptr};
    sptr<Ashmem> m_cacheBuffer {nullptr};
    std::vector<sptr<Ashmem>> m_inputAshmems;
    std::vector<mindspore::MSTensor> m_inputs;
    std::vector<sptr<Ashmem>> m_outputAshmems;
    std::vector<mindspore::MSTensor> m_outputs;
    std::vector<std::vector<int64_t>> m_inputDims;
    bool m_isDynamicShape {false};
};
} // V2_0
} // Nnrt
} // HDI
} // OHOS

#endif // OHOS_HDI_NNR_V2_0_PREPAREDMODELSERVICE_H