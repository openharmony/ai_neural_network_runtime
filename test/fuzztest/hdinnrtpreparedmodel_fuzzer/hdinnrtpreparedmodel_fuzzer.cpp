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
#include "common/log.h"
#include "frameworks/native/inner_model.h"
#include "interfaces/kits/c/neural_network_runtime.h"

#include <v2_0/nnrt_types.h>
#include <v2_0/innrt_device.h>
#include <v2_0/prepared_model_stub.h>
#include "message_parcel.h"
#include "message_option.h"
#include "securec.h"

namespace V2_0 = OHOS::HDI::Nnrt::V2_0;

namespace OHOS {
namespace NeuralNetworkRuntime {
constexpr size_t U32_AT_SIZE = 4;
bool BuildModel(OH_NNModel* model)
{
    // 指定Add算子的输入、参数和输出索引
    uint32_t inputIndicesValues[2] = {0, 1};
    uint32_t paramIndicesValues = 2;
    uint32_t outputIndicesValues = 3;
    OH_NN_UInt32Array paramIndices = {&paramIndicesValues, 1};
    OH_NN_UInt32Array inputIndices = {inputIndicesValues, 2};
    OH_NN_UInt32Array outputIndices = {&outputIndicesValues, 1};

    // 向模型实例添加Add算子
    OH_NN_ReturnCode ret = OH_NNModel_AddOperation(model, OH_NN_OPS_ADD, &paramIndices, &inputIndices, &outputIndices);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BuildModel]Add operation failed.");
        return false;
    }

    // 设置模型实例的输入、输出索引
    ret = OH_NNModel_SpecifyInputsAndOutputs(model, &inputIndices, &outputIndices);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BuildModel]Specify inputs and outputs failed.");
        return false;
    }

    // 完成模型实例的构建
    ret = OH_NNModel_Finish(model);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BuildModel]Error happened when finishing model construction.");
        return false;
    }
    return true;
}

bool BuildAddGraph(OH_NNModel **pModel)
{
    // 创建模型实例，进行模型构造
    OH_NNModel* model = OH_NNModel_Construct();
    if (model == nullptr) {
        LOGE("[BuildAddGraph]Create model failed.");
        return false;
    }

    // 添加Add算子的第一个输入Tensor，类型为float32，张量形状为[1, 2, 2, 3]
    int32_t inputDims[4] = {1, 2, 2, 3};
    OH_NN_Tensor input1 = {OH_NN_FLOAT32, 4, inputDims, nullptr, OH_NN_TENSOR};
    OH_NN_ReturnCode ret = OH_NNModel_AddTensor(model, &input1);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BuildAddGraph]Add Tensor of first input failed.");
        return false;
    }

    // 添加Add算子的第二个输入Tensor，类型为float32，张量形状为[1, 2, 2, 3]
    OH_NN_Tensor input2 = {OH_NN_FLOAT32, 4, inputDims, nullptr, OH_NN_TENSOR};
    ret = OH_NNModel_AddTensor(model, &input2);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BuildAddGraph]Add Tensor of second input failed.");
        return false;
    }

    // 添加Add算子的参数Tensor，该参数Tensor用于指定激活函数的类型，Tensor的数据类型为int8。
    int32_t activationDims = 1;
    int8_t activationValue = OH_NN_FUSED_NONE;
    OH_NN_Tensor activation = {OH_NN_INT8, 1, &activationDims, nullptr, OH_NN_ADD_ACTIVATIONTYPE};
    ret = OH_NNModel_AddTensor(model, &activation);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BuildAddGraph]Add Tensor of activation failed.");
        return false;
    }

    int opCnt = 2;
    // 将激活函数类型设置为OH_NN_FUSED_NONE，表示该算子不添加激活函数。
    ret = OH_NNModel_SetTensorData(model, opCnt, &activationValue, sizeof(int8_t));
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BuildAddGraph]Set value of activation failed.");
        return false;
    }

    // 设置Add算子的输出，类型为float32，张量形状为[1, 2, 2, 3]
    OH_NN_Tensor output = {OH_NN_FLOAT32, 4, inputDims, nullptr, OH_NN_TENSOR};
    ret = OH_NNModel_AddTensor(model, &output);
    if (ret != OH_NN_SUCCESS) {
        LOGE("[BuildAddGraph]Add Tensor of output failed.");
        return false;
    }

    bool isSuccess = BuildModel(model);
    if (!isSuccess) {
        LOGE("[BuildAddGraph]Compilation model failed.");
        return false;
    }

    *pModel = model;
    return true;
}

bool ConvertModel(OHOS::sptr<V2_0::INnrtDevice> device_, OH_NNModel *model,
                  V2_0::SharedBuffer &tensorBuffer, V2_0::Model **iModel)
{
    auto *innerModel = reinterpret_cast<InnerModel *>(model);
    std::shared_ptr<mindspore::lite::LiteGraph> m_liteGraph = innerModel->GetLiteGraphs();
    if (m_liteGraph == nullptr) {
        LOGE("[ConvertModel]Model is nullptr, cannot query supported operation.");
        return false;
    }

    size_t tensorSize = mindspore::lite::MindIR_LiteGraph_GetConstTensorSize(m_liteGraph.get());
    int32_t hdiRet {0};
    if (tensorSize > 0) {
        hdiRet = device_->AllocateBuffer(tensorSize, tensorBuffer);
        int nnrt_fd = -1;
        if (hdiRet != HDF_SUCCESS || tensorBuffer.fd == nnrt_fd) {
            LOGE("[ConvertModel]Allocate tensor buffer failed");
            return false;
        }
    }
    *iModel = mindspore::lite::MindIR_LiteGraph_To_Model(m_liteGraph.get(), tensorBuffer);
    if (iModel == nullptr) {
        LOGE("[ConvertModel]Parse litegraph to hdi model failed.");
        device_->ReleaseBuffer(tensorBuffer);
        return false;
    }
    // release model
    OH_NNModel_Destroy(&model);
    model = nullptr;
    return true;
}

bool CreatePreparedModel(OHOS::sptr<V2_0::IPreparedModel>& iPreparedModel)
{
    OHOS::sptr<V2_0::INnrtDevice> device = V2_0::INnrtDevice::Get();
    if (device == nullptr) {
        LOGE("[CreatePreparedModel]Nnrt device get failed.");
        return false;
    }
    OH_NNModel *model = nullptr;
    bool isSuccess = false;
    isSuccess = BuildAddGraph(&model);
    if (!isSuccess) {
        LOGE("[CreatePreparedModel]Build add graph failed.");
        return false;
    }

    V2_0::Model *iModel = nullptr;
    int nnrt_fd = -1;
    V2_0::SharedBuffer tensorBuffer {nnrt_fd, 0, 0, 0};
    isSuccess = ConvertModel(device, model, tensorBuffer, &iModel);
    if (!isSuccess) {
        LOGE("[CreatePreparedModel]ConvertModel failed.");
        return false;
    }
    V2_0::ModelConfig config;
    config.enableFloat16 = false;
    config.mode = V2_0::PERFORMANCE_EXTREME;
    config.priority = V2_0::PRIORITY_HIGH;
    // prepared model
    int32_t nnrtDeviceRet = device->PrepareModel(*iModel, config, iPreparedModel);
    if (nnrtDeviceRet != HDF_SUCCESS) {
        LOGE("[CreatePreparedModel]Prepare model failed.");
        return false;
    }
    return true;
}

bool HdiNnrtPreparedModelFuzzTest(const uint8_t* data, size_t size)
{
    OHOS::sptr<V2_0::IPreparedModel> preparedModel;
    bool isSuccess = CreatePreparedModel(preparedModel);
    if (!isSuccess) {
        LOGE("[HdiNnrtPreparedModelFuzzTest]Create prepare model failed.");
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
    std::shared_ptr<V2_0::PreparedModelStub> preparedModelStub =
        std::make_shared<V2_0::PreparedModelStub>(preparedModel);
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