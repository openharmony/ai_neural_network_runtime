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

#include "neural_network_runtime_test.h"

#include "mindir.h"

#include "common/utils.h"
#include "frameworks/native/compilation.h"
#include "frameworks/native/device_manager.h"
#include "frameworks/native/hdi_device.h"
#include "test/unittest/common/mock_idevice.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
OH_NN_ReturnCode HDIDevice::PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                         const ModelConfig& config,
                                         std::shared_ptr<PreparedModel>& preparedModel)
{
    if (model == nullptr) {
        return OH_NN_INVALID_PARAMETER;
    }

    if (config.enableFloat16 == false) {
        return OH_NN_FAILED;
    }

    sptr<OHOS::HDI::Nnrt::V1_0::IPreparedModel> iPreparedModel = sptr<OHOS::HDI::Nnrt::V1_0
        ::MockIPreparedModel>(new OHOS::HDI::Nnrt::V1_0::MockIPreparedModel());
    if (iPreparedModel == nullptr) {
        LOGE("HDIDevice mock PrepareModel failed, error happened when new sptr");
        return OH_NN_NULL_PTR;
    }

    preparedModel = CreateSharedPtr<HDIPreparedModel>(iPreparedModel);
    return OH_NN_SUCCESS;
}

std::shared_ptr<Device> DeviceManager::GetDevice(size_t deviceId) const
{
    sptr<OHOS::HDI::Nnrt::V1_0::INnrtDevice> idevice
        = sptr<OHOS::HDI::Nnrt::V1_0::MockIDevice>(new (std::nothrow) OHOS::HDI::Nnrt::V1_0::MockIDevice());
    if (idevice == nullptr) {
        LOGE("DeviceManager mock GetDevice failed, error happened when new sptr");
        return nullptr;
    }

    std::shared_ptr<Device> device = CreateSharedPtr<HDIDevice>(idevice);
    if (device == nullptr) {
        LOGE("DeviceManager mock GetDevice failed, the device is nullptr");
        return nullptr;
    }

    if (deviceId == 0) {
        LOGE("DeviceManager mock GetDevice failed, the passed parameter deviceId is 0");
        return nullptr;
    } else {
        return device;
    }
}

OH_NN_ReturnCode HDIDevice::GetDeviceType(OH_NN_DeviceType& deviceType)
{
    if (deviceType == OH_NN_OTHERS) {
        return OH_NN_UNAVALIDABLE_DEVICE;
    }

    return OH_NN_SUCCESS;
}

const std::string& DeviceManager::GetDeviceName(size_t deviceId)
{
    static std::string deviceName = "";
    if (deviceId == 0) {
        return deviceName;
    }

    deviceName = "deviceId";
    return deviceName;
}

const std::vector<size_t>& DeviceManager::GetAllDeviceId()
{
    static std::vector<size_t> deviceIds;
    if (OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode == OH_NN_FAILED) {
        // In order not to affect other use cases, set to the OH_NN_OPERATION_FORBIDDEN
        OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_OPERATION_FORBIDDEN;
        return deviceIds;
    }
    std::size_t device = 1;
    deviceIds.emplace_back(device);
    return deviceIds;
}

OH_NN_ReturnCode HDIDevice::IsModelCacheSupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsPerformanceModeSupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsPrioritySupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsFloat16PrecisionSupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
                                                  std::vector<bool>& ops)
{
    if (model == nullptr) {
        LOGE("HDIDevice mock GetSupportedOperation failed, Model is nullptr, cannot query supported operation.");
        return OH_NN_NULL_PTR;
    }

    ops.emplace_back(true);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDevice::IsDynamicInputSupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Unittest {
OH_NN_ReturnCode NeuralNetworkRuntimeTest::BuildModelGraph(InnerModel& innerModel)
{
    // liteGraph is released internally by innerModel
    mindspore::lite::LiteGraph* liteGraph = new (std::nothrow) mindspore::lite::LiteGraph;
    EXPECT_NE(nullptr, liteGraph);

    liteGraph->name_ = "testGraph";
    liteGraph->input_indices_ = {0};
    liteGraph->output_indices_ = {1};
    liteGraph->all_tensors_ = {nullptr};
    const std::vector<uint8_t> data(36, 1);
    const std::vector<int32_t> dim = {3, 3};
    const std::vector<mindspore::lite::QuantParam> quant_params {};

    for (size_t indexInput = 0; indexInput < liteGraph->input_indices_.size(); ++indexInput) {
        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create(liteGraph->name_,
            mindspore::lite::DATA_TYPE_FLOAT32, dim, mindspore::lite::FORMAT_NCHW, data, quant_params));
    }

    for (size_t indexOutput = 0; indexOutput < liteGraph->output_indices_.size(); ++indexOutput) {
        liteGraph->all_tensors_.emplace_back(mindspore::lite::MindIR_Tensor_Create(liteGraph->name_,
            mindspore::lite::DATA_TYPE_FLOAT32, dim, mindspore::lite::FORMAT_NCHW, data, quant_params));
    }

    return innerModel.BuildFromLiteGraph(liteGraph);
}

void NeuralNetworkRuntimeTest::InitIndices()
{
    m_inputIndices.data = m_inputIndexs;
    m_inputIndices.size = sizeof(m_inputIndexs) / sizeof(uint32_t);

    m_outputIndices.data = m_outputIndexs;
    m_outputIndices.size = sizeof(m_outputIndexs) / sizeof(uint32_t);

    m_paramIndices.data = m_paramIndexs;
    m_paramIndices.size = sizeof(m_paramIndexs) / sizeof(uint32_t);
}

void NeuralNetworkRuntimeTest::AddModelTensor(InnerModel& innerModel)
{
    const int dim[2] = {2, 2};
    const OH_NN_Tensor& tensor = {OH_NN_FLOAT32, 2, dim, nullptr, OH_NN_TENSOR};

    EXPECT_EQ(OH_NN_SUCCESS, innerModel.AddTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.AddTensor(tensor));
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.AddTensor(tensor));

    const OH_NN_Tensor& tensorParam = {OH_NN_INT8, 0, nullptr, nullptr, OH_NN_ADD_ACTIVATIONTYPE};
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.AddTensor(tensorParam));
}

void NeuralNetworkRuntimeTest::SetTensor()
{
    m_tensor.dataType = OH_NN_INT32;
    m_tensor.dimensionCount = 0;
    m_tensor.dimensions = nullptr;
    m_tensor.quantParam = nullptr;
    m_tensor.type = OH_NN_TENSOR;
}

void NeuralNetworkRuntimeTest::SetInnerBuild(InnerModel& innerModel)
{
    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SetTensorValue(index,
        static_cast<const void *>(&activation), sizeof(int8_t)));

    OH_NN_OperationType opType{OH_NN_OPS_ADD};
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.AddOperation(opType, m_paramIndices, m_inputIndices, m_outputIndices));
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SpecifyInputsAndOutputs(m_inputIndices, m_outputIndices));
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.Build());
}

void NeuralNetworkRuntimeTest::SetInputAndOutput(Executor& executor)
{
    size_t length = 9 * sizeof(int32_t);
    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void *buffer = input;
    uint32_t index = 0;

    SetTensor();
    EXPECT_EQ(OH_NN_SUCCESS, executor.SetInput(index, m_tensor, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executor.SetOutput(index, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executor.Run());
}

/*
 * @tc.name: model_construct_001
 * @tc.desc: Verify the return model of the OH_NNModel_Construct function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_construct_001, testing::ext::TestSize.Level0)
{
    OH_NNModel* ret = OH_NNModel_Construct();
    EXPECT_NE(nullptr, ret);
}

/*
 * @tc.name: model_add_tensor_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNModel_Tensor function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_add_tensor_001, testing::ext::TestSize.Level0)
{
    OH_NNModel* model = nullptr;
    const int32_t dimInput[2] = {2, 2};
    const OH_NN_Tensor tensor = {OH_NN_INT8, 2, dimInput, nullptr, OH_NN_TENSOR};
    OH_NN_ReturnCode ret = OH_NNModel_AddTensor(model, &tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_add_tensor_002
 * @tc.desc: Verify the OH_NN_Tensor is nullptr of the OH_NNModel_AddTensor function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_add_tensor_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NN_Tensor* tensor = nullptr;
    OH_NN_ReturnCode ret = OH_NNModel_AddTensor(model, tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_add_tensor_003
 * @tc.desc: Verify the success of the OH_NNModel_AddTensor function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_add_tensor_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    const int32_t dimInput[2] = {2, 2};
    const OH_NN_Tensor tensor = {OH_NN_INT8, 2, dimInput, nullptr, OH_NN_TENSOR};
    OH_NN_ReturnCode ret = OH_NNModel_AddTensor(model, &tensor);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: model_add_operation_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNModel_AddOperation function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_add_operation_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = nullptr;
    OH_NN_OperationType opType{OH_NN_OPS_ADD};

    InitIndices();
    AddModelTensor(innerModel);

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SetTensorValue(index,
        static_cast<const void *>(&activation), sizeof(int8_t)));

    OH_NN_ReturnCode ret = OH_NNModel_AddOperation(model, opType, &m_paramIndices, &m_inputIndices, &m_outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_add_operation_002
 * @tc.desc: Verify the paramIndices is nullptr of the OH_NNModel_AddOperation function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_add_operation_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NN_OperationType opType{OH_NN_OPS_ADD};

    m_inputIndices.data = m_inputIndexs;
    m_inputIndices.size = sizeof(m_inputIndexs) / sizeof(uint32_t);

    m_outputIndices.data = m_outputIndexs;
    m_outputIndices.size = sizeof(m_outputIndexs) / sizeof(uint32_t);

    AddModelTensor(innerModel);
    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SetTensorValue(index,
        static_cast<const void *>(&activation), sizeof(int8_t)));

    OH_NN_ReturnCode ret = OH_NNModel_AddOperation(model, opType, nullptr, &m_inputIndices, &m_outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_add_operation_003
 * @tc.desc: Verify the inputIndices is nullptr of the OH_NNModel_AddOperation function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_add_operation_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NN_OperationType opType{OH_NN_OPS_ADD};

    m_paramIndices.data = m_paramIndexs;
    m_paramIndices.size = sizeof(m_paramIndexs) / sizeof(uint32_t);

    m_outputIndices.data = m_outputIndexs;
    m_outputIndices.size = sizeof(m_outputIndexs) / sizeof(uint32_t);

    AddModelTensor(innerModel);
    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SetTensorValue(index,
        static_cast<const void *>(&activation), sizeof(int8_t)));

    OH_NN_ReturnCode ret = OH_NNModel_AddOperation(model, opType, &m_paramIndices, nullptr, &m_outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_add_operation_004
 * @tc.desc: Verify the outputIndices is nullptr of the OH_NNModel_AddOperation function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_add_operation_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NN_OperationType opType{OH_NN_OPS_ADD};

    m_paramIndices.data = m_paramIndexs;
    m_paramIndices.size = sizeof(m_paramIndexs) / sizeof(uint32_t);

    m_inputIndices.data = m_inputIndexs;
    m_inputIndices.size = sizeof(m_inputIndexs) / sizeof(uint32_t);

    AddModelTensor(innerModel);
    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SetTensorValue(index,
        static_cast<const void *>(&activation), sizeof(int8_t)));

    OH_NN_ReturnCode ret = OH_NNModel_AddOperation(model, opType, &m_paramIndices, &m_inputIndices, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_add_operation_005
 * @tc.desc: Verify the success of the OH_NNModel_AddOperation function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_add_operation_005, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NN_OperationType opType{OH_NN_OPS_ADD};

    InitIndices();
    AddModelTensor(innerModel);

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SetTensorValue(index,
        static_cast<const void *>(&activation), sizeof(int8_t)));

    OH_NN_ReturnCode ret = OH_NNModel_AddOperation(model, opType, &m_paramIndices, &m_inputIndices, &m_outputIndices);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: model_set_tensor_data_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNModel_SetTensorData function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_set_tensor_data_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = nullptr;
    AddModelTensor(innerModel);

    uint32_t index = 3;
    const int8_t activation = 0;

    OH_NN_ReturnCode ret = OH_NNModel_SetTensorData(model, index, static_cast<const void *>(&activation),
        sizeof(int8_t));
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_set_tensor_data_002
 * @tc.desc: Verify the data is nullptr of the OH_NNModel_SetTensorData function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_set_tensor_data_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    AddModelTensor(innerModel);

    uint32_t index = 3;

    OH_NN_ReturnCode ret = OH_NNModel_SetTensorData(model, index, nullptr, sizeof(int8_t));
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_set_tensor_data_003
 * @tc.desc: Verify the length is 0 of the OH_NNModel_SetTensorData function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_set_tensor_data_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    AddModelTensor(innerModel);

    uint32_t index = 3;
    const int8_t activation = 0;

    OH_NN_ReturnCode ret = OH_NNModel_SetTensorData(model, index, static_cast<const void *>(&activation), 0);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_set_tensor_data_004
 * @tc.desc: Verify the successs of the OH_NNModel_SetTensorData function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_set_tensor_data_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    AddModelTensor(innerModel);

    uint32_t index = 3;
    const int8_t activation = 0;

    OH_NN_ReturnCode ret = OH_NNModel_SetTensorData(model, index, static_cast<const void *>(&activation),
        sizeof(int8_t));
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: model_specify_inputs_and_outputs_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNModel_SpecifyInputsAndOutputs function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_specify_inputs_and_outputs_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = nullptr;

    InitIndices();
    AddModelTensor(innerModel);

    OH_NN_ReturnCode ret = OH_NNModel_SpecifyInputsAndOutputs(model, &m_inputIndices, &m_outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_specify_inputs_and_outputs_002
 * @tc.desc: Verify the inputIndices is nullptr of the OH_NNModel_SpecifyInputsAndOutputs function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_specify_inputs_and_outputs_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    InitIndices();
    AddModelTensor(innerModel);

    OH_NN_ReturnCode ret = OH_NNModel_SpecifyInputsAndOutputs(model, nullptr, &m_outputIndices);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_specify_inputs_and_outputs_003
 * @tc.desc: Verify the outputIndices is nullptr of the OH_NNModel_SpecifyInputsAndOutputs function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_specify_inputs_and_outputs_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    InitIndices();
    AddModelTensor(innerModel);

    OH_NN_ReturnCode ret = OH_NNModel_SpecifyInputsAndOutputs(model, &m_inputIndices, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_specify_inputs_and_outputs_004
 * @tc.desc: Verify the success of the OH_NNModel_SpecifyInputsAndOutputs function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_specify_inputs_and_outputs_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    InitIndices();
    AddModelTensor(innerModel);

    OH_NN_ReturnCode ret = OH_NNModel_SpecifyInputsAndOutputs(model, &m_inputIndices, &m_outputIndices);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: model_finish_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNModel_Finish function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_finish_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = nullptr;

    OH_NN_OperationType opType{OH_NN_OPS_ADD};

    InitIndices();
    AddModelTensor(innerModel);

    uint32_t index = 3;
    const int8_t activation = 0;
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SetTensorValue(index, static_cast<const void *>(&activation),
        sizeof(int8_t)));

    EXPECT_EQ(OH_NN_SUCCESS, innerModel.AddOperation(opType, m_paramIndices, m_inputIndices, m_outputIndices));
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SpecifyInputsAndOutputs(m_inputIndices, m_outputIndices));

    OH_NN_ReturnCode ret = OH_NNModel_Finish(model);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_finish_002
 * @tc.desc: Verify the success of the OH_NNModel_Finish function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_finish_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    OH_NN_OperationType opType{OH_NN_OPS_ADD};

    InitIndices();
    AddModelTensor(innerModel);

    const int8_t activation = 0;
    uint32_t index = 3;
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SetTensorValue(index,
        static_cast<const void *>(&activation), sizeof(int8_t)));

    EXPECT_EQ(OH_NN_SUCCESS, innerModel.AddOperation(opType, m_paramIndices, m_inputIndices, m_outputIndices));
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SpecifyInputsAndOutputs(m_inputIndices, m_outputIndices));

    OH_NN_ReturnCode ret = OH_NNModel_Finish(model);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: model_destroy_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNModel_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_destroy_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel** pModel = nullptr;
    OH_NNModel_Destroy(pModel);
    EXPECT_EQ(nullptr, pModel);
}

/*
 * @tc.name: model_destroy_002
 * @tc.desc: Verify the *OH_NNModel is nullptr of the OH_NNModel_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_destroy_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = nullptr;
    OH_NNModel** pModel = &model;
    OH_NNModel_Destroy(pModel);
    EXPECT_EQ(nullptr, model);
}

/*
 * @tc.name: model_destroy_003
 * @tc.desc: Verify the normal model of the OH_NNModel_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_destroy_003, testing::ext::TestSize.Level0)
{
    InnerModel* innerModel = new InnerModel();
    EXPECT_NE(nullptr, innerModel);
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(innerModel);
    OH_NNModel_Destroy(&model);
    EXPECT_EQ(nullptr, model);
}

/*
 * @tc.name: model_get_available_operation_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNModel_GetAvailableOperations function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_get_available_operation_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = nullptr;

    uint32_t opCount = 1;
    const bool *pIsAvailable = nullptr;

    InitIndices();
    AddModelTensor(innerModel);
    SetInnerBuild(innerModel);

    size_t deviceID = 10;
    OH_NN_ReturnCode ret = OH_NNModel_GetAvailableOperations(model, deviceID, &pIsAvailable, &opCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_get_available_operation_002
 * @tc.desc: Verify the isAvailable is nullptr of the OH_NNModel_GetAvailableOperations function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_get_available_operation_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    uint32_t opCount = 1;
    InitIndices();
    AddModelTensor(innerModel);
    SetInnerBuild(innerModel);

    size_t deviceID = 10;
    OH_NN_ReturnCode ret = OH_NNModel_GetAvailableOperations(model, deviceID, nullptr, &opCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_get_available_operation_003
 * @tc.desc: Verify the *isAvailable is no nullptr of the OH_NNModel_GetAvailableOperations function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_get_available_operation_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    const bool isAvailable = true;
    const bool *pIsAvailable = &isAvailable;
    uint32_t opCount = 1;

    InitIndices();
    AddModelTensor(innerModel);
    SetInnerBuild(innerModel);

    size_t deviceID = 10;
    OH_NN_ReturnCode ret = OH_NNModel_GetAvailableOperations(model, deviceID, &pIsAvailable, &opCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_get_available_operation_004
 * @tc.desc: Verify the opCount is nullptr of the OH_NNModel_GetAvailableOperations function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_get_available_operation_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    const bool *pIsAvailable = nullptr;
    uint32_t* opCount = nullptr;

    InitIndices();
    AddModelTensor(innerModel);
    SetInnerBuild(innerModel);

    size_t deviceID = 10;
    OH_NN_ReturnCode ret = OH_NNModel_GetAvailableOperations(model, deviceID, &pIsAvailable, opCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: model_get_available_operation_005
 * @tc.desc: Verify the success of the OH_NNModel_GetAvailableOperations function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, model_get_available_operation_005, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);

    const bool *pIsAvailable = nullptr;
    uint32_t opCount = 1;

    InitIndices();
    AddModelTensor(innerModel);
    SetInnerBuild(innerModel);

    size_t deviceID = 10;
    OH_NN_ReturnCode ret = OH_NNModel_GetAvailableOperations(model, deviceID, &pIsAvailable, &opCount);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_construct_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNCompilation_Construct function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_construct_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    const OH_NNModel* model = nullptr;
    OH_NNCompilation* ret = OH_NNCompilation_Construct(model);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_002
 * @tc.desc: Verify the not OH_NNModel_Build before creating compilation of the OH_NNCompilation_Construct function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_construct_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* ret = OH_NNCompilation_Construct(model);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_003
 * @tc.desc: Verify the normal model of the OH_NNCompilation_Construct function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_construct_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* ret = OH_NNCompilation_Construct(model);
    EXPECT_NE(nullptr, ret);
}

/*
 * @tc.name: compilation_set_device_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_device_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* compilation = nullptr;
    size_t deviceId = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetDevice(compilation, deviceId);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_device_002
 * @tc.desc: Verify the success of the OH_NNCompilation_SetDevice function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_device_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    size_t deviceId = 1;
    OH_NN_ReturnCode ret = OH_NNCompilation_SetDevice(nnCompilation, deviceId);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_cache_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_cache_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = nullptr;
    const char* cacheDir = "../";
    uint32_t version = 1;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));
    OH_NN_ReturnCode ret = OH_NNCompilation_SetCache(nnCompilation, cacheDir, version);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_cache_002
 * @tc.desc: Verify the cachePath is nullptr of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_cache_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    const char* cacheDir = nullptr;
    uint32_t version = 1;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));
    OH_NN_ReturnCode ret = OH_NNCompilation_SetCache(nnCompilation, cacheDir, version);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_cache_003
 * @tc.desc: Verify the success of the OH_NNCompilation_SetCache function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_cache_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    const char* cacheDir = "../";
    uint32_t version = 1;
    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));
    OH_NN_ReturnCode ret = OH_NNCompilation_SetCache(nnCompilation, cacheDir, version);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_performance_mode_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetPerformanceMode function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_performance_mode_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = nullptr;
    OH_NN_PerformanceMode performanceMode = OH_NN_PERFORMANCE_NONE;

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));
    OH_NN_ReturnCode ret = OH_NNCompilation_SetPerformanceMode(nnCompilation, performanceMode);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_performance_mode_002
 * @tc.desc: Verify the success of the OH_NNCompilation_SetPerformanceMode function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_performance_mode_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    OH_NN_PerformanceMode performanceMode = OH_NN_PERFORMANCE_NONE;

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));

    OH_NN_ReturnCode ret = OH_NNCompilation_SetPerformanceMode(nnCompilation, performanceMode);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_priority_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_SetPriority function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_priority_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = nullptr;
    OH_NN_Priority priority = OH_NN_PRIORITY_LOW;

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));

    OH_NN_ReturnCode ret = OH_NNCompilation_SetPriority(nnCompilation, priority);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_priority_002
 * @tc.desc: Verify the success of the OH_NNCompilation_SetPriority function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_priority_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    OH_NN_Priority priority = OH_NN_PRIORITY_LOW;

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));

    OH_NN_ReturnCode ret = OH_NNCompilation_SetPriority(nnCompilation, priority);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_set_enable_float16_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_EnableFloat16 function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_enable_float16_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = nullptr;
    bool enableFloat16 = true;

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));

    OH_NN_ReturnCode ret = OH_NNCompilation_EnableFloat16(nnCompilation, enableFloat16);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_set_enable_float16_002
 * @tc.desc: Verify the success of the OH_NNCompilation_EnableFloat16 function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_set_enable_float16_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    bool enableFloat16 = true;

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));

    OH_NN_ReturnCode ret = OH_NNCompilation_EnableFloat16(nnCompilation, enableFloat16);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_build_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_Build function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_build_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = nullptr;

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetPerformance(OH_NN_PERFORMANCE_EXTREME));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetPriority(OH_NN_PRIORITY_HIGH));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetEnableFp16(true));

    OH_NN_ReturnCode ret = OH_NNCompilation_Build(nnCompilation);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: compilation_build_002
 * @tc.desc: Verify the success of the OH_NNCompilation_Build function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_build_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetPerformance(OH_NN_PERFORMANCE_EXTREME));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetPriority(OH_NN_PRIORITY_HIGH));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetEnableFp16(true));

    OH_NN_ReturnCode ret = OH_NNCompilation_Build(nnCompilation);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: compilation_destroy_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNCompilation_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_destroy_001, testing::ext::TestSize.Level0)
{
    OH_NNCompilation** pCompilation = nullptr;
    OH_NNCompilation_Destroy(pCompilation);
    EXPECT_EQ(nullptr, pCompilation);
}

/*
 * @tc.name: compilation_destroy_002
 * @tc.desc: Verify the *OH_NNCompilation is nullptr of the OH_NNCompilation_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_destroy_002, testing::ext::TestSize.Level0)
{
    OH_NNCompilation* compilation = nullptr;
    OH_NNCompilation** pCompilation = &compilation;
    OH_NNCompilation_Destroy(pCompilation);
    EXPECT_EQ(nullptr, compilation);
}

/*
 * @tc.name: compilation_destroy_003
 * @tc.desc: Verify the normal model of the OH_NNCompilation_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_destroy_003, testing::ext::TestSize.Level0)
{
    InnerModel* innerModel = new InnerModel();
    EXPECT_NE(nullptr, innerModel);
    Compilation* compilation = new(std::nothrow) Compilation(innerModel);
    EXPECT_NE(nullptr, compilation);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(compilation);
    OH_NNCompilation_Destroy(&nnCompilation);
    EXPECT_EQ(nullptr, nnCompilation);
}

/**
 * @tc.name: excutor_construct_001
 * @tc.desc: Verify the OH_NNCompilation is nullptr of the OH_NNExecutor_Construct function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_construct_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetEnableFp16(true));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetPerformance(OH_NN_PERFORMANCE_EXTREME));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetPriority(OH_NN_PRIORITY_HIGH));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.Build());

    OH_NNCompilation* nnCompilation = nullptr;
    OH_NNExecutor* executor = OH_NNExecutor_Construct(nnCompilation);
    EXPECT_EQ(nullptr, executor);
}

/**
 * @tc.name: excutor_construct_002
 * @tc.desc: Verify the not OH_NNCompilation_Build before creating executor of the OH_NNExecutor_Construct function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_construct_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);
    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    OH_NNExecutor * executor = OH_NNExecutor_Construct(nnCompilation);
    EXPECT_EQ(nullptr, executor);
}

/**
 * @tc.name: excutor_construct_003
 * @tc.desc: Verify the success of the OH_NNExecutor_Construct function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_construct_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation compilation(&innerModel);

    std::size_t deviceId = 1;
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetDevice(deviceId));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetPerformance(OH_NN_PERFORMANCE_EXTREME));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetPriority(OH_NN_PRIORITY_HIGH));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.SetEnableFp16(true));
    EXPECT_EQ(OH_NN_SUCCESS, compilation.Build());

    OH_NNCompilation* nnCompilation = reinterpret_cast<OH_NNCompilation*>(&compilation);
    OH_NNExecutor * executor = OH_NNExecutor_Construct(nnCompilation);
    EXPECT_NE(nullptr, executor);
}

/**
 * @tc.name: excutor_setinput_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_001, testing::ext::TestSize.Level0)
{
    SetTensor();

    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const void *buffer = input;
    size_t length = 2 * sizeof(float);
    uint32_t inputIndex = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInput(nullptr, inputIndex, &m_tensor, buffer, length));
}

/**
 * @tc.name: excutor_setinput_002
 * @tc.desc: Verify the OH_NN_Tensor is nullptr of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t inputIndex = 0;
    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const void *buffer = input;
    size_t length = 2 * sizeof(float);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInput(nnExecutor, inputIndex, nullptr, buffer, length));
}

/**
 * @tc.name: excutor_setinput_003
 * @tc.desc: Verify the data is nullptr of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    SetTensor();

    uint32_t inputIndex = 0;
    const void *buffer = nullptr;
    size_t length = 2 * sizeof(float);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInput(nnExecutor, inputIndex, &m_tensor, buffer, length));
}

/**
 * @tc.name: excutor_setinput_004
 * @tc.desc: Verify the length is 0 of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t inputIndex = 0;
    SetTensor();

    size_t length = 0;
    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const void *buffer = input;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetInput(nnExecutor, inputIndex, &m_tensor, buffer, length));
}

/**
 * @tc.name: excutor_setinput_005
 * @tc.desc: Verify the success of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_005, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t inputIndex = 0;
    SetTensor();

    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    const void *buffer = input;
    size_t length = 9 * sizeof(int32_t);
    OH_NN_ReturnCode ret = OH_NNExecutor_SetInput(nnExecutor, inputIndex, &m_tensor, buffer, length);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/**
 * @tc.name: excutor_setoutput_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_SetOutput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setoutput_001, testing::ext::TestSize.Level0)
{
    uint32_t outputIndex = 0;
    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void *buffer = input;
    size_t length = 9 * sizeof(int32_t);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutput(nullptr, outputIndex, buffer, length));
}

/**
 * @tc.name: excutor_setoutput_002
 * @tc.desc: Verify the data is nullptr of the OH_NNExecutor_SetOutput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setoutput_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    void *buffer = nullptr;
    size_t length = 9 * sizeof(int32_t);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutput(nnExecutor, outputIndex, buffer, length));
}

/**
 * @tc.name: excutor_setoutput_003
 * @tc.desc: Verify the length is 0 of the OH_NNExecutor_SetOutput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setoutput_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void *buffer = input;
    size_t length = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutput(nnExecutor, outputIndex, buffer, length));
}

/**
 * @tc.name: excutor_setoutput_004
 * @tc.desc: Verify the success of the OH_NNExecutor_SetOutput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setoutput_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void *buffer = input;
    size_t length = 9 * sizeof(int32_t);
    EXPECT_EQ(OH_NN_SUCCESS, OH_NNExecutor_SetOutput(nnExecutor, outputIndex, buffer, length));
}

/**
 * @tc.name: excutor_getoutputshape_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_GetOutputShape function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_getoutputshape_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = nullptr;

    SetInputAndOutput(executor);

    int32_t* ptr = nullptr;
    int32_t** shape = &ptr;
    uint32_t length = 2;
    uint32_t outputIndex = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputShape(nnExecutor, outputIndex,
        shape, &length));
}

/**
 * @tc.name: excutor_getoutputshape_002
 * @tc.desc: Verify the shape is nullptr of the OH_NNExecutor_GetOutputShape function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_getoutputshape_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    SetInputAndOutput(executor);

    uint32_t outputIndex = 0;
    int32_t** shape = nullptr;
    uint32_t length = 2;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputShape(nnExecutor, outputIndex,
        shape, &length));
}

/**
 * @tc.name: excutor_getoutputshape_003
 * @tc.desc: Verify the *shape is not nullptr of the OH_NNExecutor_GetOutputShape function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_getoutputshape_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    SetInputAndOutput(executor);

    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    int32_t** shape = &ptr;
    uint32_t length = 2;
    uint32_t outputIndex = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputShape(nnExecutor, outputIndex,
        shape, &length));
}

/**
 * @tc.name: excutor_getoutputshape_004
 * @tc.desc: Verify the length is nullptr of the OH_NNExecutor_GetOutputShape function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_getoutputshape_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    SetInputAndOutput(executor);

    int32_t* ptr = nullptr;
    int32_t** shape = &ptr;
    uint32_t outputIndex = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputShape(nnExecutor, outputIndex, shape, nullptr));
}

/**
 * @tc.name: excutor_getoutputshape_005
 * @tc.desc: Verify the success of the OH_NNExecutor_GetOutputShape function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_getoutputshape_005, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    SetInputAndOutput(executor);

    int32_t* ptr = nullptr;
    int32_t** shape = &ptr;
    uint32_t length = 2;
    uint32_t outputIndex = 0;
    EXPECT_EQ(OH_NN_SUCCESS, OH_NNExecutor_GetOutputShape(nnExecutor, outputIndex, shape, &length));
}

/**
 * @tc.name: excutor_run_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_Run function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_run_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* nnExecutor = nullptr;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_Run(nnExecutor));
}

/**
 * @tc.name: excutor_run_002
 * @tc.desc: Verify the success of the OH_NNExecutor_Run function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_run_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t index = 0;
    size_t length = 9 * sizeof(int32_t);
    float input[9] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void *buffer = input;

    SetTensor();
    EXPECT_EQ(OH_NN_SUCCESS, executor.SetInput(index, m_tensor, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, executor.SetOutput(index, buffer, length));
    EXPECT_EQ(OH_NN_SUCCESS, OH_NNExecutor_Run(nnExecutor));
}

/*
 * @tc.name: executor_allocate_input_memory_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_input_memory_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* nnExecutor = nullptr;
    uint32_t outputIndex = 0;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateInputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_input_memory_002
 * @tc.desc: Verify the passed length equals 0 of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_input_memory_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    size_t length = 0;

    OH_NN_Memory* ret = OH_NNExecutor_AllocateInputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_input_memory_003
 * @tc.desc: Verify the error when creating input memory in executor of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_input_memory_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 6;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateInputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_input_memory_004
 * @tc.desc: Verify the success of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_input_memory_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateInputMemory(nnExecutor, outputIndex, length);
    EXPECT_NE(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_output_memory_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_AllocateOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_output_memory_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* nnExecutor = nullptr;
    uint32_t outputIndex = 0;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateOutputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_output_memory_002
 * @tc.desc: Verify the passed length equals 0 of the OH_NNExecutor_AllocateOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_output_memory_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    size_t length = 0;

    OH_NN_Memory* ret = OH_NNExecutor_AllocateOutputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_output_memory_003
 * @tc.desc: Verify the error when create output memory in executor of the OH_NNExecutor_AllocateOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_output_memory_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 6;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateOutputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_output_memory_004
 * @tc.desc: Verify the success of the OH_NNExecutor_AllocateOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_output_memory_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateOutputMemory(nnExecutor, outputIndex, length);
    EXPECT_NE(nullptr, ret);
}


/*
 * @tc.name: executor_destroy_input_memory_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_DestroyInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_input_memory_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildModelGraph(innerModel);
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = nullptr;

    uint32_t inputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    size_t length = 9 * sizeof(float);
    EXPECT_EQ(OH_NN_SUCCESS, executor.CreateInputMemory(inputIndex, length, &pMemory));
    OH_NNExecutor_DestroyInputMemory(nnExecutor, inputIndex, &pMemory);
    EXPECT_EQ(nullptr, nnExecutor);
}

/*
 * @tc.name: executor_destroy_input_memory_002
 * @tc.desc: Verify the memory is nullptr of the OH_NNExecutor_DestroyInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_input_memory_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildModelGraph(innerModel);
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t inputIndex = 0;
    OH_NN_Memory** memory = nullptr;
    OH_NNExecutor_DestroyInputMemory(nnExecutor, inputIndex, memory);
    EXPECT_EQ(nullptr, memory);
}

/*
 * @tc.name: executor_destroy_input_memory_003
 * @tc.desc: Verify the *memory is nullptr of the OH_NNExecutor_DestroyInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_input_memory_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildModelGraph(innerModel);
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t inputIndex = 0;
    OH_NN_Memory* memory = nullptr;
    OH_NN_Memory** pMemory = &memory;
    OH_NNExecutor_DestroyInputMemory(nnExecutor, inputIndex, pMemory);
    EXPECT_EQ(nullptr, memory);
}

/*
 * @tc.name: executor_destroy_input_memory_004
 * @tc.desc: Verify the error happened when destroying input memory of the OH_NNExecutor_DestroyInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_input_memory_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildModelGraph(innerModel);
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t inputIndex = 6;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    OH_NNExecutor_DestroyInputMemory(nnExecutor, inputIndex, &pMemory);
    EXPECT_NE(nullptr, pMemory);
}

/*
 * @tc.name: executor_destroy_input_memory_005
 * @tc.desc: Verify the success of the OH_NNExecutor_DestroyInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_input_memory_005, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildModelGraph(innerModel);
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t inputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    size_t length = 9 * sizeof(float);
    EXPECT_EQ(OH_NN_SUCCESS, executor.CreateInputMemory(inputIndex, length, &pMemory));
    OH_NNExecutor_DestroyInputMemory(nnExecutor, inputIndex, &pMemory);
    EXPECT_EQ(nullptr, pMemory);
}

/*
 * @tc.name: executor_destroy_output_memory_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_DestroyOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_output_memory_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* nnExecutor = nullptr;
    uint32_t outputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, &pMemory);
    EXPECT_EQ(nullptr, nnExecutor);
}

/*
 * @tc.name: executor_destroy_output_memory_002
 * @tc.desc: Verify the memory is nullptr of the OH_NNExecutor_DestroyOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_output_memory_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    OH_NN_Memory** memory = nullptr;
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, memory);
    EXPECT_EQ(nullptr, memory);
}

/*
 * @tc.name: executor_destroy_output_memory_003
 * @tc.desc: Verify the *memory is nullptr of the OH_NNExecutor_DestroyOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_output_memory_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    OH_NN_Memory* memory = nullptr;
    OH_NN_Memory** pMemory = &memory;
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, pMemory);
    EXPECT_EQ(nullptr, memory);
}

/*
 * @tc.name: executor_destroy_output_memory_004
 * @tc.desc: Verify the error happened when destroying output memory of the OH_NNExecutor_DestroyOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_output_memory_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 6;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, &pMemory);
    EXPECT_NE(nullptr, pMemory);
}

/*
 * @tc.name: executor_destroy_output_memory_005
 * @tc.desc: Verify the success of the OH_NNExecutor_DestroyOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_output_memory_005, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    size_t length = 9 * sizeof(float);
    uint32_t outputIndex = 0;
    EXPECT_EQ(OH_NN_SUCCESS, executor.CreateOutputMemory(outputIndex, length, &pMemory));
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, &pMemory);
    EXPECT_EQ(nullptr, pMemory);
}

/*
 * @tc.name: executor_set_input_with_memory_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_SetInputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_input_with_memory_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* nnExecutor = nullptr;

    SetTensor();

    uint32_t inputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = OH_NNExecutor_SetInputWithMemory(nnExecutor, inputIndex, &m_tensor, &memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_with_memory_002
 * @tc.desc: Verify the operand is nullptr of the OH_NNExecutor_SetInputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_input_with_memory_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    OH_NN_Tensor* operand = nullptr;

    uint32_t inputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = OH_NNExecutor_SetInputWithMemory(nnExecutor, inputIndex, operand, &memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_with_memory_003
 * @tc.desc: Verify the memory is nullptr of the OH_NNExecutor_SetInputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_input_with_memory_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    SetTensor();

    uint32_t inputIndex = 0;
    OH_NN_Memory* memory = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetInputWithMemory(nnExecutor, inputIndex, &m_tensor, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_input_with_memory_004
 * @tc.desc: Verify the success of the OH_NNExecutor_SetInputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_input_with_memory_004, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    SetTensor();

    uint32_t inputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = OH_NNExecutor_SetInputWithMemory(nnExecutor, inputIndex, &m_tensor, &memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}


/*
 * @tc.name: executor_set_output_with_memory_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_SetOutputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_output_with_memory_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* nnExecutor = nullptr;
    uint32_t outputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOutputWithMemory(nnExecutor, outputIndex, &memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_with_memory_002
 * @tc.desc: Verify the memory is nullptr of the OH_NNExecutor_SetOutputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_output_with_memory_002, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    OH_NN_Memory* memory = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOutputWithMemory(nnExecutor, outputIndex, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: executor_set_output_with_memory_003
 * @tc.desc: Verify the success of the OH_NNExecutor_SetOutputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_output_with_memory_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModelGraph(innerModel));
    Compilation innerCompilation(&innerModel);
    Executor executor(&innerCompilation);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(&executor);

    uint32_t outputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOutputWithMemory(nnExecutor, outputIndex, &memory);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: executor_destroy_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_001, testing::ext::TestSize.Level0)
{
    OH_NNExecutor** pExecutor = nullptr;
    OH_NNExecutor_Destroy(pExecutor);
    EXPECT_EQ(nullptr, pExecutor);
}

/*
 * @tc.name: executor_destroy_002
 * @tc.desc: Verify the *OH_NNExecutor is nullptr of the OH_NNExecutor_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_002, testing::ext::TestSize.Level0)
{
    OH_NNExecutor* nnExecutor = nullptr;
    OH_NNExecutor** pExecutor = &nnExecutor;
    OH_NNExecutor_Destroy(pExecutor);
    EXPECT_EQ(nullptr, nnExecutor);
}

/*
 * @tc.name: executor_destroy_003
 * @tc.desc: Verify the normal model of the OH_NNExecutor_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_003, testing::ext::TestSize.Level0)
{
    InnerModel* innerModel = new InnerModel();
    EXPECT_NE(nullptr, innerModel);
    Compilation* innerCompilation = new(std::nothrow) Compilation(innerModel);
    EXPECT_NE(nullptr, innerCompilation);
    Executor* executor = new(std::nothrow) Executor(innerCompilation);
    EXPECT_NE(nullptr, executor);

    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);
    OH_NNExecutor_Destroy(&nnExecutor);
    EXPECT_EQ(nullptr, nnExecutor);
}

/*
 * @tc.name: device_get_all_devices_id_001
 * @tc.desc: Verify the allDevicesID is nullptr of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_all_devices_id_001, testing::ext::TestSize.Level0)
{
    const size_t** allDevicesId = nullptr;
    uint32_t deviceCount = 1;
    uint32_t* pDeviceCount = &deviceCount;
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(allDevicesId, pDeviceCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_all_devices_id_002
 * @tc.desc: Verify the *allDevicesID is not nullptr of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_all_devices_id_002, testing::ext::TestSize.Level0)
{
    const size_t devicesId = 1;
    const size_t* allDevicesId = &devicesId;
    const size_t** pAllDevicesId = &allDevicesId;
    uint32_t deviceCount = 1;
    uint32_t* pDeviceCount = &deviceCount;
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(pAllDevicesId, pDeviceCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_all_devices_id_003
 * @tc.desc: Verify the deviceCount is nullptr of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_all_devices_id_003, testing::ext::TestSize.Level0)
{
    const size_t* allDevicesId = nullptr;
    const size_t** pAllDevicesId = &allDevicesId;
    uint32_t* pDeviceCount = nullptr;
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(pAllDevicesId, pDeviceCount);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_all_devices_id_004
 * @tc.desc: Verify the get no device of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_all_devices_id_004, testing::ext::TestSize.Level0)
{
    const size_t* allDevicesId = nullptr;
    const size_t** pAllDevicesId = &allDevicesId;
    uint32_t deviceCount = 1;
    uint32_t* pDeviceCount = &deviceCount;
    OHOS::HDI::Nnrt::V1_0::MockIPreparedModel::m_ExpectRetCode = OH_NN_FAILED;
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(pAllDevicesId, pDeviceCount);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: device_get_all_devices_id_005
 * @tc.desc: Verify the success of the OH_NNDevice_GetAllDevicesID function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_all_devices_id_005, testing::ext::TestSize.Level0)
{
    const size_t* allDevicesId = nullptr;
    const size_t** pAllDevicesId = &allDevicesId;
    uint32_t deviceCount = 1;
    uint32_t* pDeviceCount = &deviceCount;
    OH_NN_ReturnCode ret = OH_NNDevice_GetAllDevicesID(pAllDevicesId, pDeviceCount);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: device_get_name_001
 * @tc.desc: Verify the name is nullptr of the OH_NNDevice_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_name_001, testing::ext::TestSize.Level0)
{
    size_t deviceID = 1;
    const char **name = nullptr;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceID, name);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_name_002
 * @tc.desc: Verify the *name is not nullptr of the OH_NNDevice_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_name_002, testing::ext::TestSize.Level0)
{
    size_t deviceID = 1;
    const char* name = "diviceId";
    const char** pName = &name;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceID, pName);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_name_003
 * @tc.desc: Verify the error happened when getting name of deviceID of the OH_NNDevice_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_name_003, testing::ext::TestSize.Level0)
{
    size_t deviceID = 0;
    const char* name = nullptr;
    const char** pName = &name;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceID, pName);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: device_get_name_004
 * @tc.desc: Verify the success of the OH_NNDevice_GetName function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_name_004, testing::ext::TestSize.Level0)
{
    size_t deviceID = 1;
    const char* name = nullptr;
    const char** pName = &name;
    OH_NN_ReturnCode ret = OH_NNDevice_GetName(deviceID, pName);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: device_get_type_001
 * @tc.desc: Verify the device is nullptr of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_type_001, testing::ext::TestSize.Level0)
{
    size_t deviceID = 0;
    OH_NN_DeviceType deviceType = OH_NN_CPU;
    OH_NN_DeviceType* pDeviceType = &deviceType;
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceID, pDeviceType);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_type_002
 * @tc.desc: Verify the OH_NN_DeviceType is nullptr of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_type_002, testing::ext::TestSize.Level0)
{
    size_t deviceID = 1;
    OH_NN_DeviceType* pDeviceType = nullptr;
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceID, pDeviceType);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: device_get_type_003
 * @tc.desc: Verify the error happened when getting name of deviceID of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_type_003, testing::ext::TestSize.Level0)
{
    size_t deviceID = 1;
    OH_NN_DeviceType deviceType = OH_NN_OTHERS;
    OH_NN_DeviceType* pDeviceType = &deviceType;
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceID, pDeviceType);
    EXPECT_EQ(OH_NN_UNAVALIDABLE_DEVICE, ret);
}

/*
 * @tc.name: device_get_type_004
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_type_004, testing::ext::TestSize.Level0)
{
    size_t deviceID =  1;
    OH_NN_DeviceType deviceType = OH_NN_CPU;
    OH_NN_DeviceType* pDeviceType = &deviceType;
    OH_NN_ReturnCode ret = OH_NNDevice_GetType(deviceID, pDeviceType);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}
} // namespace Unittest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
