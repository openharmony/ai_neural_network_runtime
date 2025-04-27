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

#include "utils.h"
#include "compilation.h"
#include "hdi_device_v1_0.h"
#include "test/unittest/common/v1_0/mock_idevice.h"
#include "nnexecutor.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
OH_NN_ReturnCode HDIDeviceV1_0::PrepareModel(std::shared_ptr<const mindspore::lite::LiteGraph> model,
    const ModelConfig& config, std::shared_ptr<PreparedModel>& preparedModel)
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
        LOGE("HDIDeviceV1_0 mock PrepareModel failed, error happened when new sptr");
        return OH_NN_NULL_PTR;
    }

    preparedModel = CreateSharedPtr<HDIPreparedModelV1_0>(iPreparedModel);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::GetDeviceType(OH_NN_DeviceType& deviceType)
{
    if (deviceType == OH_NN_OTHERS) {
        return OH_NN_UNAVAILABLE_DEVICE;
    }

    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsModelCacheSupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsPerformanceModeSupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsPrioritySupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsFloat16PrecisionSupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::GetSupportedOperation(std::shared_ptr<const mindspore::lite::LiteGraph> model,
    std::vector<bool>& ops)
{
    if (model == nullptr) {
        LOGE("HDIDeviceV1_0 mock GetSupportedOperation failed, Model is nullptr, cannot query supported operation.");
        return OH_NN_NULL_PTR;
    }

    ops.emplace_back(true);
    return OH_NN_SUCCESS;
}

OH_NN_ReturnCode HDIDeviceV1_0::IsDynamicInputSupported(bool& isSupported)
{
    isSupported = true;
    return OH_NN_SUCCESS;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Unittest {
OH_NN_ReturnCode NeuralNetworkRuntimeTest::BuildModel(InnerModel& model)
{
    int32_t inputDims[2] = {3, 4};
    OH_NN_Tensor input1 = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    OH_NN_ReturnCode ret = model.AddTensor(input1);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 添加Add算子的第二个输入Tensor，类型为float32，张量形状为[3, 4]
    OH_NN_Tensor input2 = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    ret = model.AddTensor(input2);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 添加Add算子的参数Tensor，该参数Tensor用于指定激活函数的类型，Tensor的数据类型为int8。
    int32_t activationDims = 1;
    int8_t activationValue = OH_NN_FUSED_NONE;
    OH_NN_Tensor activation = {OH_NN_INT8, 1, &activationDims, nullptr, OH_NN_ADD_ACTIVATIONTYPE};
    ret = model.AddTensor(activation);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 将激活函数类型设置为OH_NN_FUSED_NONE，表示该算子不添加激活函数。
    uint32_t index = 2;
    ret = model.SetTensorValue(index, &activationValue, sizeof(int8_t));
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 设置Add算子的输出，类型为float32，张量形状为[3, 4]
    OH_NN_Tensor output = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    ret = model.AddTensor(output);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 指定Add算子的输入、参数和输出索引
    uint32_t inputIndicesValues[2] = {0, 1};
    uint32_t paramIndicesValues = 2;
    uint32_t outputIndicesValues = 3;
    OH_NN_UInt32Array paramIndices = {&paramIndicesValues, 1};
    OH_NN_UInt32Array inputIndices = {inputIndicesValues, 2};
    OH_NN_UInt32Array outputIndices = {&outputIndicesValues, 1};

    // 向模型实例添加Add算子
    ret = model.AddOperation(OH_NN_OPS_ADD, paramIndices, inputIndices, outputIndices);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 设置模型实例的输入、输出索引
    ret = model.SpecifyInputsAndOutputs(inputIndices, outputIndices);
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    // 完成模型实例的构建
    ret = model.Build();
    if (ret != OH_NN_SUCCESS) {
        return ret;
    }

    return ret;
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

    OH_NN_OperationType opType {OH_NN_OPS_ADD};
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.AddOperation(opType, m_paramIndices, m_inputIndices, m_outputIndices));
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.SpecifyInputsAndOutputs(m_inputIndices, m_outputIndices));
    EXPECT_EQ(OH_NN_SUCCESS, innerModel.Build());
}

void NeuralNetworkRuntimeTest::SetInputAndOutput(Executor& executor)
{
    size_t input1Index = 0;
    int32_t inputDims[2] = {3, 4};
    size_t lengthSize = 12 * sizeof(float);
    size_t *length = &lengthSize;

    size_t minInputDims = 1;
    size_t maxInputDims = 12;

    size_t *minInputDimsAdress = &minInputDims;
    size_t **minInputDimsAdressA = &minInputDimsAdress;
    
    size_t *maxInputDimsAdress = &maxInputDims;
    size_t **maxInputDimsAdressA = &maxInputDimsAdress;

    m_tensor = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_SUCCESS, executor.GetInputDimRange(input1Index, minInputDimsAdressA, maxInputDimsAdressA, length));

    uint32_t outputIndex = 0;

    int32_t shape = 3;
    int32_t* shapeA = &shape;
    int32_t** shapeAA = &shapeA;
    uint32_t* shapeNum = &outputIndex;
    EXPECT_EQ(OH_NN_SUCCESS, executor.GetOutputShape(outputIndex, shapeAA, shapeNum));
}

class MockIPreparedModel : public PreparedModel {
public:
    MOCK_METHOD1(ExportModelCache, OH_NN_ReturnCode(std::vector<Buffer>&));
    MOCK_METHOD4(Run, OH_NN_ReturnCode(const std::vector<IOTensor>&,
                                 const std::vector<IOTensor>&,
                                 std::vector<std::vector<int32_t>>&,
                                 std::vector<bool>&));
    MOCK_METHOD4(Run, OH_NN_ReturnCode(const std::vector<NN_Tensor*>&,
                                 const std::vector<NN_Tensor*>&,
                                 std::vector<std::vector<int32_t>>&,
                                 std::vector<bool>&));
    MOCK_CONST_METHOD1(GetModelID, OH_NN_ReturnCode(uint32_t&));
    MOCK_METHOD2(GetInputDimRanges, OH_NN_ReturnCode(std::vector<std::vector<uint32_t>>&,
                                               std::vector<std::vector<uint32_t>>&));
    MOCK_METHOD0(ReleaseBuiltModel, OH_NN_ReturnCode());
};

class MockIDevice : public Device {
public:
    MOCK_METHOD1(GetDeviceName, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetVendorName, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetVersion, OH_NN_ReturnCode(std::string&));
    MOCK_METHOD1(GetDeviceType, OH_NN_ReturnCode(OH_NN_DeviceType&));
    MOCK_METHOD1(GetDeviceStatus, OH_NN_ReturnCode(DeviceStatus&));
    MOCK_METHOD2(GetSupportedOperation, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
        std::vector<bool>&));
    MOCK_METHOD1(IsFloat16PrecisionSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsPerformanceModeSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsPrioritySupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsDynamicInputSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD1(IsModelCacheSupported, OH_NN_ReturnCode(bool&));
    MOCK_METHOD3(PrepareModel, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
                                          const ModelConfig&,
                                          std::shared_ptr<PreparedModel>&));
    MOCK_METHOD3(PrepareModel, OH_NN_ReturnCode(const void*,
                                          const ModelConfig&,
                                          std::shared_ptr<PreparedModel>&));
    MOCK_METHOD4(PrepareModelFromModelCache, OH_NN_ReturnCode(const std::vector<Buffer>&,
                                                              const ModelConfig&,
                                                              std::shared_ptr<PreparedModel>&,
                                                              bool&));
    MOCK_METHOD3(PrepareOfflineModel, OH_NN_ReturnCode(std::shared_ptr<const mindspore::lite::LiteGraph>,
                                                 const ModelConfig&,
                                                 std::shared_ptr<PreparedModel>&));
    MOCK_METHOD1(AllocateBuffer, void*(size_t));
    MOCK_METHOD2(AllocateTensorBuffer, void*(size_t, std::shared_ptr<TensorDesc>));
    MOCK_METHOD2(AllocateTensorBuffer, void*(size_t, std::shared_ptr<NNTensor>));
    MOCK_METHOD1(ReleaseBuffer, OH_NN_ReturnCode(const void*));
    MOCK_METHOD2(AllocateBuffer, OH_NN_ReturnCode(size_t, int&));
    MOCK_METHOD2(ReleaseBuffer, OH_NN_ReturnCode(int, size_t));
    MOCK_METHOD1(ReadOpVersion, OH_NN_ReturnCode(int&));
};

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
    OH_NN_OperationType opType {OH_NN_OPS_ADD};

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
    OH_NN_OperationType opType {OH_NN_OPS_ADD};

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
    OH_NN_OperationType opType {OH_NN_OPS_ADD};

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
    OH_NN_OperationType opType {OH_NN_OPS_ADD};

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
    OH_NN_OperationType opType {OH_NN_OPS_ADD};

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

    OH_NN_OperationType opType {OH_NN_OPS_ADD};

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

    OH_NN_OperationType opType {OH_NN_OPS_ADD};

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
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: compilation_construct_001
 * @tc.desc: Verify the OH_NNModel is nullptr of the OH_NNCompilation_Construct function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_construct_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
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
    EXPECT_NE(nullptr, ret);
}

/*
 * @tc.name: compilation_construct_003
 * @tc.desc: Verify the normal model of the OH_NNCompilation_Construct function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_construct_003, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    OH_NNCompilation* nnCompilation = nullptr;
    const char* cacheDir = "../";
    uint32_t version = 1;
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    const char* cacheDir = nullptr;
    uint32_t version = 1;
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    const char* cacheDir = "../";
    uint32_t version = 1;
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    OH_NNCompilation* nnCompilation = nullptr;
    OH_NN_PerformanceMode performanceMode = OH_NN_PERFORMANCE_NONE;

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NN_PerformanceMode performanceMode = OH_NN_PERFORMANCE_NONE;

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    OH_NNCompilation* nnCompilation = nullptr;
    OH_NN_Priority priority = OH_NN_PRIORITY_LOW;

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NN_Priority priority = OH_NN_PRIORITY_LOW;

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    OH_NNCompilation* nnCompilation = nullptr;
    bool enableFloat16 = true;

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    bool enableFloat16 = true;

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    OH_NNCompilation* nnCompilation = nullptr;

    OH_NN_ReturnCode ret = OH_NNCompilation_Build(nnCompilation);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
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
 * @tc.name: compilation_destroy_003
 * @tc.desc: Verify the normal model of the OH_NNCompilation_Destroy function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, compilation_destroy_003, testing::ext::TestSize.Level0)
{
    InnerModel* innerModel = new InnerModel();
    EXPECT_NE(nullptr, innerModel);

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor * executor = OH_NNExecutor_Construct(nnCompilation);
    EXPECT_EQ(nullptr, executor);
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

    uint32_t inputIndex = 0;
    int32_t dims[2] = {3, 4};
    m_tensor = {OH_NN_FLOAT32, 2, dims, nullptr, OH_NN_TENSOR};

    float input[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const void *buffer = input;
    size_t length = 12 * sizeof(float);
    OH_NN_ReturnCode ret = OH_NNExecutor_SetInput(nnExecutor, inputIndex, &m_tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/**
 * @tc.name: excutor_setinput_006
 * @tc.desc: Verify the success of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_006, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetInput excutor_setinput_006");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t inputIndex = 0;
    int32_t dims[2] = {3, 4};
    m_tensor = {OH_NN_FLOAT32, 2, dims, nullptr, OH_NN_TENSOR};

    float input[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const void *buffer = input;
    size_t length = 12 * sizeof(float);
    OH_NN_ReturnCode ret = OH_NNExecutor_SetInput(nnExecutor, inputIndex, &m_tensor, buffer, length);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: excutor_setinput_007
 * @tc.desc: Verify the success of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_007, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetInput excutor_setinput_007");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t inputIndex = 0;

    float input[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const void *buffer = input;
    size_t length = 12 * sizeof(float);
    OH_NN_ReturnCode ret = OH_NNExecutor_SetInput(nnExecutor, inputIndex, nullptr, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: excutor_setinput_008
 * @tc.desc: Verify the success of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_008, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetInput excutor_setinput_008");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t inputIndex = 0;
    int32_t dims[2] = {3, 4};
    m_tensor = {OH_NN_FLOAT32, 2, dims, nullptr, OH_NN_TENSOR};

    size_t length = 12 * sizeof(float);
    OH_NN_ReturnCode ret = OH_NNExecutor_SetInput(nnExecutor, inputIndex, &m_tensor, nullptr, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: excutor_setinput_009
 * @tc.desc: Verify the success of the OH_NNExecutor_SetInput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setinput_009, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetInput excutor_setinput_009");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t inputIndex = 0;
    int32_t dims[2] = {3, 4};
    m_tensor = {OH_NN_FLOAT32, 2, dims, nullptr, OH_NN_TENSOR};

    float input[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    const void *buffer = input;
    size_t length = 0;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetInput(nnExecutor, inputIndex, &m_tensor, buffer, length);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: excutor_setoutput_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_SetOutput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setoutput_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetOutput excutor_setoutput_001");
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

    uint32_t outputIndex = 0;
    float output[12];
    size_t length = 12 * sizeof(float);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutput(nnExecutor, outputIndex, output, length));
}

/**
 * @tc.name: excutor_setoutput_005
 * @tc.desc: Verify the success of the OH_NNExecutor_SetOutput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setoutput_005, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetOutput excutor_setinput_006");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    float output[12];
    size_t length = 12 * sizeof(float);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutput(nnExecutor, outputIndex, output, length));

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: excutor_setoutput_006
 * @tc.desc: Verify the success of the OH_NNExecutor_SetOutput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setoutput_006, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetOutput excutor_setinput_006");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    size_t length = 12 * sizeof(float);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutput(nnExecutor, outputIndex, nullptr, length));

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: excutor_setoutput_007
 * @tc.desc: Verify the success of the OH_NNExecutor_SetOutput function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_setoutput_007, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetOutput excutor_setoutput_007");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    float output[12];
    size_t length = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_SetOutput(nnExecutor, outputIndex, output, length));

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/**
 * @tc.name: excutor_getoutputshape_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_GetOutputShape function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_getoutputshape_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));
    OH_NNExecutor* nnExecutor = nullptr;

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

    int32_t* ptr = nullptr;
    int32_t** shape = &ptr;
    uint32_t length = 2;
    uint32_t outputIndex = 0;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_GetOutputShape(nnExecutor, outputIndex, shape, &length));
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

    int32_t inputDims[2] = {3, 4};
    m_tensor = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, OH_NNExecutor_Run(nnExecutor));
}

/**
 * @tc.name: excutor_run_003
 * @tc.desc: Verify the success of the OH_NNExecutor_Run function
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, excutor_run_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_Run excutor_run_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    int32_t inputDims[2] = {3, 4};
    m_tensor = {OH_NN_FLOAT32, 2, inputDims, nullptr, OH_NN_TENSOR};
    OH_NN_ReturnCode ret = OH_NNExecutor_Run(nnExecutor);
    EXPECT_EQ(OH_NN_SUCCESS, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

    uint32_t outputIndex = 0;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateInputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_input_memory_005
 * @tc.desc: Verify the success of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_input_memory_005, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_AllocateInputMemory executor_allocate_input_memory_005");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateInputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_allocate_input_memory_006
 * @tc.desc: Verify the success of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_input_memory_006, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_AllocateInputMemory executor_allocate_input_memory_006");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_inputTensorDescs[0].first))
        .WillRepeatedly(::testing::Return(reinterpret_cast<void*>(0x1000)));

    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);
    EXPECT_NE(nullptr, executor);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;

    OH_NN_Memory* ret = OH_NNExecutor_AllocateInputMemory(nnExecutor, outputIndex, length);
    EXPECT_NE(nullptr, ret);

    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: executor_allocate_input_memory_007
 * @tc.desc: Verify the success of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_input_memory_007, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_AllocateInputMemory executor_allocate_input_memory_007");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    size_t length = 0;

    OH_NN_Memory* ret = OH_NNExecutor_AllocateInputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

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
    EXPECT_EQ(OH_NN_SUCCESS, BuildModel(innerModel));

    OH_NNModel* model = reinterpret_cast<OH_NNModel*>(&innerModel);
    OH_NNCompilation* nnCompilation = OH_NNCompilation_Construct(model);
    OH_NNExecutor* nnExecutor = OH_NNExecutor_Construct(nnCompilation);

    uint32_t outputIndex = 0;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateOutputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);
}

/*
 * @tc.name: executor_allocate_output_memory_005
 * @tc.desc: Verify the success of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_output_memory_005, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_AllocateOutputMemory executor_allocate_output_memory_005");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    size_t length = 9 * sizeof(float);

    OH_NN_Memory* ret = OH_NNExecutor_AllocateOutputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_allocate_output_memory_006
 * @tc.desc: Verify the success of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_output_memory_006, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_AllocateInputMemory executor_allocate_output_memory_006");
    size_t m_backendID {0};
    std::shared_ptr<MockIDevice> device = std::make_shared<MockIDevice>();

    std::shared_ptr<PreparedModel> m_preparedModel {nullptr};
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;

    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair1;
    std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType> pair2;
    std::shared_ptr<TensorDesc> tensorDesr = std::make_shared<TensorDesc>();
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesr->SetShape(ptr, dimensionCount);
    pair1.first = tensorDesr;
    pair2.first = tensorDesr;
    m_inputTensorDescs.emplace_back(pair1);
    m_inputTensorDescs.emplace_back(pair2);
    m_outputTensorDescs.emplace_back(pair1);
    m_outputTensorDescs.emplace_back(pair2);

    size_t length = 9 * sizeof(float);
    EXPECT_CALL(*((MockIDevice *) device.get()), AllocateTensorBuffer(length, m_outputTensorDescs[0].first))
        .WillRepeatedly(::testing::Return(reinterpret_cast<void*>(0x1000)));

    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, device, m_preparedModel, m_inputTensorDescs, m_outputTensorDescs);
    EXPECT_NE(nullptr, executor);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;

    OH_NN_Memory* ret = OH_NNExecutor_AllocateOutputMemory(nnExecutor, outputIndex, length);
    EXPECT_NE(nullptr, ret);

    testing::Mock::AllowLeak(device.get());
}

/*
 * @tc.name: executor_allocate_output_memory_007
 * @tc.desc: Verify the success of the OH_NNExecutor_AllocateInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_allocate_output_memory_007, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_AllocateInputMemory executor_allocate_output_memory_007");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    size_t length = 0;

    OH_NN_Memory* ret = OH_NNExecutor_AllocateOutputMemory(nnExecutor, outputIndex, length);
    EXPECT_EQ(nullptr, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_destroy_input_memory_001
 * @tc.desc: Verify the OH_NNExecutor is nullptr of the OH_NNExecutor_DestroyInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_input_memory_001, testing::ext::TestSize.Level0)
{
    InnerModel innerModel;
    BuildModel(innerModel);
    OH_NNExecutor* nnExecutor = nullptr;

    uint32_t inputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
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
    LOGE("OH_NNExecutor_DestroyInputMemory executor_destroy_input_memory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t inputIndex = 0;
    OH_NN_Memory** memory = nullptr;
    OH_NNExecutor_DestroyInputMemory(nnExecutor, inputIndex, memory);
    EXPECT_EQ(nullptr, memory);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_destroy_input_memory_003
 * @tc.desc: Verify the *memory is nullptr of the OH_NNExecutor_DestroyInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_input_memory_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_DestroyInputMemory executor_destroy_input_memory_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    EXPECT_NE(executor, nullptr);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t inputIndex = 0;
    OH_NN_Memory* memory = nullptr;
    OH_NN_Memory** pMemory = &memory;
    OH_NNExecutor_DestroyInputMemory(nnExecutor, inputIndex, pMemory);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_destroy_input_memory_004
 * @tc.desc: Verify the error happened when destroying input memory of the OH_NNExecutor_DestroyInputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_input_memory_004, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_DestroyInputMemory executor_destroy_input_memory_004");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t inputIndex = 6;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    OH_NNExecutor_DestroyInputMemory(nnExecutor, inputIndex, &pMemory);
    EXPECT_NE(nullptr, pMemory);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
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
    LOGE("OH_NNExecutor_DestroyOutputMemory executor_destroy_output_memory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    OH_NN_Memory** memory = nullptr;
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, memory);
    EXPECT_EQ(nullptr, memory);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_destroy_output_memory_003
 * @tc.desc: Verify the *memory is nullptr of the OH_NNExecutor_DestroyOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_output_memory_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_DestroyOutputMemory executor_destroy_output_memory_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    EXPECT_NE(executor, nullptr);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    OH_NN_Memory* memory = nullptr;
    OH_NN_Memory** pMemory = &memory;
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, pMemory);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_destroy_output_memory_004
 * @tc.desc: Verify the error happened when destroying output memory of the OH_NNExecutor_DestroyOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_output_memory_004, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_DestroyOutputMemory executor_destroy_output_memory_004");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 6;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, &pMemory);
    EXPECT_NE(nullptr, pMemory);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_destroy_output_memory_005
 * @tc.desc: Verify the success of the OH_NNExecutor_DestroyOutputMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_destroy_output_memory_005, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_DestroyOutputMemory executor_destroy_output_memory_005");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};
    OH_NN_Memory* pMemory = &memory;
    uint32_t outputIndex = 0;
    OH_NNExecutor_DestroyOutputMemory(nnExecutor, outputIndex, &pMemory);
    EXPECT_NE(nullptr, pMemory);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
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
    LOGE("OH_NNExecutor_SetInputWithMemory executor_set_input_with_memory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    OH_NN_Tensor* operand = nullptr;

    uint32_t inputIndex = 0;
    float dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 9 * sizeof(float)};

    OH_NN_ReturnCode ret = OH_NNExecutor_SetInputWithMemory(nnExecutor, inputIndex, operand, &memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_set_input_with_memory_003
 * @tc.desc: Verify the memory is nullptr of the OH_NNExecutor_SetInputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_input_with_memory_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetInputWithMemory executor_set_input_with_memory_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    SetTensor();

    uint32_t inputIndex = 0;
    OH_NN_Memory* memory = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetInputWithMemory(nnExecutor, inputIndex, &m_tensor, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_set_input_with_memory_004
 * @tc.desc: Verify the success of the OH_NNExecutor_SetInputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_input_with_memory_004, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetInputWithMemory executor_set_input_with_memory_004");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t inputIndex = 0;
    int32_t dims[2] = {3, 4};
    m_tensor = {OH_NN_FLOAT32, 2, dims, nullptr, OH_NN_TENSOR};

    float dataArry[12] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 12 * sizeof(float)};

    OH_NN_ReturnCode ret = OH_NNExecutor_SetInputWithMemory(nnExecutor, inputIndex, &m_tensor, &memory);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
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
    LOGE("OH_NNExecutor_SetOutputWithMemory executor_set_output_with_memory_002");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    OH_NN_Memory* memory = nullptr;
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOutputWithMemory(nnExecutor, outputIndex, memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
}

/*
 * @tc.name: executor_set_output_with_memory_003
 * @tc.desc: Verify the success of the OH_NNExecutor_SetOutputWithMemory function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, executor_set_output_with_memory_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNExecutor_SetOutputWithMemory executor_set_output_with_memory_003");
    size_t m_backendID {0};
    std::shared_ptr<Device> m_device {nullptr};
    std::shared_ptr<MockIPreparedModel> mockIPreparedMode = std::make_shared<MockIPreparedModel>();
    EXPECT_CALL(*((MockIPreparedModel *) mockIPreparedMode.get()), GetInputDimRanges(::testing::_, ::testing::_))
        .WillRepeatedly(::testing::Return(OH_NN_FAILED));
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_inputTensorDescs;
    std::vector<std::pair<std::shared_ptr<TensorDesc>, OH_NN_TensorType>> m_outputTensorDescs;
    NNExecutor* executor = new (std::nothrow) NNExecutor(
        m_backendID, m_device, mockIPreparedMode, m_inputTensorDescs, m_outputTensorDescs);
    OH_NNExecutor* nnExecutor = reinterpret_cast<OH_NNExecutor*>(executor);

    uint32_t outputIndex = 0;
    float dataArry[12] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
    void* const data = dataArry;
    OH_NN_Memory memory = {data, 12 * sizeof(float)};
    OH_NN_ReturnCode ret = OH_NNExecutor_SetOutputWithMemory(nnExecutor, outputIndex, &memory);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);

    testing::Mock::AllowLeak(mockIPreparedMode.get());
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
    size_t deviceID = 12345;
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
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/*
 * @tc.name: device_get_type_001
 * @tc.desc: Verify the device is nullptr of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, device_get_type_001, testing::ext::TestSize.Level0)
{
    size_t deviceID = 12345;
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
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
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
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_create_001
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_create_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_Create oh_nnquantparam_create_001");
    NN_QuantParam* ret = OH_NNQuantParam_Create();
    EXPECT_NE(nullptr, ret);
}

/*
 * @tc.name: oh_nnquantparam_setscales_001
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setscales_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetScales oh_nnquantparam_setscales_001");
    size_t quantNum = 1;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetScales(nullptr, nullptr, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setscales_002
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setscales_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetScales oh_nnquantparam_setscales_002");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    size_t quantNum = 1;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetScales(quantParams, nullptr, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setscales_003
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setscales_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetScales oh_nnquantparam_setscales_003");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    double scale = 2;
    size_t quantNum = 0;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetScales(quantParams, &scale, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setscales_004
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setscales_004, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetScales oh_nnquantparam_setscales_004");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    double scale = 2;
    size_t quantNum = 2;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetScales(quantParams, &scale, quantNum);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: oh_nnquantparam_setzeropoints_001
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setzeropoints_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetZeroPoints oh_nnquantparam_setzeropoints_001");
    size_t quantNum = 2;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetZeroPoints(nullptr, nullptr, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setzeropoints_002
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setzeropoints_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetZeroPoints oh_nnquantparam_setzeropoints_002");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    size_t quantNum = 2;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetZeroPoints(quantParams, nullptr, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setzeropoints_003
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setzeropoints_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetZeroPoints oh_nnquantparam_setzeropoints_003");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    int32_t zeroPoints = 2;
    size_t quantNum = 0;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetZeroPoints(quantParams, &zeroPoints, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setzeropoints_004
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setzeropoints_004, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetZeroPoints oh_nnquantparam_setzeropoints_004");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    int32_t zeroPoints = 2;
    size_t quantNum = 2;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetZeroPoints(quantParams, &zeroPoints, quantNum);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: oh_nnquantparam_setnumbits_001
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setnumbits_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetNumBits oh_nnquantparam_setnumbits_001");
    size_t quantNum = 2;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetNumBits(nullptr, nullptr, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setnumbits_002
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setnumbits_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetNumBits oh_nnquantparam_setnumbits_002");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    size_t quantNum = 2;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetNumBits(quantParams, nullptr, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setnumbits_003
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setnumbits_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetNumBits oh_nnquantparam_setnumbits_003");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    uint32_t zeroPoints = 2;
    size_t quantNum = 0;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetNumBits(quantParams, &zeroPoints, quantNum);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_setnumbits_004
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_setnumbits_004, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_SetNumBits oh_nnquantparam_setnumbits_004");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    uint32_t zeroPoints = 2;
    size_t quantNum = 2;
    OH_NN_ReturnCode ret = OH_NNQuantParam_SetNumBits(quantParams, &zeroPoints, quantNum);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: oh_nnquantparam_destroy_001
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_destroy_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_Destroy oh_nnquantparam_destroy_001");
    OH_NN_ReturnCode ret = OH_NNQuantParam_Destroy(nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_destroy_002
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_destroy_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_Destroy oh_nnquantparam_destroy_002");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    NN_QuantParam** quantParamsDex = &quantParams;
    *quantParamsDex = nullptr;
    OH_NN_ReturnCode ret = OH_NNQuantParam_Destroy(quantParamsDex);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnquantparam_destroy_003
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnquantparam_destroy_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNQuantParam_Destroy oh_nnquantparam_destroy_003");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    NN_QuantParam** quantParamsDex = &quantParams;
    OH_NN_ReturnCode ret = OH_NNQuantParam_Destroy(quantParamsDex);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: oh_nnmodel_addtensortomodel_001
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnmodel_addtensortomodel_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_AddTensorToModel oh_nnmodel_addtensortomodel_001");
    TensorDesc* tensorDescImpl = new (std::nothrow) TensorDesc();
    NN_TensorDesc* tensor = reinterpret_cast<NN_TensorDesc*>(tensorDescImpl);
    OH_NN_ReturnCode ret = OH_NNModel_AddTensorToModel(nullptr, tensor);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_addtensortomodel_002
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnmodel_addtensortomodel_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_AddTensorToModel oh_nnmodel_addtensortomodel_002");
    OH_NNModel* model = OH_NNModel_Construct();
    OH_NN_ReturnCode ret = OH_NNModel_AddTensorToModel(model, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_addtensortomodel_003
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnmodel_addtensortomodel_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_AddTensorToModel oh_nnmodel_addtensortomodel_003");
    OH_NNModel* model = OH_NNModel_Construct();
    TensorDesc* tensorDescImpl = new (std::nothrow) TensorDesc();
    NN_TensorDesc* tensor = reinterpret_cast<NN_TensorDesc*>(tensorDescImpl);
    OH_NN_ReturnCode ret = OH_NNModel_AddTensorToModel(model, tensor);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}

/*
 * @tc.name: oh_nnmodel_settensorquantparams_001
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnmodel_settensorquantparams_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetTensorQuantParams oh_nnmodel_settensorquantparams_001");
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    uint32_t index = 10;
    OH_NN_ReturnCode ret = OH_NNModel_SetTensorQuantParams(nullptr, index, quantParams);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_settensorquantparams_002
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnmodel_settensorquantparams_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetTensorQuantParams oh_nnmodel_settensorquantparams_002");
    OH_NNModel* model = OH_NNModel_Construct();
    uint32_t index = 10;
    OH_NN_ReturnCode ret = OH_NNModel_SetTensorQuantParams(model, index, nullptr);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_settensorquantparams_003
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnmodel_settensorquantparams_003, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetTensorQuantParams oh_nnmodel_settensorquantparams_003");
    OH_NNModel* model = OH_NNModel_Construct();
    NN_QuantParam* quantParams = OH_NNQuantParam_Create();
    uint32_t index = 10;
    OH_NN_ReturnCode ret = OH_NNModel_SetTensorQuantParams(model, index, quantParams);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_settensortype_001
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnmodel_settensortype_001, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetTensorType oh_nnmodel_settensortype_001");
    OH_NN_TensorType tensorType = OH_NN_REDUCE_MIN_KEEP_DIMS;
    uint32_t index = 10;
    OH_NN_ReturnCode ret = OH_NNModel_SetTensorType(nullptr, index, tensorType);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}

/*
 * @tc.name: oh_nnmodel_settensortype_002
 * @tc.desc: Verify the success of the OH_NNDevice_GetType function.
 * @tc.type: FUNC
 */
HWTEST_F(NeuralNetworkRuntimeTest, oh_nnmodel_settensortype_002, testing::ext::TestSize.Level0)
{
    LOGE("OH_NNModel_SetTensorType oh_nnmodel_settensortype_002");
    OH_NNModel* model = OH_NNModel_Construct();
    OH_NN_TensorType tensorType = OH_NN_REDUCE_MIN_COEFF;
    uint32_t index = 10;
    OH_NN_ReturnCode ret = OH_NNModel_SetTensorType(model, index, tensorType);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, ret);
}
} // namespace Unittest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
