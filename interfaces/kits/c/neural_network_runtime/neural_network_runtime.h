/*
 * Copyright (c) 2022-2023 Huawei Device Co., Ltd.
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

/**
 * @addtogroup NeuralNeworkRuntime
 * @{
 *
 * @brief Provides APIs of Neural Network Runtime for accelerating the model inference.
 *
 * @Syscap SystemCapability.Ai.NeuralNetworkRuntime
 * @since 9
 * @version 2.0
 */

/**
 * @file neural_network_runtime.h
 *
 * @brief Defines the Neural Network Runtime APIs. The AI inference framework uses the Native APIs provided by Neural Network Runtime
 *        to construct models.
 * Note: Currently, the APIs of Neural Network Runtime do not support multi-thread calling. \n
 *
 * @library libneural_network_runtime.so
 * @since 9
 * @version 2.0
 */

#ifndef NEURAL_NETWORK_RUNTIME_H
#define NEURAL_NETWORK_RUNTIME_H

#include "neural_network_runtime_type.h"
#include "neural_network_core.h"
#include "neural_network_runtime_compat.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a {@link NN_QuantParam} instance.
 *
 * After the NN_QuantParam instance is created, call {@link OH_NNQuantParam_SetScales}, {@link OH_NNQuantParam_SetZeroPoints}, 
 * {@link OH_NNQuantParam_SetNumBits} to set its attributes, and then call {@link OH_NNModel_SetTensorQuantParams} to set it 
 * to a tensor. After that you should destroy it by calling {@link OH_NNQuantParam_Destroy} to avoid memory leak. \n
 *
 * @return Pointer to a {@link NN_QuantParam} instance.
 * @since 11
 * @version 1.0
 */
NN_QuantParam *OH_NNQuantParam_Create();

/**
 * @brief Sets the scales of the quantization parameter instance.
 *
 * The parameter <b>quantCount<\b> is the number of quantization parameters of a tensor, e.g. the quantCount is the channel 
 * count if the tensor is per-channel quantized. \n
 *
 * @param quantParams Pointer to the {@link NN_QuantParam} instance.
 * @param scales An array of scales for all quantization parameters of the tensor.
 * @param quantCount Number of quantization parameters of the tensor.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNQuantParam_SetScales(NN_QuantParam *quantParams, const double *scales, size_t quantCount);

/**
 * @brief Sets the zero points of the quantization parameter instance.
 *
 * The parameter <b>quantCount<\b> is the number of quantization parameters of a tensor, e.g. the quantCount is the channel 
 * count if the tensor is per-channel quantized. \n
 *
 * @param quantParams Pointer to the {@link NN_QuantParam} instance.
 * @param zeroPoints An array of zero points for all quantization parameters of the tensor.
 * @param quantCount Number of quantization parameters of the tensor.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNQuantParam_SetZeroPoints(NN_QuantParam *quantParams, const int32_t *zeroPoints, size_t quantCount);

/**
 * @brief Sets the number bits of the quantization parameter instance.
 *
 * The parameter <b>quantCount<\b> is the number of quantization parameters of a tensor, e.g. the quantCount is the channel 
 * count if the tensor is per-channel quantized. \n
 *
 * @param quantParams Pointer to the {@link NN_QuantParam} instance.
 * @param numBits An array of number bits for all quantization parameters of the tensor.
 * @param quantCount Number of quantization parameters of the tensor.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNQuantParam_SetNumBits(NN_QuantParam *quantParams, const uint32_t *numBits, size_t quantCount);

/**
 * @brief Releases a quantization parameter instance.
 *
 * The {@link NN_QuantParam} instance needs to be released to avoid memory leak after it is set to a {@link NN_TensorDesc}. 
 * If <b>quantParams</b> or <b>*quantParams</b> is a null pointer, this method only prints warning logs and does not 
 * execute the release. \n
 *
 * @param quantParams Double pointer to the {@link NN_QuantParam} instance.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNQuantParam_Destroy(NN_QuantParam **quantParams);

/**
 * @brief Creates a model instance of the {@link OH_NNModel} type and uses other APIs provided by OH_NNModel to construct the model instance.
 *
 * Before composition, call {@link OH_NNModel_Construct} to create a model instance. Based on the model topology, 
 * call the {@link OH_NNModel_AddTensor}, {@link OH_NNModel_AddOperation}, and {@link OH_NNModel_SetTensorData} methods 
 * to fill in the data and operator nodes of the model, and then call {@link OH_NNModel_SpecifyInputsAndOutputs} to specify the inputs and outputs of the model. 
 * After the model topology is constructed, call {@link OH_NNModel_Finish} to build the model. \n
 *
 * After a model instance is no longer used, you need to destroy it by calling {@link OH_NNModel_Destroy} to avoid memory leak. \n
 *
 * @return Pointer to a {@link OH_NNModel} instance.
 * @since 9
 * @version 1.0
 */
OH_NNModel *OH_NNModel_Construct(void);

/**
 * @brief Adds a tensor to the model instance.
 *
 * The data node and operator parameters in the Neural Network Runtime model are composed of tensors of the model. 
 * This method is used to add tensors to a model instance based on the <b>tensorDesc</b> parameter with type of {@link NN_TensorDesc}. 
 * {@link NN_TensorDesc} contains some attributes such as shape, format, data type and provides corresponding APIs to access them. 
 * The order of adding tensors is specified by the indices recorded in the model. The {@link OH_NNModel_SetTensorData}, {@link OH_NNModel_AddOperation}, 
 * and {@link OH_NNModel_SpecifyInputsAndOutputs} methods specify tensors based on the indices. \n
 *
 * Neural Network Runtime supports inputs and outputs of the dynamic shape. When adding a data node with a dynamic shape, 
 * you need to set the dimensions that support dynamic changes to <b>-1</b>. 
 * For example, if the shape of a four-dimensional tensor is set to <b>[1, -1, 2, 2]</b>, the second dimension supports dynamic changes. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param tensorDesc Pointer to the {@link NN_TensorDesc} instance. The tensor descriptor specifies the attributes of the tensor added to the model instance.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_AddTensorToModel(OH_NNModel *model, const NN_TensorDesc *tensorDesc);

/**
 * @brief Sets the tensor value.
 *
 * For tensors with constant values (such as model weights), you need to use this method to set their data. 
 * The index of a tensor is determined by the order in which the tensor is added to the model. 
 * For details about how to add a tensor, see {@link OH_NNModel_AddTensor}. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param index Index of a tensor.
 * @param dataBuffer Pointer to real data.
 * @param length Length of the data buffer.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_SetTensorData(OH_NNModel *model, uint32_t index, const void *dataBuffer, size_t length);

/**
 * @brief Sets the quantization parameter of a tensor.
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param index Index of a tensor.
 * @param quantParam Pointer to the quantization parameter instance.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_SetTensorQuantParams(OH_NNModel *model, uint32_t index, NN_QuantParam *quantParam);

/**
 * @brief Sets the tensor type. See {@link OH_NN_TensorType} for details.
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param index Index of a tensor.
 * @param tensorType Tensor type of {@link OH_NN_TensorType}.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 11
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_SetTensorType(OH_NNModel *model, uint32_t index, OH_NN_TensorType tensorType);

/**
 * @brief Adds an operator to a model instance.
 *
 * This method is used to add an operator to a model instance. The operator type is specified by <b>op</b>, and 
 * the operator parameters, inputs, and outputs are specified by <b>paramIndices</b>, <b>inputIndices</b>, and <b>outputIndices</b> respectively. 
 * This method verifies the attributes of operator parameters and the number of input and output parameters. 
 * These attributes must be correctly set when {@link OH_NNModel_AddTensor} is called to add tensors. 
 * For details about the expected parameters, input attributes, and output attributes of each operator, see {@link OH_NN_OperationType}. \n
 *
 * <b>paramIndices</b>, <b>inputIndices</b>, and <b>outputIndices</b> store the indices of tensors. 
 * The indices are determined by the order in which tensors are added to the model. 
 * For details about how to add a tensor, see {@link OH_NNModel_AddTensor}. \n
 *
 * If unnecessary parameters are added when adding an operator, this method returns {@link OH_NN_INVALID_PARAMETER}. 
 * If no operator parameter is set, the operator uses the default parameter value. 
 * For details about the default values, see {@link OH_NN_OperationType}. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param op Specifies the type of an operator to be added. For details, see the enumerated values of {@link OH_NN_OperationType}.
 * @param paramIndices Pointer to the <b>OH_NN_UInt32Array</b> instance, which is used to set operator parameters.
 * @param inputIndices Pointer to the <b>OH_NN_UInt32Array</b> instance, which is used to set the operator input.
 * @param outputIndices Pointer to the <b>OH_NN_UInt32Array</b> instance, which is used to set the operator output.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_AddOperation(OH_NNModel *model,
                                         OH_NN_OperationType op,
                                         const OH_NN_UInt32Array *paramIndices,
                                         const OH_NN_UInt32Array *inputIndices,
                                         const OH_NN_UInt32Array *outputIndices);

/**
 * @brief Specifies the inputs and outputs of a model.
 *
 * A tensor must be specified as the end-to-end inputs and outputs of a model instance. This type of tensor cannot be set 
 * using {@link OH_NNModel_SetTensorData}. \n
 *
 * The index of a tensor is determined by the order in which the tensor is added to the model. 
 * For details about how to add a tensor, see {@link OH_NNModel_AddTensor}. \n
 *
 * Currently, the model inputs and outputs cannot be set asynchronously. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param inputIndices Pointer to the <b>OH_NN_UInt32Array</b> instance, which is used to set the operator input.
 * @param outputIndices Pointer to the <b>OH_NN_UInt32Array</b> instance, which is used to set the operator output.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_SpecifyInputsAndOutputs(OH_NNModel *model,
                                                    const OH_NN_UInt32Array *inputIndices,
                                                    const OH_NN_UInt32Array *outputIndices);

/**
 * @brief Completes model composition.
 *
 * After the model topology is set up, call this method to indicate that the composition is complete. After this method is called, 
 * additional composition operations cannot be performed. If {@link OH_NNModel_AddTensor}, {@link OH_NNModel_AddOperation}, 
 * {@link OH_NNModel_SetTensorData}, and {@link OH_NNModel_SpecifyInputsAndOutputs} are called, 
 * {@link OH_NN_OPERATION_FORBIDDEN} is returned. \n
 *
 * Before calling {@link OH_NNModel_GetAvailableOperations} and {@link OH_NNCompilation_Construct}, 
 * you must call this method to complete composition. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_Finish(OH_NNModel *model);

/**
 * @brief Releases a model instance.
 *
 * This method needs to be called to release the model instance created by calling {@link OH_NNModel_Construct}. Otherwise, memory leak will occur. \n
 *
 * If <b>model</b> or <b>*model</b> is a null pointer, this method only prints warning logs and does not execute the release. \n
 *
 * @param model Double pointer to the {@link OH_NNModel} instance. After a model instance is destroyed, this method sets <b>*model</b> to a null pointer.
 * @since 9
 * @version 1.0
 */
void OH_NNModel_Destroy(OH_NNModel **model);

/**
 * @brief Queries whether the device supports operators in the model. The support status is indicated by the Boolean value.
 *
 * Queries whether underlying device supports operators in a model instance. The device is specified by <b>deviceID</b>, 
 * and the result is represented by the array pointed by <b>isSupported</b>. If the <i>i</i>th operator is supported, 
 * the value of <b>(*isSupported)</b>[<i>i</i>] is <b>true</b>. Otherwise, the value is <b>false</b>. \n
 *
 * After this method is successfully executed, <b>(*isSupported)</b> points to the bool array that records the operator support status. 
 * The operator quantity for the array length is the same as that for the model instance. The memory corresponding to this array is 
 * managed by Neural Network Runtime and is automatically destroyed after the model instance is destroyed or this method is called again. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param deviceID Device ID to be queried, which can be obtained by using {@link OH_NNDevice_GetAllDevicesID}.
 * @param isSupported Pointer to the bool array. When this method is called, <b>(*isSupported)</b> must be a null pointer. 
 *                    Otherwise, {@link OH_NN_INVALID_PARAMETER} is returned.
 * @param opCount Number of operators in a model instance, corresponding to the length of the <b>(*isSupported)</b> array.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_GetAvailableOperations(OH_NNModel *model,
                                                   size_t deviceID,
                                                   const bool **isSupported,
                                                   uint32_t *opCount);

#ifdef __cplusplus
}
#endif // __cplusplus

/** @} */
#endif // NEURAL_NETWORK_RUNTIME_H
