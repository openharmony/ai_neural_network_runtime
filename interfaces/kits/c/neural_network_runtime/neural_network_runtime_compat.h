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
 * @since 11
 * @version 1.0
 */

/**
 * @file neural_network_runtime_compat.h
 *
 * @brief The APIs defined in this document are all compatible APIs of the previous version, which will be deprecated after five versions. 
 *        It is recommended to use the new APIs defined by neural_network_core.h and neural_network_runtime.h.
 * Note: Currently, the APIs of Neural Network Runtime do not support multi-thread calling. \n
 *
 * @library libneural_network_runtime.so
 * @since 11
 * @version 1.0
 */

#ifndef NEURAL_NETWORK_RUNTIME_COMPAT_H
#define NEURAL_NETWORK_RUNTIME_COMPAT_H

#include "neural_network_runtime_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Adds a tensor to a model instance.
 *
 * The data node and operator parameters in the Neural Network Runtime model are composed of tensors of the model. 
 * This method is used to add tensors to a model instance based on the <b>tensor</b> parameter. 
 * The sequence of adding tensors is specified by the index value recorded in the model. The {@link OH_NNModel_SetTensorData}, {@link OH_NNModel_AddOperation}, 
 * and {@link OH_NNModel_SpecifyInputsAndOutputs} methods specifies tensors based on the index value. \n
 *
 * Neural Network Runtime supports inputs and outputs of the dynamic shape. When adding a data node with a dynamic shape, 
 * you need to set the dimensions that support dynamic changes in <b>tensor.dimensions</b> to <b>-1</b>. 
 * For example, if <b>tensor.dimensions</b> of a four-dimensional tensor is set to <b>[1, -1, 2, 2]</b>, the second dimension supports dynamic changes. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param tensor Pointer to the {@link OH_NN_Tensor} tensor. The tensor specifies the attributes of the tensor added to the model instance. 
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @deprecated since 11
 * @useinstead {@link OH_NNModel_AddTensorToModel}
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_AddTensor(OH_NNModel *model, const OH_NN_Tensor *tensor);

/**
 * @brief Sets the single input data for a model.
 *
 * This method copies the data whose length is specified by <b>length</b> (in bytes) in <b>dataBuffer</b> to the shared memory 
 * of the underlying device. <b>inputIndex</b> specifies the input to be set and <b>tensor</b> sets information such as the input shape, 
 * type, and quantization parameters. \n
 *
 * Neural Network Runtime supports models with dynamical shape input. For fixed shape input and dynamic shape input scenarios, 
 * this method uses different processing policies. \n
 *
 * - Fixed shape input: The attributes of <b>tensor</b> must be the same as those of the tensor added by calling 
 *   {@link OH_NNModel_AddTensor} in the composition phase.
 * - Dynamic shape input: In the composition phase, because the shape is not fixed, each value in <b>tensor.dimensions</b> must be greater than 
 *   <b>0</b> in the method calls to determine the shape input in the calculation phase. When setting the shape, you can modify 
 *   only the dimension whose value is <b>-1</b>. Assume that <b>[-1, 224, 224, 3]</b> is input as the the dimension of A in the composition phase. 
 *   When this method is called, only the size of the first dimension can be modified, for example, to <b>[3, 224, 224, 3]</b>. 
 *   If other dimensions are adjusted, {@link OH_NN_INVALID_PARAMETER} is returned. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param inputIndex Input index value, which is in the same sequence of the data input when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                   Assume that the value of <b>inputIndices</b> is <b>{1, 5, 9}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                   In input settings, the index value for the three inputs is <b>{0, 1, 2}</b>. \n
 *
 * @param tensor Sets the tensor corresponding to the input data.
 * @param dataBuffer Pointer to the input data.
 * @param length Length of the data buffer, in bytes.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails, 
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @deprecated since 11
 * @useinstead {@link OH_NNExecutor_RunSync}
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetInput(OH_NNExecutor *executor,
                                        uint32_t inputIndex,
                                        const OH_NN_Tensor *tensor,
                                        const void *dataBuffer,
                                        size_t length);

/**
 * @brief Sets the buffer for a single output of a model.
 *
 * This method binds the buffer to which <b>dataBuffer</b> points to the output specified by <b>outputIndex</b>. 
 * The length of the buffer is specified by <b>length</b>. \n
 *
 * After {@link OH_NNExecutor_Run} is called to complete a single model inference, Neural Network Runtime compares 
 * the length of the buffer to which <b>dataBuffer</b> points with the length of the output data and returns different results 
 * based on the actual situation. \n
 *
 * - If the buffer length is greater than or equal to the data length, the inference result is copied to the buffer and 
 *   {@link OH_NN_SUCCESS} is returned. You can read the inference result from <b>dataBuffer</b>. 
 * - If the buffer length is smaller than the data length, {@link OH_NNExecutor_Run} returns {@link OH_NN_INVALID_PARAMETER} 
 *   and generates a log indicating that the buffer is too small. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param outputIndex Output Index value, which is in the same sequence of the data output when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                    Assume that the value of <b>outputIndices</b> is <b>{4, 6, 8}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} 
 *                    is called. In output buffer settings, the index value for the three outputs is <b>{0, 1, 2}</b>.
 * @param dataBuffer Pointer to the output data.
 * @param length Length of the data buffer, in bytes.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails, 
 *                an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @deprecated since 11
 * @useinstead {@link OH_NNExecutor_RunSync}
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetOutput(OH_NNExecutor *executor,
                                         uint32_t outputIndex,
                                         void *dataBuffer,
                                         size_t length);

/**
 * @brief Performs inference.
 *
 * Performs end-to-end inference and computing of the model on the device associated with the executor. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails, 
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @deprecated since 11
 * @useinstead {@link OH_NNExecutor_RunSync}
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_Run(OH_NNExecutor *executor);

/**
 * @brief Allocates shared memory to a single input on a device.
 *
 * Neural Network Runtime provides a method for proactively allocating shared memory on a device. By specifying the executor and input index value, 
 * this method allocates shared memory whose size is specified by <b>length</b> on the device associated with a single input and returns the 
 * operation result through the {@link OH_NN_Memory} instance. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param inputIndex Input index value, which is in the same sequence of the data input when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                   Assume that the value of <b>inputIndices</b> is <b>{1, 5, 9}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                   In the memory input application, the index value for the three inputs is <b>{0, 1, 2}</b>.
 * @param length Memory size to be applied for, in bytes.
 * @return Pointer to a {@link OH_NN_Memory} instance.
 * @deprecated since 11
 * @useinstead {@link OH_NNTensor_CreateWithSize}
 * @since 9
 * @version 1.0
 */
OH_NN_Memory *OH_NNExecutor_AllocateInputMemory(OH_NNExecutor *executor, uint32_t inputIndex, size_t length);

/**
 * @brief Allocates shared memory to a single output on a device.
 *
 * Neural Network Runtime provides a method for proactively allocating shared memory on a device. By specifying the executor and 
 * output index value, this method allocates shared memory whose size is specified by <b>length</b> on the device associated with 
 * a single output and returns the operation result through the {@link OH_NN_Memory} instance. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param outputIndex Output Index value, which is in the same sequence of the data output when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                    Assume that the value of <b>outputIndices</b> is <b>{4, 6, 8}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                    In output memory application, the index value for the three outputs is <b>{0, 1, 2}</b>.
 * @param length Memory size to be applied for, in bytes.
 * @return Pointer to a {@link OH_NN_Memory} instance.
 * @deprecated since 11
 * @useinstead {@link OH_NNTensor_CreateWithSize}
 * @since 9
 * @version 1.0
 */
OH_NN_Memory *OH_NNExecutor_AllocateOutputMemory(OH_NNExecutor *executor, uint32_t outputIndex, size_t length);

/**
 * @brief Releases the input memory to which the {@link OH_NN_Memory} instance points.
 *
 * This method needs to be called to release the memory instance created by calling {@link OH_NNExecutor_AllocateInputMemory}. 
 * Otherwise, memory leak will occur. 
 * The mapping between <b>inputIndex</b> and <b>memory</b> must be the same as that in memory instance creation. \n
 *
 * If <b>memory</b> or <b>*memory</b> is a null pointer, this method only prints warning logs and does not execute the release logic. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param inputIndex Input index value, which is in the same sequence of the data input when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                   Assume that the value of <b>inputIndices</b> is <b>{1, 5, 9}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                   In memory input release, the index value for the three inputs is <b>{0, 1, 2}</b>.
 * @param memory Double pointer to the {@link OH_NN_Memory} instance. After shared memory is destroyed, this method sets <b>*memory</b> to a null pointer.
 * @deprecated since 11
 * @useinstead {@link OH_NNTensor_Destroy}
 * @since 9
 * @version 1.0
 */
void OH_NNExecutor_DestroyInputMemory(OH_NNExecutor *executor, uint32_t inputIndex, OH_NN_Memory **memory);

/**
 * @brief Releases the output memory to which the {@link OH_NN_Memory} instance points.
 *
 * This method needs to be called to release the memory instance created by calling {@link OH_NNExecutor_AllocateOutputMemory}. Otherwise, memory leak will occur. 
 * The mapping between <b>outputIndex</b> and <b>memory</b> must be the same as that in memory instance creation. \n
 *
 * If <b>memory</b> or <b>*memory</b> is a null pointer, this method only prints warning logs and does not execute the release logic. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param outputIndex Output Index value, which is in the same sequence of the data output when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                    Assume that the value of <b>outputIndices</b> is <b>{4, 6, 8}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                    In output memory release, the index value for the three outputs is <b>{0, 1, 2}</b>.
 * @param memory Double pointer to the {@link OH_NN_Memory} instance. After shared memory is destroyed, this method sets <b>*memory</b> to a null pointer.
 * @deprecated since 11
 * @useinstead {@link OH_NNTensor_Destroy} 
 * @since 9
 * @version 1.0
 */
void OH_NNExecutor_DestroyOutputMemory(OH_NNExecutor *executor, uint32_t outputIndex, OH_NN_Memory **memory);

/**
 * @brief Specifies the hardware shared memory pointed to by the {@link OH_NN_Memory} instance as the shared memory used by a single input.
 *
 * In scenarios where memory needs to be managed by yourself, this method binds the execution input to the {@link OH_NN_Memory} memory instance. 
 * During computing, the underlying device reads the input data from the shared memory pointed to by the memory instance. 
 * By using this method, concurrent execution of input setting, computing, and read can be implemented to improve inference efficiency of a data flow. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param inputIndex Input index value, which is in the same sequence of the data input when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                   Assume that the value of <b>inputIndices</b> is <b>{1, 5, 9}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                   When the input shared memory is specified, the index value for the three inputs is <b>{0, 1, 2}</b>.
 * @param tensor Pointer to {@link OH_NN_Tensor}, used to set the tensor corresponding to a single input.
 * @param memory Pointer to {@link OH_NN_Memory}.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails, 
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @deprecated since 11
 * @useinstead {@link OH_NNExecutor_RunSync}
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetInputWithMemory(OH_NNExecutor *executor,
                                                  uint32_t inputIndex,
                                                  const OH_NN_Tensor *tensor,
                                                  const OH_NN_Memory *memory);

/**
 * @brief Specifies the hardware shared memory pointed to by the {@link OH_NN_Memory} instance as the shared memory used by a single output.
 *
 * In scenarios where memory needs to be managed by yourself, this method binds the execution output to the {@link OH_NN_Memory} memory instance. 
 * When computing is performed, the underlying hardware directly writes the computing result to the shared memory to which the memory instance points. 
 * By using this method, concurrent execution of input setting, computing, and read can be implemented to improve inference efficiency of a data flow. \n
 *
 * @param executor Executor.
 * @param outputIndex Output Index value, which is in the same sequence of the data output when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                    Assume that the value of <b>outputIndices</b> is <b>{4, 6, 8}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} is called. 
 *                    When output shared memory is specified, the index value for the three outputs is <b>{0, 1, 2}</b>.
 * @param memory Pointer to {@link OH_NN_Memory}.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails, 
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @deprecated since 11
 * @useinstead {@link OH_NNExecutor_RunSync}
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetOutputWithMemory(OH_NNExecutor *executor,
                                                   uint32_t outputIndex,
                                                   const OH_NN_Memory *memory);

#ifdef __cplusplus
}
#endif // __cplusplus

/** @} */
#endif // NEURAL_NETWORK_RUNTIME_COMPAT_H
