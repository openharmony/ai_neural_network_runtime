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

/**
 * @addtogroup NeuralNeworkRuntime
 * @{
 *
 * @brief Provides APIs of Neural Network Runtime for accelerating the model inference.
 *
 * @Syscap SystemCapability.Ai.NeuralNetworkRuntime
 * @since 9
 * @version 1.0
 */

/**
 * @file neural_network_runtime.h
 *
 * @brief Defines the Neural Network Runtime APIs. The AI inference framework uses the Native APIs provided by Neural Network Runtime
 *        to construct and compile models and perform inference and computing on acceleration hardware.
 * Note: Currently, the APIs of Neural Network Runtime do not support multi-thread calling. \n
 *
 * @since 9
 * @version 1.0
 */

#ifndef NEURAL_NETWORK_RUNTIME_H
#define NEURAL_NETWORK_RUNTIME_H

#include "neural_network_runtime_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Creates a model instance of the {@link OH_NNModel} type and uses other APIs provided by OH_NNModel to construct the model instance.
 *
 * Before composition, call {@link OH_NNModel_Construct} to create a model instance. Based on the model topology, 
 * call the {@link OH_NNModel_AddTensor}, {@link OH_NNModel_AddOperation}, and {@link OH_NNModel_SetTensorData} methods 
 * to fill in the data and operator nodes of the model, and then call {@link OH_NNModel_SpecifyInputsAndOutputs} to specify the inputs and outputs of the model.
 * After the model topology is constructed, call {@link OH_NNModel_Finish} to build the model. \n
 *
 * After a model instance is used, you need to destroy it by calling {@link OH_NNModel_Destroy} to avoid memory leak. \n
 *
 * @return Returns the pointer to a {@link OH_NNModel} instance.
 * @since 9
 * @version 1.0
 */
OH_NNModel *OH_NNModel_Construct(void);

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
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_AddTensor(OH_NNModel *model, const OH_NN_Tensor *tensor);

/**
 * @brief Sets the tensor value.
 *
 * For tensors with constant values (such as model weights), you need to use this method in the composition phase.
 * The index value of a tensor is determined by the sequence in which the tensor is added to the model. 
 * For details about how to add a tensor, see {@link OH_NNModel_AddTensor}. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @param index Index value of a tensor.
 * @param dataBuffer Pointer to real data.
 * @param length Length of the data buffer.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_SetTensorData(OH_NNModel *model, uint32_t index, const void *dataBuffer, size_t length);

/**
 * @brief Adds an operator to a model instance.
 *
 * This method is used to add an operator to a model instance. The operator type is specified by <b>op</b>, and 
 * the operator parameters, inputs, and outputs are specified by <b>paramIndices</b>, <b>inputIndices</b>, and <b>outputIndices</b> respectively.
 * This method verifies the attributes of operator parameters and the number of input and output parameters. 
 * These attributes must be correctly set when {@link OH_NNModel_AddTensor} is called to add tensors.
 * For details about the expected parameters, input attributes, and output attributes of each operator, see {@link OH_NN_OperationType}. \n
 *
 * <b>paramIndices</b>, <b>inputIndices</b>, and <b>outputIndices</b> store index values of tensors. 
 * Index values are determined by the sequence in which tensors are added to the model. 
 * For details about how to add a tensor, see {@link OH_NNModel_AddTensor}. \n
 *
 * If unnecessary parameters are added for adding an operator, this method returns {@link OH_NN_INVALID_PARAMETER}. 
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
 * using {@link OH_NNModel_SetTensorData}. The <b>OH_NNExecutor</b> method needs to be called in the execution phase to set the input and output data. \n
 *
 * The index value of a tensor is determined by the sequence in which the tensor is added to the model. 
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
 * If <b>model</b> or <b>*model</b> is a null pointer, this method only prints warning logs and does not execute the release logic. \n
 *
 * @param model Level-2 pointer to the {@link OH_NNModel} instance. After a model instance is destroyed, this method sets <b>*model</b> to a null pointer.
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
 *                                      Otherwise, {@link OH_NN_INVALID_PARAMETER} is returned.
 *                    
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


/**
 * @brief Creates a compilation instance of the {@link OH_NNCompilation} type.
 *
 * After the OH_NNModel module completes model construction, APIs provided by the OH_NNCompilation module pass the model 
 * to underlying device for compilation. This method creates a {@link OH_NNCompilation} instance 
 * based on the passed {@link OH_NNModel} instance. The {@link OH_NNCompilation_SetDevice} method is called  
 * to set the device to compile on, and {@link OH_NNCompilation_Build} is then called to complete compilation.\n
 *
 * In addition to computing device selection, the OH_NNCompilation module supports features such as model caching, performance preference, 
 * priority setting, and float16 computing, which can be implemented by the following methods:
 * - {@link OH_NNCompilation_SetCache}
 * - {@link OH_NNCompilation_SetPerformanceMode}
 * - {@link OH_NNCompilation_SetPriority}
 * - {@link OH_NNCompilation_EnableFloat16} \n
 *
 * After {@link OH_NNCompilation} is created by calling this method, the {@link OH_NNModel} instance can be released. \n
 *
 * @param model Pointer to the {@link OH_NNModel} instance.
 * @return Returns the pointer to a {@link OH_NNCompilation} instance.
 * @since 9
 * @version 1.0
 */
OH_NNCompilation *OH_NNCompilation_Construct(const OH_NNModel *model);

/**
 * @brief Specifies the device for model compilation and computing.
 *
 * In the compilation phase, you need to specify the device for model compilation and computing. Call {@link OH_NNDevice_GetAllDevicesID} 
 * to obtain available device IDs. Call {@link OH_NNDevice_GetType} and {@link OH_NNDevice_GetName} to obtain device information 
 * and pass target device IDs to this method for setting. \n
 *
 * @param compilation Pointer to the {@link OH_NNCompilation} instance.
 * @param deviceID Device ID.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation *compilation, size_t deviceID);

/**
 * @brief Set the cache directory and version of the compiled model.
 *
 * On the device that supports caching, a model can be saved as a cache file after being compiled at the device driver layer. 
 * The model can be directly read from the cache file in the next compilation, saving recompilation time. 
 * This method performs different operations based on the passed cache directory and version:\n
 *
 * - No file exists in the cache directory:
 * Caches the compiled model to the directory and sets the cache version to <b>version</b>. \n
 *
 * - A complete cache file exists in the cache directory, and its version is <b>version</b>:
 * Reads the cache file in the path and passes the data to the underlying device for conversion into executable model instances. \n
 *
 * - A complete cache file exists in the cache directory, and its version is earlier than <b>version</b>:
 * When model compilation is complete on the underlying device, overwrites the cache file and changes the version number to <b>version</b>. \n
 *
 * - A complete cache file exists in the cache directory, and its version is later than <b>version</b>:
 * Returns the {@link OH_NN_INVALID_PARAMETER} error code without reading the cache file. \n
 *
 * - The cache file in the cache directory is incomplete or you do not have the permission to access the cache file.
 * Returns the {@link OH_NN_INVALID_FILE} error code. \n
 *
 * - The cache directory does not exist or you do not have the access permission.
 * Returns the {@link OH_NN_INVALID_PATH} error code. \n
 *
 * @param compilation Pointer to the {@link OH_NNCompilation} instance.
 * @param cachePath Directory for storing model cache files. This method creates directories for different devices in the <b>cachePath</b> directory. 
 *                  You are advised to use a separate cache directory for each model.
 * @param version Cache version.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails,
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_SetCache(OH_NNCompilation *compilation, const char *cachePath, uint32_t version);

/**
 * @brief Sets the performance mode for model computing.
 *
 * Neural Network Runtime allows you to set the performance mode for model computing to meet the requirements of low power consumption
 * and ultimate performance. If this method is not called to set the performance mode in the compilation phase, the compilation instance assigns
 * the {@link OH_NN_PERFORMANCE_NONE} mode for the model by default. In this case, the device performs computing in the default performance mode. \n
 *
 * If this method is called on the device that does not support the setting of the performance mode, the {@link OH_NN_UNAVALIDABLE_DEVICE} error code is returned. \n
 *
 * @param compilation Pointer to the {@link OH_NNCompilation} instance.
 * @param performanceMode Performance mode. For details about the available performance modes, see {@link OH_NN_PerformanceMode}.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails,
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_SetPerformanceMode(OH_NNCompilation *compilation,
                                                     OH_NN_PerformanceMode performanceMode);

/**
 * @brief Sets the model computing priority.
 *
 * Neural Network Runtime allows you to set computing priorities for models.  
 * The priorities apply only to models created by the process with the same UID.
 * The settings will not affect models created by processes with different UIDs on different devices. \n
 *
 * If this method is called on the device that does not support the priority setting, the {@link OH_NN_UNAVALIDABLE_DEVICE} error code is returned. \n
 *
 * @param compilation Pointer to the {@link OH_NNCompilation} instance.
 * @param priority Priority. For details about the optional priorities, see {@link OH_NN_Priority}.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails, 
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_SetPriority(OH_NNCompilation *compilation, OH_NN_Priority priority);

/**
 * @brief Enables float16 for computing.
 *
 * Currently, Neural Network Runtime supports only float32 and int8. If this method is called on a device that supports float16, 
 * float16 will be used for computing the float32 model to reduce memory usage and execution time. \n
 *
 * If this method is called on the device that does not support float16, the {@link OH_NN_UNAVALIDABLE_DEVICE} error code is returned. \n
 *
 * @param compilation Pointer to the {@link OH_NNCompilation} instance.
 * @param enableFloat16 Indicates whether to enable float16. If this parameter is set to <b>true</b>, float16 inference is performed. 
 *                      If this parameter is set to <b>false</b>, float32 inference is performed.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails, an error code is returned.
 *         For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_EnableFloat16(OH_NNCompilation *compilation, bool enableFloat16);

/**
 * @brief Compiles a model.
 *
 * After the compilation configuration is complete, call this method to return the compilation result. The compilation instance pushes the model and
 * compilation options to the device for compilation. After this method is called, additional compilation operations cannot be performed. 
 * If the {@link OH_NNCompilation_SetDevice}, {@link OH_NNCompilation_SetCache}, {@link OH_NNCompilation_SetPerformanceMode}, 
 * {@link OH_NNCompilation_SetPriority}, and {@link OH_NNCompilation_EnableFloat16} methods are called, {@link OH_NN_OPERATION_FORBIDDEN} is returned. \n
 *
 * @param compilation Pointer to the {@link OH_NNCompilation} instance.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation *compilation);

/**
 * @brief Releases the <b>Compilation</b> object.
 *
 * This method needs to be called to release the compilation instance created by calling {@link OH_NNCompilation_Construct}. Otherwise, memory leak will occur. \n
 *
 * If <b>compilation</b> or <b>*compilation</b> is a null pointer, this method only prints warning logs and does not execute the release logic. \n
 *
 * @param compilation Level-2 pointer to the {@link OH_NNCompilation} instance. After a compilation instance is destroyed, 
 *                    this method sets <b>*compilation</b> to a null pointer.
 * @since 9
 * @version 1.0
 */
void OH_NNCompilation_Destroy(OH_NNCompilation **compilation);


/**
 * @brief Creates an executor instance of the {@link OH_NNExecutor} type.
 *
 * This method constructs a model inference executor associated with the device based on the passed compiler. Use {@link OH_NNExecutor_SetInput}
 * to set the model input data. After the input data is set, call {@link OH_NNExecutor_Run} to perform inference and then call 
 * {@link OH_NNExecutor_SetOutput} to obtain the computing result. \n
 *
 * After calling this method to create the {@link OH_NNExecutor} instance, you can release the {@link OH_NNCompilation} 
 * instance if you do not need to create any other executors. \n
 *
 * @param compilation Pointer to the {@link OH_NNCompilation} instance.
 * @return Pointer to a {@link OH_NNExecutor} instance.
 * @since 9
 * @version 1.0
 */
OH_NNExecutor *OH_NNExecutor_Construct(OH_NNCompilation *compilation);

/**
 * @brief Sets the single input data for a model.
 *
 * This method copies the data whose length is specified by <b>length</b> (in bytes) in <b>dataBuffer</b> to the shared memory 
 * of the underlying device. <b>inputIndex</b> specifies the input to be set and <b>tensor</b> sets information such as the input shape,
 * type, and quantization parameters. \n
 *
 * Neural Network Runtime supports models with dynamical shape input. For fixed shape input and dynamic shape input scenarios, 
 * this method uses different processing policies.
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
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetOutput(OH_NNExecutor *executor,
                                         uint32_t outputIndex,
                                         void *dataBuffer,
                                         size_t length);

/**
 * @brief Obtains the dimension information about the output tensor.
 *
 * After {@link OH_NNExecutor_Run} is called to complete a single inference, call this method to obtain the specified output dimension
 * information and number of dimensions. It is commonly used in dynamic shape input and output scenarios. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @param outputIndex Output Index value, which is in the same sequence of the data output when {@link OH_NNModel_SpecifyInputsAndOutputs} is called.
 *                                      Assume that <b>outputIndices</b> is <b>{4, 6, 8}</b> when {@link OH_NNModel_SpecifyInputsAndOutputs} is called.
 *                                      When {@link OH_NNExecutor_GetOutputShape} is called to obtain dimension information about the output tensor, 
 *                                      <b>outputIndices</b> is <b>{0, 1, 2}</b>.
 *
 * @param shape Pointer to the int32_t array. The value of each element in the array is the length of the output tensor in each dimension.
 * @param shapeLength Pointer to the uint32_t type. The number of output dimensions is returned.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails, 
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_GetOutputShape(OH_NNExecutor *executor,
                                              uint32_t outputIndex,
                                              int32_t **shape,
                                              uint32_t *shapeLength);

/**
 * @brief Performs inference.
 *
 * Performs end-to-end inference and computing of the model on the device associated with the executor. \n
 *
 * @param executor Pointer to the {@link OH_NNExecutor} instance.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails,
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
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
 * @param memory Level-2 pointer to the {@link OH_NN_Memory} instance. After shared memory is destroyed, this method sets <b>*memory</b> to a null pointer.
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
 * @param memory Level-2 pointer to the {@link OH_NN_Memory} instance. After shared memory is destroyed, this method sets <b>*memory</b> to a null pointer.
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
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetOutputWithMemory(OH_NNExecutor *executor,
                                                   uint32_t outputIndex,
                                                   const OH_NN_Memory *memory);

/**
 * @brief Destroys an executor instance to release the memory occupied by the executor.
 *
 * This method needs to be called to release the executor instance created by calling {@link OH_NNExecutor_Construct}. Otherwise,
 * memory leak will occur. \n
 *
 * If <b>executor</b> or <b>*executor</b> is a null pointer, this method only prints warning logs and does not execute the release logic. \n
 *
 * @param executor Level-2 pointer to the {@link OH_NNExecutor} instance.
 * @since 9
 * @version 1.0
 */
void OH_NNExecutor_Destroy(OH_NNExecutor **executor);


/**
 * @brief Obtains the ID of the device connected to Neural Network Runtime.
 *
 * Each device has a unique and fixed ID in Neural Network Runtime. This method returns device IDs on the current device through the uint32_t array. \n
 *
 * Device IDs are returned through the size_t array. Each element of the array is the ID of a single device. 
 * The array memory is managed by Neural Network Runtime. 
 * The data pointer is valid before this method is called next time. \n
 *
 * @param allDevicesID Pointer to the size_t array. The input <b>*allDevicesID</b> must be a null pointer. Otherwise, {@link OH_NN_INVALID_PARAMETER} is returned.
 * @param deviceCount Pointer of the uint32_t type, which is used to return the length of <b>(*allDevicesID)</b>.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. 
 *         If the operation fails, an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t **allDevicesID, uint32_t *deviceCount);

/**
 * @brief Obtains the name of the specified device.
 *
 * <b>deviceID</b> specifies the device whose name will be obtained. The device ID needs to be obtained by calling {@link OH_NNDevice_GetAllDevicesID}. \n
 *
 * @param deviceID Device ID.
 * @param name Pointer to the char array. The passed <b>(*char)</b> must be a null pointer. Otherwise, {@link OH_NN_INVALID_PARAMETER} is returned.
 *             The value of <b>(*name)</b> is a C-style string ended with <b>'\0'</b>.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails,
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNDevice_GetName(size_t deviceID, const char **name);

/**
 * @brief Obtains the type information of the specified device.
 *
 * <b>deviceID</b> specifies the device whose type will be obtained. Currently, Neural Network Runtime supports the following device types:
 * - <b>OH_NN_CPU</b>: CPU device.
 * - <b>OH_NN_GPU</b>: GPU device.
 * - <b>OH_NN_ACCELERATOR</b>: machine learning dedicated accelerator.
 * - <b>OH_NN_OTHERS</b>: other hardware types. \n
 *
 * @param deviceID Device ID.
 * @param deviceType Pointer to the {@link OH_NN_DeviceType} instance. The device type information is returned.
 * @return Execution result of the function. If the operation is successful, <b>OH_NN_SUCCESS</b> is returned. If the operation fails,
 *         an error code is returned. For details about the error codes, see {@link OH_NN_ReturnCode}.
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNDevice_GetType(size_t deviceID, OH_NN_DeviceType *deviceType);

#ifdef __cplusplus
}
#endif // __cplusplus

/** @} */
#endif // NEURAL_NETWORK_RUNTIME_H
