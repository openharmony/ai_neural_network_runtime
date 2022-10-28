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

#ifndef NEURAL_NETWORK_RUNTIME_H
#define NEURAL_NETWORK_RUNTIME_H
/**
 * @file neural_network_runtime.h
 *
 * @brief Neural Network Runtime部件接口定义，通过调用以下接口，在硬件加速器上执行深度学习模型推理计算。
 *
 * 注意：Neural Network Runtime的接口目前均不支持多线程调用。\n 
 *
 * @since 9
 * @version 1.0
 */
#include "neural_network_runtime_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @addtogroup NNModel
 * @{
 *
 * @brief Neural Network Runtime 构图模块，提供了一系列构图接口实现操作数的添加、算子的添加和输入输出的设置，帮助开发者完成
 *        AI模型的构建。
 *
 * @since 9
 * @version 1.0
 */

/**
 * @brief 创建{@link OH_NNModel}类型的模型实例，搭配OH_NNModel模块提供的其他接口，完成模型实例的构造。
 *
 * 在开始构图前，先调用{@link OH_NNModel_Construct}创建模型实例，根据模型的拓扑结构，调用
 * {@link OH_NNModel_AddTensor}、{@link OH_NNModel_AddOperation}和
 * {@link OH_NNModel_SetTensorData}方法，填充模型的数据节点和算子节点；然后调用
 * {@link OH_NNModel_SpecifyInputsAndOutputs}指定模型的输入和输出；当构造完模型的拓扑结构，调用
 * {@link OH_NNModel_Finish}完成模型的构建。\n 
 *
 * 模型实例使用完毕后，需要调用{@link OH_NNModel_Destroy}销毁模型实例，避免内存泄漏。\n 
 *
 * @return 返回一个指向{@link OH_NNModel}实例的指针。
 * @since 9
 * @version 1.0
 */
OH_NNModel *OH_NNModel_Construct(void);

/**
 * @brief 向模型实例中添加操作数
 *
 * Neural Network Runtime模型中的数据节点和算子参数均由模型的操作数构成。本方法根据tensor，向model实
 * 例中添加操作数。操作数添加的顺序是模型中记录操作数的索引值，{@link OH_NNModel_SetTensorData}、
 * {@link OH_NNModel_AddOperation}和{@link OH_NNModel_SpecifyInputsAndOutputs}
 * 方法根据该索引值，指定不同的操作数。\n 
 *
 * Neural Network Runtime支持动态形状输入和输出。在添加动态形状的数据节点时，需要将tensor.dimensions中支持动态
 * 变化的维度设置为-1。例如：一个4维tensor，将tensor.dimensions设置为[1, -1, 2, 2]，表示其第二个维度支持
 * 动态变化。\n 
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param tensor {@link OH_NN_Tensor}操作数的指针，tensor指定了添加到模型实例中操作数的属性。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_AddTensor(OH_NNModel *model, const OH_NN_Tensor *tensor);

/**
 * @brief 设置操作数的数值
 *
 * 对于具有常量值的操作数（如模型的权重），需要在构图阶段使用本方法设置数值。操作数的索引值根据操作数添加进模型的顺序决定，操作数的添加参考
 * {@link OH_NNModel_AddTensor}。\n 
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param index 操作数的索引值。
 * @param dataBuffer 指向真实数据的指针。
 * @param length 数据缓冲区的长度。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_SetTensorData(OH_NNModel *model, uint32_t index, const void *dataBuffer, size_t length);

/**
 * @brief 向模型实例中添加算子
 *
 * 本方法向模型实例中添加算子，算子类型由op指定，算子的参数、输入和输出由paramIndices、inputIndices和
 * outputIndices指定。本方法将对算子参数的属性和输入输出的数量进行校验，这些属性需要在调用
 * {@link OH_NNModel_AddTensor}添加操作数的时候正确设置。每个算子期望的参数、输入和输出属性请参考
 * {@link OH_NN_OperationType}。\n 
 *
 * paramIndices、inputIndices和outputIndices中存储的是操作数的索引值，每个索引值根据操作数添加进模型的顺序决定，正确
 * 设置并添加算子要求准确设置每个操作数的索引值。操作数的添加参考{@link OH_NNModel_AddTensor}。\n 
 *
 * 如果添加算子时，添加了额外的参数（非算子需要的参数），本方法返回{@link OH_NN_INVALID_PARAMETER}；如果没有设置算子参数，
 * 则算子按默认值设置缺省的参数，默认值请参考{@link OH_NN_OperationType}。\n 
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param op 指定添加的算子类型，取值请参考{@link OH_NN_OperationType}的枚举值。
 * @param paramIndices OH_NN_UInt32Array实例的指针，设置算子的参数。
 * @param inputIndices OH_NN_UInt32Array实例的指针，指定算子的输入。
 * @param outputIndices OH_NN_UInt32Array实例的指针，设置算子的输出。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_AddOperation(OH_NNModel *model,
                                         OH_NN_OperationType op,
                                         const OH_NN_UInt32Array *paramIndices,
                                         const OH_NN_UInt32Array *inputIndices,
                                         const OH_NN_UInt32Array *outputIndices);

/**
 * @brief 指定模型的输入输出
 *
 * 模型实例需要指定操作数作为端到端的输入和输出，设置为输入和输出的操作数不能使用{@link OH_NNModel_SetTensorData}设置
 * 数值，需要在执行阶段调用OH_NNExecutor的方法设置输入、输出数据。\n 
 *
 * 操作数的索引值根据操作数添加进模型的顺序决定，操作数的添加参考
 * {@link OH_NNModel_AddTensor}。\n 
 *
 * 暂时不支持异步设置模型输入输出。\n 
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param inputIndices OH_NN_UInt32Array实例的指针，指定算子的输入。
 * @param outputIndices OH_NN_UInt32Array实例的指针，指定算子的输出。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_SpecifyInputsAndOutputs(OH_NNModel *model,
                                                    const OH_NN_UInt32Array *inputIndices,
                                                    const OH_NN_UInt32Array *outputIndices);

/**
 * @brief 完成模型构图
 *
 * 完成模型拓扑结构的搭建后，调用本方法指示构图已完成。在调用本方法后，无法进行额外的构图操作，调用
 * {@link OH_NNModel_AddTensor}、{@link OH_NNModel_AddOperation}、
 * {@link OH_NNModel_SetTensorData}和
 * {@link OH_NNModel_SpecifyInputsAndOutputs}将返回
 * {@link OH_NN_OPERATION_FORBIDDEN}。\n 
 *
 * 在调用{@link OH_NNModel_GetAvailableOperations}和{@link OH_NNCompilation_Construct}
 * 之前，必须先调用本方法完成构图。\n 
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_Finish(OH_NNModel *model);

/**
 * @brief 释放模型实例。
 *
 * 调用{@link OH_NNModel_Construct}创建的模型实例需要调用本方法主动释放，否则将造成内存泄漏。\n 
 *
 * 如果model为空指针或者*model为空指针，本方法只打印warning日志，不执行释放逻辑。\n 
 *
 * @param model 指向{@link OH_NNModel}实例的二级指针。模型实例销毁后，本方法将*model主动设置为空指针。
 * @since 9
 * @version 1.0
 */
void OH_NNModel_Destroy(OH_NNModel **model);

/**
 * @brief 查询硬件对模型内所有算子的支持情况，通过布尔值序列指示支持情况。
 *
 * 查询底层硬件对模型实例内每个算子的支持情况，硬件由deviceID指定，结果将通过isSupported指向的数组表示。如果支持第i个算子，则
 * (*isSupported)[i] == true，否则为 false。\n 
 *
 * 本方法成功执行后，(*isSupported)将指向记录算子支持情况的bool数组，数组长度和模型实例的算子数量相等。该数组对应的内存由
 * Neural Network Runtime管理，在模型实例销毁或再次调用本方法后自动销毁。\n 
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param deviceID 指定查询的硬件ID，通过{@link OH_NNDevice_GetAllDevicesID}获取。
 * @param isSupported 指向bool数组的指针。调用本方法时，要求(*isSupported)为空指针，否则返回
 *                    {@link OH_NN_INVALID_PARAMETER}。
 * @param opCount 模型实例中算子的数量，对应(*isSupported)数组的长度。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_GetAvailableOperations(OH_NNModel *model,
                                                   size_t deviceID,
                                                   const bool **isSupported,
                                                   uint32_t *opCount);
/** @} */

/**
 * @addtogroup NNCompilation
 * @{
 *
 * @brief Neural Network Runtime 编译模块
 *
 * @since 9
 * @version 1.0
 */

/**
 * @brief 创建{@link OH_NNCompilation}类型的编译实例
 *
 * 使用OH_NNModel模块完成模型的构造后，借助OH_NNCompilation模块提供的接口，将模型传递到底层硬件完成编译。本方法接受一个
 * {@link OH_NNModel}实例，创建出{@link OH_NNCompilation}实例；通过
 * {@link OH_NNCompilation_SetDevice}方法，设置编译的设备，最后调用
 * {@link OH_NNCompilation_Build}完成编译。\n 
 *
 * 除了计算硬件的选择，OH_NNCompilation模块支持模型缓存、性能偏好、优先级设置、float16计算等特性，参考以下方法：
 * - {@link OH_NNCompilation_SetCache}
 * - {@link OH_NNCompilation_SetPerformanceMode}
 * - {@link OH_NNCompilation_SetPriority}
 * - {@link OH_NNCompilation_EnableFloat16}\n 
 *
 * 调用本方法创建{@link OH_NNCompilation}后，{@link OH_NNModel}实例可以释放。\n 
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @return 返回一个指向{@link OH_NNCompilation}实例的指针。
 * @since 9
 * @version 1.0
 */
OH_NNCompilation *OH_NNCompilation_Construct(const OH_NNModel *model);

/**
 * @brief 指定模型编译和计算的硬件。
 *
 * 编译阶段，需要指定模型编译和执行计算的硬件设备。先调用{@link OH_NNDevice_GetAllDevicesID}获取可用的设备ID，
 * 通过{@link OH_NNDevice_GetType}和{@link OH_NNDevice_GetType}获取设备信息后，将期望编译执行的
 * 设备ID传入本方法进行设置。\n 
 *
 * @param compilation 指向{@link OH_NNCompilation}实例的指针。
 * @param deviceID 指定的硬件ID。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_SetDevice(OH_NNCompilation *compilation, size_t deviceID);

/**
 * @brief 设置编译后的模型缓存路径和缓存版本。
 *
 * 在支持缓存的硬件上，模型在硬件驱动层编译后可以保存为缓存文件，下次编译时直接从缓存文件读取模型，减少重新编译的耗时。本方法接受缓存路径和版本，根据缓存
 * 路径中和版本的不同情况，本方法采取不同的行为：\n 
 *
 * - 缓存路径指定的目录下没有文件：
 * 将编译后的模型缓存到目录下，设置缓存版本等于version。\n 
 *
 * - 缓存路径指定的目录下存在完整的缓存文件，且版本号 == version：
 * 读取路径下的缓存文件，传递到底层硬件中转换为可以执行的模型实例。\n 
 *
 * - 缓存路径指定的目录下存在完整的缓存文件，但版本号 < version：
 * 路径下的缓存文件需要更新，模型在底层硬件完成编译后，覆写路径下的缓存文件，将版本号更新为version。\n 
 *
 * - 缓存路径指定的目录下存在完整的缓存文件，但版本号 > version：
 * 路径下的缓存文件版本高于version，不读取缓存文件，同时返回{@link OH_NN_INVALID_PARAMETER}错误码。\n 
 *
 * - 缓存路径指定的目录下的缓存文件不完整或没有缓存文件的访问权限：
 * 返回{@link OH_NN_INVALID_FILE}错误码。\n 
 *
 * - 缓存目录不存在，或者没有访问权限：
 * 返回{@link OH_NN_INVALID_PATH}错误码。\n 
 *
 * @param compilation 指向{@link OH_NNCompilation}实例的指针。
 * @param cachePath 模型缓存文件目录，本方法在cachePath目录下为不同的硬件创建缓存目录。建议每个模型使用单独的缓存目录。
 * @param version 缓存版本。
 * @return  函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_SetCache(OH_NNCompilation *compilation, const char *cachePath, uint32_t version);

/**
 * @brief 设置模型计算的性能模式。
 *
 * Neural Network Runtime 支持为模型计算设置性能模式，满足低功耗到极致性能的需求。如果编译阶段没有调用本方法设置性能模式，
 * 编译实例为模型默认分配{@link OH_NN_PERFORMANCE_NONE}模式。在{@link OH_NN_PERFORMANCE_NONE}
 * 模式下，硬件按默认的性能模式执行计算。\n 
 *
 * 在不支持性能模式设置的硬件上调用本方法，将返回{@link OH_NN_UNAVALIDABLE_DEVICE}错误码。\n 
 *
 * @param compilation 指向{@link OH_NNCompilation}实例的指针。
 * @param performanceMode 指定性能模式，可选的性能模式参考{@link OH_NN_PerformanceMode}。
 * @return  函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_SetPerformanceMode(OH_NNCompilation *compilation,
                                                     OH_NN_PerformanceMode performanceMode);

/**
 * @brief 设置模型计算的优先级。
 *
 * Neural Network Runtime 支持为模型设置计算优先级，优先级仅作用于相同uid进程创建的模型，不同uid进程、不同设备的优先级不会
 * 相互影响。\n 
 *
 * 在不支持优先级设置的硬件上调用本方法，将返回{@link OH_NN_UNAVALIDABLE_DEVICE}错误码。\n 
 *
 * @param compilation 指向{@link OH_NNCompilation}实例的指针。
 * @param priority 指定优先级，可选的优先级参考{@link OH_NN_Priority}。
 * @return  函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_SetPriority(OH_NNCompilation *compilation, OH_NN_Priority priority);

/**
 * @brief 是否以float16的浮点数精度计算。
 *
 * Neural Network Runtime目前仅支持构造float32浮点模型和int8量化模型。在支持float16精度的硬件上调用本方法，
 * float32浮点数精度的模型将以float16的精度执行计算，以减少内存占用和执行时间。\n 
 *
 * 在不支持float16精度计算的硬件上调用本方法，将返回{@link OH_NN_UNAVALIDABLE_DEVICE}错误码。\n 
 *
 * @param compilation 指向{@link OH_NNCompilation}实例的指针。
 * @param enableFloat16 Float16低精度计算标志位。设置为true时，执行Float16推理；设置为false时，执行float32推理。
 * @return  函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_EnableFloat16(OH_NNCompilation *compilation, bool enableFloat16);

/**
 * @brief 进行模型编译
 *
 * 完成编译配置后，调用本方法指示模型编译已完成。编译实例将模型和编译选项推送至硬件设备进行编译。在调用本方法后，无法进行额外的编译操作，调用
 * {@link OH_NNCompilation_SetDevice}、{@link OH_NNCompilation_SetCache}、
 * {@link OH_NNCompilation_SetPerformanceMode}、
 * {@link OH_NNCompilation_SetPriority}和{@link OH_NNCompilation_EnableFloat16}
 * 方法将返回{@link OH_NN_OPERATION_FORBIDDEN}。\n 
 *
 * @param compilation 指向{@link OH_NNCompilation}实例的指针。
 * @return  函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNCompilation_Build(OH_NNCompilation *compilation);

/**
 * @brief 释放Compilation对象。
 *
 * 调用{@link OH_NNCompilation_Construct}创建的编译实例需要调用本方法主动释放，否则将造成内存泄漏。\n 
 *
 * 如果compilation为空指针或者*compilation为空指针，本方法只打印warning日志，不执行释放逻辑。\n 
 *
 * @param compilation 指向{@link OH_NNCompilation}实例的二级指针。编译实例销毁后，本方法将*compilation主动设置为空指针。
 * @since 9
 * @version 1.0
 */
void OH_NNCompilation_Destroy(OH_NNCompilation **compilation);
/** @} */

/**
 * @addtogroup NNExecutor
 * @{
 *
 * @brief Neural Network Runtime 执行模块
 *
 * @since 9
 * @version 1.0
 */

/**
 * @brief 创建{@link OH_NNExecutor}类型的执行器实例
 *
 * 本方法接受一个编译器，构造一个与硬件关联的模型推理执行器。通过{@link OH_NNExecutor_SetInput}设置模型输入数据，
 * 设置输入数据后，调用{@link OH_NNExecutor_Run}方法执行推理，最后通过
 * {@link OH_NNExecutor_GetOutput}获取计算结果。\n 
 *
 * 调用本方法创建{@link OH_NNExecutor}实例后，如果不需要创建其他执行器，可以安全释放{@link OH_NNCompilation}实例。\n 
 *
 * @param compilation 指向{@link OH_NNCompilation}实例的指针。
 * @return 返回指向{@link OH_NNExecutor}实例的指针。
 * @since 9
 * @version 1.0
 */
OH_NNExecutor *OH_NNExecutor_Construct(OH_NNCompilation *compilation);

/**
 * @brief 设置模型单个输入的数据。
 *
 * 本方法将dataBuffer中，长度为length个字节的数据，拷贝到底层硬件的共享内存。inputIndex指定设置的输入，tensor用于设置输入的
 * 形状、类型、量化参数等信息。\n 
 *
 * 由于Neural Network Runtime支持动态输入形状的模型，在固定形状输入和动态形状输入的场景下，本方法采取不同的处理策略：
 *
 * - 固定形状输入的场景：tensor各属性必须和构图阶段调用{@link OH_NNModel_AddTensor}添加的操作数保持一致；
 * - 动态形状输入的场景：在构图阶段，由于动态输入的形状不确定，调用本方法时，要求tensor.dimensions中的每个值必须大于0，
 * 以确定执行计算阶段输入的形状。设置形状时，只允许调整数值为-1的维度。假设在构图阶段，输入A的维度为
 * [-1, 224, 224, 3]，调用本方法时，只能调整第一个维度的尺寸，如：[3, 224, 224, 3]。调整其他维度将返回
 * {@link OH_NN_INVALID_PARAMETER}。\n 
 *
 * inputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输入的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，inputIndices为{1，2，3}，
 * 则在执行阶段，三个输入的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @param inputIndex 输入的索引值。
 * @param tensor 设置输入数据对应的操作数。
 * @param dataBuffer 指向输入数据的指针。
 * @param length 数据缓冲区的字节长度。
 * @return  函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetInput(OH_NNExecutor *executor,
                                        uint32_t inputIndex,
                                        const OH_NN_Tensor *tensor,
                                        const void *dataBuffer,
                                        size_t length);

/**
 * @brief 设置模型单个输出的缓冲区。
 *
 * 本方法将dataBuffer指向的缓冲区与outputIndex指定的输出绑定，缓冲区的长度由length指定。\n 
 *
 * 调用{@link OH_NNExecutor_Run}完成单次模型推理后，Neural Network Runtime将比对dataBuffer指向的缓冲区与
 * 输出数据的长度，根据不同情况，返回不同结果：\n 
 *
 * - 如果缓冲区大于或等于数据长度：则推理后的结果将拷贝至缓冲区，并返回{@link OH_NN_SUCCESS}，可以通过访问dataBuffer读取推理结果。
 * - 如果缓冲区小于数据长度：则{@link OH_NNExecutor_Run}将返回{@link OH_NN_INVALID_PARAMETER}，
 * 并输出日志告知缓冲区太小的信息。\n 
 *
 * outputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输出的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，outputIndices为{4, 5, 6}，
 * 则在执行阶段，三个输出的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @param outputIndex 输出的索引值。
 * @param dataBuffer 指向输出数据的指针。
 * @param length 数据缓冲区的字节长度。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetOutput(OH_NNExecutor *executor,
                                         uint32_t outputIndex,
                                         void *dataBuffer,
                                         size_t length);

/**
 * @brief 获取输出tensor的维度信息。
 *
 * 调用{@link OH_NNExecutor_Run}完成单次推理后，本方法获取指定输出的维度信息和维数。在动态形状输入、输出的场景中常用。\n 
 *
 * outputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输出的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，outputIndices为{4, 5, 6}，
 * 则在执行阶段，三个输出的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @param outputIndex 输出的索引值。
 * @param shape 指向int32_t数组的指针，数组中的每个元素值，是输出tensor在每个维度上的长度。
 * @param length uint32_t类型的指针，返回输出的维数。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_GetOutputShape(OH_NNExecutor *executor,
                                              uint32_t outputIndex,
                                              int32_t **shape,
                                              uint32_t *shapeLength);

/**
 * @brief 执行推理。
 *
 * 在执行器关联的硬件上，执行模型的端到端推理计算。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_Run(OH_NNExecutor *executor);

/**
 * @brief 在硬件上为单个输入申请共享内存。
 *
 * Neural Network Runtime 提供主动申请硬件共享内存的方法。通过指定执行器和输入索引值，本方法在单个输入关联的硬件
 * 上，申请大小为length的共享内存，通过{@link OH_NN_Memory}实例返回。\n 
 *
 * inputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输入的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，inputIndices为{1，2，3}，
 * 则在执行阶段，三个输入的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @param inputIndex 输入的索引值。
 * @param length 申请的内存字节。
 * @return 指向{@link OH_NN_Memory}实例的指针。
 * @since 9
 * @version 1.0
 */
OH_NN_Memory *OH_NNExecutor_AllocateInputMemory(OH_NNExecutor *executor, uint32_t inputIndex, size_t length);

/**
 * @brief 在硬件上为单个输出申请共享内存。
 *
 * Neural Network Runtime 提供主动申请硬件共享内存的方法。通过指定执行器和输出索引值，本方法在单个输出关联的硬件
 * 上，申请大小为length的共享内存，通过{@link OH_NN_Memory}实例返回。\n 
 *
 * outputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输出的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，outputIndices为{4, 5, 6}，
 * 则在执行阶段，三个输出的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @param outputIndex 输出的索引值。
 * @param length 申请的内存字节。
 * @return 指向{@link OH_NN_Memory}实例的指针。
 * @since 9
 * @version 1.0
 */
OH_NN_Memory *OH_NNExecutor_AllocateOutputMemory(OH_NNExecutor *executor, uint32_t outputIndex, size_t length);

/**
 * @brief 释放{@link OH_NN_Memory}实例指向的输入内存。
 *
 * 调用{@link OH_NNExecutor_AllocateInputMemory}创建的内存实例，需要主动调用本方法进行释放，否则将造成内存泄漏。
 * inputIndex和memory的对应关系需要和创建内存实例时保持一致。\n 
 *
 * 如果memory或*memory为空指针，本方法只打印warning日志，不执行释放逻辑。\n 
 * 
 * inputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输入的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，inputIndices为{1，2，3}，
 * 则在执行阶段，三个输入的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @param inputIndex 输入的索引值。
 * @param memory 指向{@link OH_NN_Memory}实例的二级指针。共享内存销毁后，本方法将*memory主动设置为空指针。
 * @since 9
 * @version 1.0
 */
void OH_NNExecutor_DestroyInputMemory(OH_NNExecutor *executor, uint32_t inputIndex, OH_NN_Memory **memory);

/**
 * @brief 释放{@link OH_NN_Memory}实例指向的输出内存。
 *
 * 调用{@link OH_NNExecutor_AllocateOutputMemory}创建的内存实例，需要主动调用本方法进行释放，否则将造成内存泄漏。
 * outputIndex和memory的对应关系需要和创建内存实例时保持一致。\n 
 *
 * 如果memory或*memory为空指针，本方法只打印warning日志，不执行释放逻辑。\n 
 * 
 * outputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输出的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，outputIndices为{4, 5, 6}，
 * 则在执行阶段，三个输出的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @param outputIndex 输出的索引值。
 * @param memory 指向{@link OH_NN_Memory}实例的二级指针。共享内存销毁后，本方法将*memory主动设置为空指针。
 * @since 9
 * @version 1.0
 */
void OH_NNExecutor_DestroyOutputMemory(OH_NNExecutor *executor, uint32_t outputIndex, OH_NN_Memory **memory);

/**
 * @brief 将{@link OH_NN_Memory}实例指向的硬件共享内存，指定为单个输入使用的共享内存。
 *
 * 在需要自行管理内存的场景下，本方法将执行输入和{@link OH_NN_Memory}内存实例绑定。执行计算时，底层硬件从内存实例指向的共享内存中读取
 * 输入数据。通过本方法，可以实现设置输入、执行计算、读取输出的并发执行，提升数据流的推理效率。\n 
 * 
 * inputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输入的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，inputIndices为{1，2，3}，
 * 则在执行阶段，三个输入的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的指针。
 * @param inputIndex 输入的索引值。
 * @param tensor 指向{@link OH_NN_Tensor}的指针，设置单个输入所对应的操作数。
 * @param memory 指向{@link OH_NN_Memory}的指针。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetInputWithMemory(OH_NNExecutor *executor,
                                                  uint32_t inputIndex,
                                                  const OH_NN_Tensor *tensor,
                                                  const OH_NN_Memory *memory);

/**
 * @brief 将{@link OH_NN_Memory}实例指向的硬件共享内存，指定为单个输出使用的共享内存。
 *
 * 在需要自行管理内存的场景下，本方法将执行输出和{@link OH_NN_Memory}内存实例绑定。执行计算时，底层硬件将计算结果直接写入内存实例指向
 * 的共享内存。通过本方法，可以实现设置输入、执行计算、读取输出的并发执行，提升数据流的推理效率。\n 
 * 
 * outputIndex的值，从0开始，根据调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，指定输出的顺序，
 * 依次加一。假设调用{@link OH_NNModel_SpecifyInputsAndOutputs}时，outputIndices为{4, 5, 6}，
 * 则在执行阶段，三个输出的索引值分别为{0, 1, 2}。\n 
 *
 * @param executor 执行器。
 * @param outputIndex 输出的索引值。
 * @param memory 指向{@link OH_NN_Memory}的指针。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNExecutor_SetOutputWithMemory(OH_NNExecutor *executor,
                                                   uint32_t outputIndex,
                                                   const OH_NN_Memory *memory);

/**
 * @brief 销毁执行器实例，释放执行器占用的内存。
 *
 * 调用{@link OH_NNExecutor_Construct}创建的执行器实例需要调用本方法主动释放，否则将造成内存泄漏。\n 
 *
 * 如果executor为空指针或者*executor为空指针，本方法只打印warning日志，不执行释放逻辑。\n 
 *
 * @param executor 指向{@link OH_NNExecutor}实例的二级指针。
 * @since 9
 * @version 1.0
 */
void OH_NNExecutor_Destroy(OH_NNExecutor **executor);
/** @} */

/**
 * @addtogroup NNDevice
 * @{
 *
 * @brief Neural Network Runtime 设备管理模块
 *
 * @since 9
 * @version 1.0
 */

/**
 * @brief 获取对接到 Neural Network Runtime 的硬件ID。
 *
 * 每个硬件在 Neural Network Runtime 中存在唯一且固定ID，本方法通过uin32_t数组返回当前设备上已经对接的硬件ID。\n 
 *
 * 硬件ID通过size_t数组返回，数组的每个元素是单个硬件的ID值。数组内存由Neural Network Runtime管理。在下次调用本方法前，
 * 数据指针有效。\n 
 *
 * @param allDevicesID 指向size_t数组的指针。要求传入的(*allDevicesID)为空指针，否则返回
 *                     {@link OH_NN_INVALID_PARAMETER}。
 * @param deviceCount uint32_t类型的指针，用于返回(*allDevicesID)的长度。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNDevice_GetAllDevicesID(const size_t **allDevicesID, uint32_t *deviceCount);

/**
 * @brief 获取指定硬件的类型信息。
 *
 * 通过deviceID指定计算硬件，获取硬件的名称。硬件ID需要调用{@link OH_NNDevice_GetAllDevicesID}获取。\n 
 *
 * @param deviceID 指定硬件ID。
 * @param name 指向char数组的指针，要求传入的(*char)为空指针，否则返回
 *             {@link OH_NN_INVALID_PARAMETER}。（*name）以C风格字符串保存硬件名称，数组以\0结尾。
 * @return 函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNDevice_GetName(size_t deviceID, const char **name);

/**
 * @brief 获取指定硬件的类别信息。
 *
 * 通过deviceID指定计算硬件，获取硬件的类别。目前 Neural Network Runtime 支持的设备类型有：
 * - CPU设备：OH_NN_CPU
 * - GPU设备：OH_NN_GPU
 * - 机器学习专用加速器：OH_NN_ACCELERATOR
 * - 不属于以上类型的其他硬件类型：OH_NN_OTHERS\n 
 *
 * @param deviceID 指定硬件ID。
 * @param deviceType 指向{@link OH_NN_DeviceType}实例的指针，返回硬件的类别信息。
 * @return  函数执行的结果状态。执行成功返回OH_NN_SUCCESS；失败返回具体错误码，具体失败错误码可参考{@link OH_NN_ReturnCode}。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNDevice_GetType(size_t deviceID, OH_NN_DeviceType *deviceType);
/** @} */

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // NEURAL_NETWORK_RUNTIME_H
