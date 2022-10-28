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

#ifndef NEURAL_NETWORK_RUNTIME_INNER_H
#define NEURAL_NETWORK_RUNTIME_INNER_H

#include "interfaces/kits/c/neural_network_runtime_type.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 直接加载LiteGraph，完成模型搭建。
 *
 * 调用{@link OH_NNModel_Construct}创建模型实例后，直接调用本方法加载LiteGraph。加载LiteGraph后，只能调用
 * {@link OH_NNCompilation_Construct}创建模型编译器，或者调用{@link OH_NNModel_Destroy}销毁模型实例。\n
 *
 * 不允许本方法与{@link OH_NNModel_AddTensor}、{@link OH_NNModel_AddOperation}、
 * {@link OH_NNModel_SetTensorData}和{@link OH_NNModel_SpecifyInputsAndOutputs}
 * 等构图接口混用，否则返回{@link OH_NN_OPERATION_FORBIDDEN}错误。\n
 *
 * 如果本方法调用成功，返回{@link OH_NN_SUCCESS}，liteGraph将由NNRt管理，调用者无需释放，避免造成二次释放;
 * 如果方法返回其他错误码，则NNRt不会持有liteGraph，此时需要调用者主动释放内存。
 *
 *
 * 本接口不作为Neural Network Runtime接口对外开放。\n
 *
 * @param model 指向{@link OH_NNModel}实例的指针。
 * @param liteGraph 指向LiteGraph的指针。
 * @return 函数执行的结果状态，执行成功返回OH_NN_SUCCESS，失败返回具体错误码，参考{@link OH_NN_ReturnCode}。
 * @throw std::bad_alloc 本方法可能在转换原始指针到智能指针的过程中，抛出std::bad_alloc异常，此时liteGraph将被
 *        主动释放。
 * @since 9
 * @version 1.0
 */
OH_NN_ReturnCode OH_NNModel_BuildFromLiteGraph(OH_NNModel *model, const void *liteGraph);

#ifdef __cplusplus
}
#endif // __cpluscplus
#endif // NEURAL_NETWORK_RUNTIME_INNER_H