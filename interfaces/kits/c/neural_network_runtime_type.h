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

#ifndef NEURAL_NETWORK_RUNTIME_TYPE_H
#define NEURAL_NETWORK_RUNTIME_TYPE_H
/**
 * @file neural_network_runtime_type.h
 *
 * @brief Neural Network Runtime定义的结构体和枚举值。
 *
 * @since 9
 * @version 1.0
 */
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Neural Network Runtime的模型句柄
 *
 * @since 9
 * @version 1.0
 */
typedef struct OH_NNModel OH_NNModel;

/**
 * @brief Neural Network Runtime的编译器句柄
 *
 * @since 9
 * @version 1.0
 */
typedef struct OH_NNCompilation OH_NNCompilation;

/**
 * @brief Neural Network Runtime的执行器句柄
 *
 * @since 9
 * @version 1.0
 */
typedef struct OH_NNExecutor OH_NNExecutor;

/**
 * @brief 硬件的执行性能模式
 *
 * @since 9
 * @version 1.0
 */
typedef enum {
    /** 无性能模式偏好 */
    OH_NN_PERFORMANCE_NONE = 0,
    /** 低能耗模式 */
    OH_NN_PERFORMANCE_LOW = 1,
    /** 中性能模式 */
    OH_NN_PERFORMANCE_MEDIUM = 2,
    /** 高性能模式 */
    OH_NN_PERFORMANCE_HIGH = 3,
    /** 极致性能模式 */
    OH_NN_PERFORMANCE_EXTREME = 4
} OH_NN_PerformanceMode;

/**
 * @brief 模型推理任务优先级
 *
 * @since 9
 * @version 1.0
 */
typedef enum {
    /** 无优先级偏好 */
    OH_NN_PRIORITY_NONE = 0,
    /** 低优先级 */
    OH_NN_PRIORITY_LOW = 1,
    /** 中优先级 */
    OH_NN_PRIORITY_MEDIUM = 2,
    /** 高优先级 */
    OH_NN_PRIORITY_HIGH = 3
} OH_NN_Priority;

/**
 * @brief Neural Network Runtime 定义的错误码类型
 *
 * @since 9
 * @version 1.0
 */
typedef enum {
    /** 操作成功 */
    OH_NN_SUCCESS = 0,
    /** 操作失败 */
    OH_NN_FAILED = 1,
    /** 非法参数 */
    OH_NN_INVALID_PARAMETER = 2,
    /** 内存相关的错误，包括：内存不足、内存数据拷贝失败、内存申请失败等。 */
    OH_NN_MEMORY_ERROR = 3,
    /** 非法操作 */
    OH_NN_OPERATION_FORBIDDEN = 4,
    /** 空指针异常 */
    OH_NN_NULL_PTR = 5,
    /** 无效文件 */
    OH_NN_INVALID_FILE = 6,
    /** 硬件发生错误，错误可能包含：HDL服务崩溃 */
    OH_NN_UNAVALIDABLE_DEVICE = 7,
    /** 非法路径 */
    OH_NN_INVALID_PATH = 8
} OH_NN_ReturnCode;

/**
 * @brief Neural Network Runtime 融合算子中激活函数的类型
 *
 * @since 9
 * @version 1.0
 */
typedef enum : int8_t {
    /** 未指定融合激活函数 */
    OH_NN_FUSED_NONE = 0,
    /** 融合relu激活函数 */
    OH_NN_FUSED_RELU = 1,
    /** 融合relu6激活函数 */
    OH_NN_FUSED_RELU6 = 2
} OH_NN_FuseType;

/**
 * @brief tensor数据的排布类型
 *
 * @since 9
 * @version 1.0
 */
typedef enum {
    /** 当tensor没有特定的排布类型时（如标量或矢量），使用{@link OH_NN_FORMAT_NONE} */
    OH_NN_FORMAT_NONE = 0,
    /** 读取（使用）维度信息时按照NCHW读取（使用）*/
    OH_NN_FORMAT_NCHW = 1,
    /** 读取（使用）维度信息时按照NHWC读取（使用） */
    OH_NN_FORMAT_NHWC = 2
} OH_NN_Format;

/**
 * @brief Neural Network Runtime 支持的设备类型
 *
 * @since 9
 * @version 1.0
 */
typedef enum {
    /** 不属于CPU、GPU、专用加速器的设备 */
    OH_NN_OTHERS = 0,
    /** CPU设备 */
    OH_NN_CPU = 1,
    /** GPU设备 */
    OH_NN_GPU = 2,
    /** 专用硬件加速器 */
    OH_NN_ACCELERATOR = 3,
} OH_NN_DeviceType;

/**
 * @brief Neural Network Runtime 支持的数据类型
 *
 * @since 9
 * @version 1.0
 */
typedef enum {
    /** 操作数数据类型未知 */
    OH_NN_UNKNOWN = 0,
    /** 操作数数据类型为bool */
    OH_NN_BOOL = 1,
    /** 操作数数据类型为int8 */
    OH_NN_INT8 = 2,
    /** 操作数数据类型为int16 */
    OH_NN_INT16 = 3,
    /** 操作数数据类型为int32 */
    OH_NN_INT32 = 4,
    /** 操作数数据类型为int64 */
    OH_NN_INT64 = 5,
    /** 操作数数据类型为uint8 */
    OH_NN_UINT8 = 6,
    /** 操作数数据类型为uint16 */
    OH_NN_UINT16 = 7,
    /** 操作数数据类型为uint32 */
    OH_NN_UINT32 = 8,
    /** 操作数数据类型为uint64 */
    OH_NN_UINT64 = 9,
    /** 操作数数据类型为float16 */
    OH_NN_FLOAT16 = 10,
    /** 操作数数据类型为float32 */
    OH_NN_FLOAT32 = 11,
    /** 操作数数据类型为float64 */
    OH_NN_FLOAT64 = 12
} OH_NN_DataType;


/**
 * @brief Neural Network Runtime 支持算子的类型
 *
 * @since 9
 * @version 1.0
 */
typedef enum {
    /**
     * 返回两个输入张量对应元素相加的和的张量。
     *
     * 输入：
     *
     * * x，第一个输入的张量，数据类型要求为布尔值或者数字。
     * * y，第二个输入的张量，数据类型和形状需要和第一个输入保持一致。
     *
     * 参数：
     *
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * 0 输出x和y的和，数据形状与输入broadcast之后一样，数据类型与较高精度的输入精度一致
     */
    OH_NN_OPS_ADD = 1,

    /**
     * 在输入tensor上应用 2D 平均池化，仅支持NHWC格式的tensor。支持int8量化输入。
     *
     * 如果输入中含有padMode参数：
     *
     * 输入：
     *
     * * x，一个张量。
     *
     * 参数：
     *
     * * kernelSize，用来取平均值的kernel大小，是一个长度为2的int数组[kernel_height，kernel_weight]，
     *      第一个数表示kernel高度，第二个数表示kernel宽度。
     * * strides，kernel移动的距离，是一个长度为2的int数组[stride_height，stride_weight]，
     *      第一个数表示高度上的移动步幅，第二个数表示宽度上的移动步幅。
     * * padMode，填充模式，int类型的可选值，0表示same，1表示valid，并且以最近邻的值填充。
     *      same，输出的高度和宽度与x相同，填充总数将在水平和垂直方向计算，并在可能的情况下均匀分布到顶部
     *      和底部、左侧和右侧。否则，最后一个额外的填充将从底部和右侧完成。
     *      valid，输出的可能最大高度和宽度将在不填充的情况下返回。额外的像素将被丢弃。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 如果输入中含有padList参数：
     *
     * 输入：
     *
     * * x，一个张量。
     *
     * 参数：
     *
     * * kernelSize，用来取平均值的kernel大小，是一个长度为2的int数组[kernel_height，kernel_weight]，
     *      第一个数表示kernel高度，第二个数表示kernel宽度。
     * * strides，kernel移动的距离，是一个长度为2的int数组[stride_height，stride_weight]，
     *      第一个数表示高度上的移动步幅，第二个数表示宽度上的移动步幅。
     * * padList，输入x周围的填充，是一个长度为4的int数组[top，bottom，left，right]，并且以最近邻的值填充。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * 输出x平均池化后的张量。
     */
    OH_NN_OPS_AVG_POOL = 2,

    /**
     * 对一个tensor进行batch normalization，对tensor元素进行缩放和位移，缓解一批数据中潜在的covariate shift。
     *
     * 输入：
     *
     * * x，一个n维的tensor，要求形状为[N，...，C]，即第n维是通道数（channel）。
     * * scale，缩放因子的1D张量，用于缩放归一化的第一个张量。
     * * offset，用于偏移的1D张量，以移动到归一化的第一个张量。
     * * mean，总体均值的一维张量，仅用于推理；对于训练，必须为空。
     * * variance，用于总体方差的一维张量。仅用于推理；对于训练，必须为空。
     *
     * 参数：
     *
     * * epsilon，数值稳定性的小附加值。
     *
     * 输出：
     *
     * * 输出张量，形状和数据类型与输入x一致。
     */
    OH_NN_OPS_BATCH_NORM = 3,

    /**
     * 将一个4维tensor的batch维度按block_shape切分成小块，并将这些小块拼接到空间维度。
     *
     * 参数：
     *
     * * x，输入张量，维将被切分，拼接回空间维度。
     *
     * 输出：
     *
     * * blockSize，一个长度为2的数组[height_block，weight_block]，指定切分到空间维度上的block大小。
     * * crops，一个shape为(2，2)的2维数组[[crop0_start，crop0_end]，[crop1_start，crop1_end]]，
     *      表示在output的空间维度上截掉部分元素。
     *
     * 输出：
     *
     * * 输出张量，假设x的形状为(n，h，w，c)，output的形状为（n'，h'，w'，c'）：
     *      n' = n / (block_shape[0] * block_shape[1])
     *      h' = h * block_shape[0] - crops[0][0] - crops[0][1]
     *      w' = w * block_shape[1] - crops[1][0] - crops[1][1]
     *      c'= c
     */
    OH_NN_OPS_BATCH_TO_SPACE_ND = 4,

    /**
     * 对给出的输入张量上的各个维度方向上的数据进行偏置。
     *
     * 输入：
     *
     * * x，输入张量，可为2-5维度。
     * * bias，参数对应输入维度数量的偏移值。
     *
     * 输出：
     *
     * * 输出张量，根据输入中每个维度方向偏移后的结果。
     */
    OH_NN_OPS_BIAS_ADD = 5,

    /**
     * 对输入张量中的数据类型进行转换。
     *
     * 输入：
     *
     * * x，输入张量。
     * * type，输入转换目的的数据类型。
     *
     * 输出：
     *
     * * 输出张量，输出转换为目的数据类型后的张量。
     */
    OH_NN_OPS_CAST = 6,

    /**
     * 在指定轴上连接张量，将输入张量按给定的轴连接起来。
     *
     * 输入：
     *
     * * x：N个输入张量。
     *
     * 参数：
     *
     * * axis，指定轴的位置。
     *
     * 输出：
     *
     * * 输出n个张量按axis轴连接的结果。
     */
    OH_NN_OPS_CONCAT = 7,

    /**
     * 二维卷积层。
     *
     * 如果输入中含有padMode参数：
     *
     * 输入：
     *
     * * x，输入张量。
     * * weight，卷积的权重，要求weight排布为[outChannel，kernelHeight，kernelWidth，inChannel/group]，
     *      inChannel必须要能整除group。
     * * bias，卷积的偏置，是长度为[outChannel]的数组。在量化场景下，bias 参数不需要量化参数，其量化
     *      版本要求输入 OH_NN_INT32 类型数据，实际量化参数由 x 和 weight 共同决定。
     *
     * 参数：
     *
     * * stride，卷积核在height和weight上的步幅，是一个长度为2的int数组[strideHeight，strideWidth]。
     * * dilation，表示扩张卷积在height和weight上的扩张率，是一个长度为2的int数组[dilationHeight，dilationWidth]，
     *      值必须大于或等于1，并且不能超过x的height和width。
     * * padMode，x的填充模式，支持same和valid，int类型，0表示same，1表示valid。
     *      same，输出的高度和宽度与x相同，填充总数将在水平和垂直方向计算，并在可能的情况下均匀分布到顶部和底部、左侧
     *      和右侧。否则，最后一个额外的填充将从底部和右侧完成。
     *      Valid，输出的可能最大高度和宽度将在不填充的情况下返回。额外的像素将被丢弃。
     * * group，将输入x按in_channel分组，int类型。group等于1，这是常规卷积；group大于1且小于或等于in_channel，这是分组卷积。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     *
     * 如果输入中含有padList参数：
     *
     * 输入：
     *
     * * x，输入张量。
     * * weight，卷积的权重，要求weight排布为[outChannel，kernelHeight，kernelWidth，inChannel/group]，
     *      inChannel必须要能整除group。
     * * bias，卷积的偏置，是长度为[outChannel]的数组。在量化场景下，bias 参数不需要量化参数，其量化
     *      版本要求输入 OH_NN_INT32 类型数据，实际量化参数由 x 和 weight 共同决定。
     *
     * 参数：
     *
     * * stride，卷积核在height和weight上的步幅，是一个长度为2的int数组[strideHeight，strideWidth]。
     * * dilation，表示扩张卷积在height和weight上的扩张率，是一个长度为2的int数组[dilationHeight，dilationWidth]。
     *      值必须大于或等于1，并且不能超过x的height和width。
     * * padList，输入x周围的填充，是一个长度为4的int数组[top，bottom，left，right]。
     * * group，将输入x按in_channel分组，int类型。
     *      group等于1，这是常规卷积。
     *      group等于in_channel，这是depthwiseConv2d，此时group==in_channel==out_channel。
     *      group大于1且小于in_channel，这是分组卷积，out_channel==group。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * 输出张量，卷积的输出。
     */
    OH_NN_OPS_CONV2D = 8,

    /**
     * 二维卷积转置。
     *
     * 如果输入中含有padMode参数：
     *
     * 输入：
     *
     * * x，输入张量。
     * * weight，卷积的权重，要求weight排布为[outChannel，kernelHeight，kernelWidth，inChannel/group]，
     *      inChannel必须要能整除group。
     * * bias，卷积的偏置，是长度为[outChannel]的数组。在量化场景下，bias 参数不需要量化参数，其量化
     *      版本要求输入 OH_NN_INT32 类型数据，实际量化参数由 x 和 weight 共同决定。
     * * stride，卷积核在height和weight上的步幅，是一个长度为2的int数组[strideHeight，strideWidth]。
     *
     * 参数：
     *
     * * dilation，表示扩张卷积在height和weight上的扩张率，是一个长度为2的int数组[dilationHeight，dilationWidth]。
     *      值必须大于或等于1，并且不能超过x的height和width。
     * * padMode，x的填充模式，支持same和valid，int类型，0表示same，1表示valid。
     *      same，输出的高度和宽度与x相同，填充总数将在水平和垂直方向计算，并在可能的情况下均匀分布到顶部和底部、左侧
     *      和右侧。否则，最后一个额外的填充将从底部和右侧完成。
     *      Valid，输出的可能最大高度和宽度将在不填充的情况下返回。额外的像素将被丢弃。
     * * group，将输入x按in_channel分组，int类型。group等于1，这是常规卷积；group大于1且小于或等于in_channel，这是分组卷积。
     * * outputPads，一个整数或元组/2 个整数的列表，指定沿输出张量的高度和宽度的填充量。可以是单个整数，用于为所
     *      有空间维度指定相同的值。沿给定维度的输出填充量必须小于沿同一维度的步幅。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 如果输入中含有padList参数：
     *
     * 输入：
     *
     * * x，输入张量。
     * * weight，卷积的权重，要求weight排布为[outChannel，kernelHeight，kernelWidth，inChannel/group]，
     *      inChannel必须要能整除group。
     * * bias，卷积的偏置，是长度为[outChannel]的数组。在量化场景下，bias 参数不需要量化参数，其量化
     *      版本要求输入 OH_NN_INT32 类型数据，实际量化参数由 x 和 weight 共同决定。
     *
     * 参数：
     *
     * * stride，卷积核在height和weight上的步幅，是一个长度为2的int数组[strideHeight，strideWidth]。
     * * dilation，表示扩张卷积在height和weight上的扩张率，是一个长度为2的int数组[dilationHeight，dilationWidth]。
     *      值必须大于或等于1，并且不能超过x的height和width。
     * * padList，输入x周围的填充，是一个长度为4的int数组[top，bottom，left，right]。
     * * group，将输入x按in_channel分组，int类型。group等于1，这是常规卷积；group大于1且小于或等于in_channel，这是分组卷积。
     * * outputPads，一个整数或元组/2 个整数的列表，指定沿输出张量的高度和宽度的填充量。可以是单个整数，用于为所
     *      有空间维度指定相同的值。沿给定维度的输出填充量必须小于沿同一维度的步幅。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * 输出张量，卷积转置后的输出。
     */
    OH_NN_OPS_CONV2D_TRANSPOSE = 9,

    /**
     * 2维深度可分离卷积
     *
     * 如果输入中含有padMode参数：
     *
     * 输入：
     *
     * * x，输入张量。
     * * weight，卷积的权重，要求weight排布为[outChannel，kernelHeight，kernelWidth，1]，outChannel = channelMultiplier x inChannel。
     * * bias，卷积的偏置，是长度为[outChannel]的数组。在量化场景下，bias 参数不需要量化参数，其量化
     *      版本要求输入 OH_NN_INT32 类型数据，实际量化参数由 x 和 weight 共同决定。
     *
     * 参数：
     *
     * * stride，卷积核在height和weight上的步幅，是一个长度为2的int数组[strideHeight，strideWidth]。
     * * dilation，表示扩张卷积在height和weight上的扩张率，是一个长度为2的int数组[dilationHeight，dilationWidth]。
     *      值必须大于或等于1，并且不能超过x的height和width。
     * * padMode，x的填充模式，支持same和valid，int类型，0表示same，1表示valid
     *      same，输出的高度和宽度与x相同，填充总数将在水平和垂直方向计算，并在可能的情况下均匀分布到顶部和底部、左侧
     *      和右侧。否则，最后一个额外的填充将从底部和右侧完成
     *      Valid，输出的可能最大高度和宽度将在不填充的情况下返回。额外的像素将被丢弃
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 如果输入中含有padList 参数：
     *
     * 输入：
     *
     * * x，输入张量。
     * * weight，卷积的权重，要求weight排布为[outChannel，kernelHeight，kernelWidth，1]，outChannel = channelMultiplier x inChannel。
     * * bias，卷积的偏置，是长度为[outChannel]的数组。在量化场景下，bias 参数不需要量化参数，其量化
     *      版本要求输入 OH_NN_INT32 类型数据，实际量化参数由 x 和 weight 共同决定。
     *
     * 参数：
     *
     * * stride，卷积核在height和weight上的步幅，是一个长度为2的int数组[strideHeight，strideWidth]。
     * * dilation，表示扩张卷积在height和weight上的扩张率，是一个长度为2的int数组[dilationHeight，dilationWidth]。
     *      值必须大于或等于1，并且不能超过x的height和width。
     * * padList，输入x周围的填充，是一个长度为4的int数组[top，bottom，left，right]。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * 输出张量，卷积后的输出。
     */
    OH_NN_OPS_DEPTHWISE_CONV2D_NATIVE = 10,

    /**
     * 对输入的两个标量或张量做商。
     *
     * 输入：
     *
     * * x1，第一个输入是标量或布尔值或数据类型为数字或布尔值的张量。
     * * x2，数据类型根据x1的类型，要求有所不同：
     *      当第一个输入是张量时，第二个输入可以是实数或布尔值或数据类型为实数/布尔值的张量。
     *      当第一个输入是实数或布尔值时，第二个输入必须是数据类型为实数/布尔值的张量。
     *
     * 参数：
     *
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * 输出张量，输出两输入相除后的结果。
     */
    OH_NN_OPS_DIV = 11,

    /**
     * 设置参数对输入进行product(点乘)、sum(相加减)或max(取大值)。
     *
     * 输入：
     *
     * * x1，第一个输入张量。
     * * x2，第二个输入张量。
     *
     * 参数：
     *
     * * mode，枚举，选择操作方式。
     *
     * 输出：
     *
     * * 输出tensor，与x1有相同的数据类型和形状。
     *
     */
    OH_NN_OPS_ELTWISE = 12,

    /**
     * 在给定轴上为tensor添加一个额外的维度。
     *
     * 输入：
     *
     * * x，输入张量。
     * * axis，需要添加的维度的index，int32_t类型，值必须在[-dim-1，dim]，且只允许常量值。
     *
     * 输出：
     *
     * * 输出tensor，与x有相同的数据类型和形状。
     */
    OH_NN_OPS_EXPAND_DIMS = 13,

    /**
     * 根据指定的维度，创建由一个标量填充的张量。
     *
     * 输入：
     *
     * * value，填充的标量。
     * * shape，指定创建张量的维度。
     *
     * 输出：
     *
     * * 输出张量，与value有相同的数据类型，shape由输入指定。
     */
    OH_NN_OPS_FILL = 14,

    /**
     * 全连接，整个输入作为feature map，进行特征提取。
     *
     * 输入：
     *
     * * x，全连接的输入张量。
     * * weight，全连接的权重张量。
     * * bias，全连接的偏置，在量化场景下，bias 参数不需要量化参数，其量化
     *      版本要求输入 OH_NN_INT32 类型数据，实际量化参数由 x 和 weight 共同决定。
     *
     * 参数：
     *
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * output，输出运算后的张量。

     * 如果输入中含有axis参数：
     *
     * 输入：
     *
     * * x，全连接的输入张量。
     * * weight，全连接的权重张量。
     * * bias，全连接的偏置，在量化场景下，bias 参数不需要量化参数，其量化
     *      版本要求输入 OH_NN_INT32 类型数据，实际量化参数由 x 和 weight 共同决定。
     *
     * 参数：
     *
     * * axis，x做全连接的轴，从指定轴axis开始，将axis和axis后面的轴展开成1维去做全连接。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * output，输出运算后的张量。
     */
    OH_NN_OPS_FULL_CONNECTION = 15,

    /**
     * 根据指定的索引和轴返回输入tensor的切片。
     *
     * 输入：
     *
     * * x，输入待切片的tensor。
     * * inputIndices，指定输入x在axis上的索引，是一个int类型的数组，值必须在[0,x.shape[axis])范围内
     * * axis，输入x被切片的轴，int32_t类型的数组，数组长度为1。
     *
     * 输出：
     *
     * * Output，输出切片后的tensor。
     */
    OH_NN_OPS_GATHER = 16,

    /**
     * 计算输入的Hswish激活值。
     *
     * 输入：
     *
     * * 一个n维输入tensor。
     *
     * 输出：
     *
     * * n维Hswish激活值，数据类型和shape和input一致。
     */
    OH_NN_OPS_HSWISH = 17,

    /**
     * 对输入x1和x2，计算每对元素的x<=y的结果。
     *
     * 输入：
     *
     * *  x1，可以是实数、布尔值或数据类型是实数/NN_BOOL的tensor。
     * *  x2，如果input_x是tensor，input_y可以是实数、布尔值，否则只能是tensor，其数据类型是实数或NN_BOOL。
     *
     * 输出：
     *
     * * Tensor，数据类型为NN_BOOL的tensor，使用量化模型时，output的量化参数不可省略，但量化参数的数值不会对输入结果产生影响。
     */
    OH_NN_OPS_LESS_EQUAL = 18,

    /**
     * 计算x1和x2的内积
     *
     * 输入：
     *
     * * x1，n维输入tensor。
     * * x2，n维输入tensor。
     *
     * 参数：
     *
     * * TransposeX，布尔值，是否对x1进行转置。
     * * TransposeY，布尔值，是否对x2进行转置。
     *
     * 输出：
     *
     * * output，计算得到内积，当type!=NN_UNKNOWN时，output数据类型由type决定；当type==NN_UNKNOWN时，
     *      output的数据类型取决于inputX和inputY进行计算时转化的数据类型。
     */
    OH_NN_OPS_MATMUL = 19,

    /**
     * 计算input1和input2对应元素最大值，input1和input2的输入遵守隐式类型转换规则，使数据类型一致。输入必须
     * 是两个张量或一个张量和一个标量。当输入是两个张量时，它们的数据类型不能同时为NN_BOOL。它们的形状支持
     * broadcast成相同的大小。当输入是一个张量和一个标量时，标量只能是一个常数。
     *
     * 输入：
     *
     * *  x1，n维输入tensor，实数或NN_BOOL类型。
     * *  x2，n维输入tensor，实数或NN_BOOL类型。
     *
     * 输出：
     *
     * * output，n维输出tensor，output的shape和数据类型和两个input中精度或者位数高的相同。
     */
    OH_NN_OPS_MAXIMUM = 20,

    /**
     * 在输入tensor上应用 2D 最大值池化。
     *
     * 如果输入中含有padMode参数：
     *
     * 输入：
     *
     * * x，一个张量。
     *
     * 参数：
     *
     * * kernelSize，用来取最大值的kernel大小，是一个长度为2的int数组[kernel_height，kernel_weight]，
     *       第一个数表示kernel高度，第二个数表示kernel宽度。
     * * strides，kernel移动的距离，是一个长度为2的int数组[stride_height，stride_weight]，
     *      第一个数表示高度上的移动步幅，第二个数表示宽度上的移动步幅。
     * * padMode，填充模式，int类型的可选值，0表示same，1表示valid，并且以最近邻的值填充。
     *      same，输出的高度和宽度与x相同，填充总数将在水平和垂直方向计算，并在可能的情况下均匀分布到顶部
     *      和底部、左侧和右侧。否则，最后一个额外的填充将从底部和右侧完成。
     *      valid，输出的可能最大高度和宽度将在不填充的情况下返回。额外的像素将被丢弃。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 如果输入中含有padList参数：
     *
     * 输入：
     *
     * * x，一个张量。
     *
     * 参数：
     *
     * * kernelSize，用来取最大值的kernel大小，是一个长度为2的int数组[kernel_height，kernel_weight]，
     *       第一个数表示kernel高度，第二个数表示kernel宽度。
     * * strides，kernel移动的距离，是一个长度为2的int数组[stride_height，stride_weight]，
     *      第一个数表示高度上的移动步幅，第二个数表示宽度上的移动步幅。
     * * padList，输入x周围的填充，是一个长度为4的int数组[top，bottom，left，right]，并且以最近邻的值填充。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * output，输出x最大值池化后的张量。
     */
    OH_NN_OPS_MAX_POOL = 21,

    /**
     * 将inputX和inputY相同的位置的元素相乘得到output。如果inputX和inputY类型shape不同，要求inputX和inputY可以
     * 通过broadcast扩充成相同的shape进行相乘。
     *
     * 输入：
     *
     * * x1，一个n维tensor。
     * * x2，一个n维tensor。
     *
     * 参数：
     *
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * output，x1和x2每个元素的乘积。
     */
    OH_NN_OPS_MUL = 22,

    /**
     * 根据indices指定的位置，生成一个由one-hot向量构成的tensor。每个onehot向量中的有效值由on_value决定，其他位置由off_value决定。
     *
     * 输入：
     *
     * *  indices，n维tensor。indices中每个元素决定每个one-hot向量，on_value的位置
     * *  depth，一个整型标量，决定one-hot向量的深度。要求depth>0。
     * *  on_value，一个标量，指定one-hot向量中的有效值。
     * *  off_value，(一个标量，指定one-hot向量中除有效位以外，其他位置的值。
     *
     * 参数：
     *
     * *  axis，一个整型标量，指定插入one-hot的维度。
     *       indices的形状是[N，C]，depth的值是D，当axis=0时，output形状为[D，N，C]，
     *       indices的形状是[N，C]，depth的值是D，当axis=-1时，output形状为[N，C，D]，
     *       indices的形状是[N，C]，depth的值是D，当axis=1时，output形状为[N，D，C]。
     *
     * 输出：
     *
     * * output，如果indices时n维tensor，则output是(n+1)维tensor。output的形状由indices和axis共同决定。
     */
    OH_NN_OPS_ONE_HOT = 23,

    /**
     * 在inputX指定维度的数据前后，添加指定数值进行增广。
     *
     * 输入：
     *
     * * inputX，一个n维tensor，要求inputX的排布为[BatchSize，…]。
     * * paddings，一个2维tensor，指定每一维度增补的长度，shape为[n，2]。paddings[i][0]表示第i维上，需要在inputX前增补的数量；
     *      paddings[i][1]表示第i维上，需要在inputX后增补的数量。
     *
     * 参数：
     *
     * * padValues，一个常数，数据类型和inputX一致，指定Pad操作补全的数值。
     *
     * 输出：
     *
     * * output，一个n维tensor，维数和数据类型和inputX保持一致。shape由inputX和paddings共同决定
     *      output.shape[i] = input.shape[i] + paddings[i][0]+paddings[i][1]。
     */
    OH_NN_OPS_PAD = 24,

    /**
     * 求x的y次幂，输入必须是两个tensor或一个tensor和一个标量。当输入是两个tensor时，它们的数据类型不能同时为NN_BOOL，
     * 且要求两个tensor的shape相同。当输入是一个tensor和一个标量时，标量只能是一个常数。
     *
     * 输入：
     *
     * * x，实数、bool值或tensor，tensor的数据类型为实数/NN_BOOL。
     * * y，实数、bool值或tensor，tensor的数据类型为实数/NN_BOOL。
     *
     * 输出：
     *
     * * output，形状由x和y broadcast后的形状决定。
     */
    OH_NN_OPS_POW = 25,

    /**
     * 给定一个tensor，计算其缩放后的值。
     *
     * 输入：
     *
     * * x，一个n维tensor。
     * * scale，缩放tensor。
     * * bias，偏置tensor。
     *
     * 参数：
     *
     * * axis，指定缩放的维度。
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * output，scale的计算结果，一个n维tensor，类型和input一致，shape由axis决定。
     */
    OH_NN_OPS_SCALE = 26,

    /**
     * 输入一个tensor，计算其shape。
     *
     * 输入：
     *
     * * x，一个n维tensor。
     *
     * 输出：
     *
     * * output，输出tensor的维度，一个整型数组。
     */
    OH_NN_OPS_SHAPE = 27,

    /**
     * 给定一个tensor，计算其sigmoid结果。
     *
     * 输入：
     *
     * * input，一个n维tensor。
     *
     * 输出：
     *
     * * output，sigmoid的计算结果，一个n维tensor，类型和shape和input一致。
     */
    OH_NN_OPS_SIGMOID = 28,

    /**
     * 在input tensor各维度，以begin为起点，截取size长度的切片。
     *
     * 输入：
     *
     * * x，n维输入tensor。
     * * begin，一组不小于0的整数，指定每个维度上的起始切分点。
     * * size，一组不小于1的整数，指定每个维度上切片的长度。假设某一维度i，1<=size[i]<=input.shape[i]-begin[i]。
     *
     * 输出：
     *
     * * output，切片得到的n维tensor，其TensorType和input一致，shape和size相同。
     */
    OH_NN_OPS_SLICE = 29,

    /**
     * 给定一个tensor，计算其softmax结果。
     *
     * 输入：
     *
     * * x，n维输入tensor。
     *
     * 参数：
     *
     * * axis，int64类型，指定计算softmax的维度。整数取值范围为[-n，n)。
     *
     * 输出：
     *
     * * output，softmax的计算结果，一个n维tensor，类型和shape和x一致。
     */
    OH_NN_OPS_SOFTMAX = 30,

    /**
     * 将4维tensor在空间维度上进行切分成blockShape[0] * blockShape[1]个小块，然后在batch维度上拼接这些小块。
     *
     * 输入：
     *
     * * x，一个4维tensor
     *
     * 参数：
     *
     * * blockShape，一对整数，每个整数不小于1。
     * * paddings，一对数组，每个数组由两个整数组成。组成paddings的4个整数都不小于0。paddings[0][0]和paddings[0][1]指
     *      定了第三个维度上padding的数量，paddings[1][0]和paddings[1][1]指定了第四个维度上padding的数量。
     *
     * 输出：
     *
     * * output，一个4维tensor，数据类型和input一致。shape由input，blockShape和paddings共同决定，假设input shape为[n，c，h，w]，则有
     *      output.shape[0] = n * blockShape[0] * blockShape[1]
     *      output.shape[1] = c
     *      output.shape[2] = (h + paddings[0][0] + paddings[0][1]) / blockShape[0]
     *      output.shape[3] = (w + paddings[1][0] + paddings[1][1]) / blockShape[1]
     *      要求(h + paddings[0][0] + paddings[0][1])和(w + paddings[1][0] + paddings[1][1])能被
     *      blockShape[0]和blockShape[1]整除。
     */
    OH_NN_OPS_SPACE_TO_BATCH_ND = 31,

    /**
     * Split 算子沿 axis 维度将 input 拆分成多个 tensor，tensor 数量由 outputNum 指定。
     *
     * 输入：
     *
     * * x，n维tensor。
     *
     * 参数：
     *
     * * outputNum，long，输出tensor的数量，output_num类型为int。
     * * size_splits，1维tensor，指定 tensor 沿 axis 轴拆分后，每个 tensor 的大小，size_splits 类型为 int。
     *      如果 size_splits 的数据为空，则 tensor 被拆分成大小均等的 tensor，此时要求 input.shape[axis] 可以被 outputNum 整除；
     *      如果 size_splits 不为空，则要求 size_splits 所有元素之和等于 input.shape[axis]。
     * * axis，指定拆分的维度，axis类型为int。
     *
     * 输出：
     *
     * * outputs，一组n维tensor，每一个tensor类型和shape相同，每个tensor的类型和input一致。
     */
    OH_NN_OPS_SPLIT = 32,

    /**
     * 给定一个tensor，计算其平方根。
     *
     * 输入：
     *
     * * x，一个n维tensor。
     *
     * 输出：
     *
     * * output，输入的平方根，一个n维tensor，类型和shape和input一致。
     */
    OH_NN_OPS_SQRT = 33,

    /**
     * 计算两个输入的差值并返回差值的平方。SquaredDifference算子支持tensor和tensor相减。
     * 如果两个tensor的TensorType不相同，Sub算子会将低精度的tensor转成更高精度的类型。
     * 如果两个tensor的shape不同，要求两个tensor可以通过broadcast拓展成相同shape的tensor。
     *
     * 输入：
     *
     * * x，被减数，inputX是一个tensor，tensor的类型可以是NN_FLOAT16、NN_FLOAT32、NN_INT32或NN_BOOL。
     * * y，减数，inputY是一个tensor，tensor的类型可以是NN_FLOAT16、NN_FLOAT32、NN_INT32或NN_BOOL。
     *
     * 输出：
     *
     * * output，两个input差值的平方。output的shape由inputX和inputY共同决定，inputX和inputY的shape相同时，
     *      output的shape和inputX、inputY相同；shape不同时，需要将inputX或inputY做broadcast操作后，相减得到output。
     *      output的TensorType由两个输入中更高精度的TensorType决定。
     */
    OH_NN_OPS_SQUARED_DIFFERENCE = 34,

    /**
     * 去除axis中，长度为1的维度。支持int8量化输入假设input的shape为[2，1，1，2，2]，axis为[0,1]，
     * 则output的shape为[2，1，2，2]。第0维到第1维之间，长度为0的维度被去除。
     *
     * 输入：
     *
     * * x，n维tensor。
     *
     * 参数：
     *
     * * axis，指定删除的维度。axis可以是一个int64_t的整数或数组，整数的取值范围为[-n，n)。
     *
     * 输出：
     *
     * * output，输出tensor。
     */
    OH_NN_OPS_SQUEEZE = 35,

    /**
     * 将一组tensor沿axis维度进行堆叠，堆叠前每个tensor的维数为n，则堆叠后output维数为n+1。
     *
     * 输入：
     *
     * * x，Stack支持传入多个输入n维tensor，每个tensor要求shape相同且类型相同。
     *
     * 参数：
     *
     * * axis，一个整数，指定tensor堆叠的维度。axis可以是负数，axis取值范围为[-(n+1)，(n+1))。
     *
     * 输出：
     *
     * * output，将input沿axis维度堆叠的输出，n+1维tensor，TensorType和input相同。
     */
    OH_NN_OPS_STACK = 36,

    /**
     * 跨步截取Tensor
     *
     * 输入：
     *
     * * x，n维输入tensor。
     * * begin，1维tensor，begin的长度等于n，begin[i]指定第i维上截取的起点。
     * * end，1维tensor，end的长度等于n，end[i]指定第i维上截取的终点。
     * * strides，1维tensor，strides的长度等于n，strides[i]指定第i维上截取的步长。
     *
     * 参数：
     *
     * * beginMask，一个整数，用于解除begin的限制。将beginMask转成二进制表示，如果binary(beginMask)[i]==1，
     *      则对于第i维，从第一个元素开始，以strides[i]为步长截取元素直到第end[i]-1个元素。
     * * endMask，个整数，用于解除end的限制。将endMask转成二进制表示，如果binary(endMask)[i]==1，则对于第i维，
     *      从第begin[i]个元素起，以strides[i]为步长截取元素直到tensor边界。
     * * ellipsisMask，一个整数，用于解除begin和end的限制。将ellipsisMask转成二进制表示，如果binary(ellipsisMask)[i]==1，
     *      则对于第i维，从第一个元素开始，以strides[i]为补偿，截取元素直到tensor边界。binary(ellipsisMask)仅允许有一位不为0。
     * * newAxisMask，一个整数，用于新增维度。将newAxisMask转成二进制表示，如果binary(newAxisMask)[i]==1，则在第i维插入长度为1的新维度。
     * * shrinkAxisMask，一个整数，用于压缩指定维度。将shrinkAxisMask转成二进制表示，如果binary(shrinkAxisMask)[i]==1，
     *      则舍去第i维所有元素，第i维长度压缩至1。
     *
     * 输出：
     *
     * * 堆叠运算后的Tensor，数据类型与x相同。输出维度rank(x[0])+1 维。
     */
    OH_NN_OPS_STRIDED_SLICE = 37,

    /**
     * 计算两个输入的差值。
     *
     * 输入：
     *
     * * x，被减数，x是一个tensor。
     * * y，减数，y是一个tensor。
     *
     * 参数：
     *
     * * activationType，是一个整型常量，且必须是FuseType中含有的值。
     *      在输出之前调用指定的激活。
     *
     * 输出：
     *
     * * output，两个input相减的差。output的shape由inputX和inputY共同决定，inputX和inputY的shape相同时，output的shape和inputX、inputY相同；
     *      shape不同时，需要将inputX或inputY做broadcast操作后，相减得到output。output的TensorType由两个输入中更高精度的TensorType决定。
     */
    OH_NN_OPS_SUB = 38,

    /**
     * 计算输入tensor的双曲正切值。
     *
     * 输入：
     *
     * * x，n维tensor。
     *
     * 输出：
     *
     * * output，input的双曲正切，TensorType和tensor shape和input相同。
     */
    OH_NN_OPS_TANH = 39,

    /**
     * 以multiples指定的次数拷贝input。
     *
     * 输入：
     * * x，n维tensor。
     * * multiples，1维tensor，指定各个维度拷贝的次数。其长度m不小于input的维数n。
     *
     * 输出：
     * * Tensor，m维tensor，TensorType与input相同。如果input和multiples长度相同，
     *      则output和input维数一致，都是n维tensor；如果multiples长度大于n，则用1填充input的维度，
     *      再在各个维度上拷贝相应的次数，得到m维tensor。
     */
    OH_NN_OPS_TILE = 40,

    /**
     * 根据permutation对input 0进行数据重排。
     *
     * 输入：
     *
     * * x，n维tensor，待重排的tensor。
     * * perm，1维tensor，其长度和input 0的维数一致。
     *
     * 输出：
     *
     * * output，n维tensor，output 0的TensorType与input 0相同，shape由input 0的shape和permutation共同决定。
     */
    OH_NN_OPS_TRANSPOSE = 41,

    /**
     * keepDims为false时，计算指定维度上的平均值，减少input的维数；当keepDims为true时，计算指定维度上的平均值，保留相应的维度。
     *
     * 输入：
     *
     * *  input，n维输入tensor，n<8。
     * *  axis，1维tensor，指定计算均值的维度，axis中每个元素的取值范围为[-n，n)。
     *
     * 参数：
     *
     * *  keepDims，布尔值，是否保留维度的标志位。
     *
     * 输出：
     *
     * *  output，m维输出tensor，数据类型和input相同。当keepDims为false时，m==n；当keepDims为true时，m<n。
     */
    OH_NN_OPS_REDUCE_MEAN = 42,

    /**
     * 采用Bilinear方法，按给定的参数对input进行变形。
     *
     * 输入：
     *
     * * x，4维输入tensor，input中的每个元素不能小于0。input排布必须是[batchSize，height，width，channels]。
     *
     * 参数：
     *
     * * newHeight，resize之后4维tensor的height值。
     * * newWidth，resize之后4维tensor的width值。
     * * preserveAspectRatio，一个布尔值，指示resize操作是否保持input tensor的height/width比例。
     * * coordinateTransformMode，一个int32整数，指示Resize操作所使用的坐标变换方法，目前支持以下方法：
     * * excludeOutside，一个int64浮点数。当excludeOutside=1时，超出input边界的采样权重被置为0,其余权重重新归一化处理。
     *
     * 输出：
     *
     * * output，n维输出tensor，output的shape和数据类型和input相同。
     */
    OH_NN_OPS_RESIZE_BILINEAR = 43,

     /**
     * 求input平方根的倒数。
     *
     * 输入：
     *
     * *  x，n维输入tensor，input中的每个元素不能小于0，n<8。
     *
     * 输出：
     *
     * *  output，n维输出tensor，output的shape和数据类型和input相同。

     */
    OH_NN_OPS_RSQRT = 44,

     /**
     * 根据inputShape调整input的形状。
     *
     * 输入：
     *
     * *  x，一个n维输入tensor。
     * *  InputShape，一个1维tensor，表示输出tensor的shape，需要是一个常量tensor。
     *
     * 输出：
     *
     * * output，输出tensor，数据类型和input一致，shape由inputShape决定。
     */
    OH_NN_OPS_RESHAPE = 45,

    /**
     * 计算input和weight的PReLU激活值。
     *
     * 输入：
     *
     * *  x，一个n维tensor，如果n>=2，则要求inputX的排布为[BatchSize，…，Channels]，第二个维度为通道数。
     * *  weight，一个1维tensor。weight的长度只能是1或者等于通道数。当weight长度为1，则inputX所有通道共享一个权重值。
     *       若weight长度等于通道数，每个通道独享一个权重，若inputX维数n<2，weight长度只能为1。
     * 输出：
     *
     *    output，x的PReLU激活值。形状和数据类型和inputX保持一致。
     */
    OH_NN_OPS_PRELU = 46,

    /**
     * 计算input的Relu激活值。
     *
     * 输入：
     *
     * * input，一个n维输入tensor。
     *
     * 输出：
     *
     * * output，n维Relu输出tensor，数据类型和shape和input一致。
     */
    OH_NN_OPS_RELU = 47,

    /**
     * 计算input的Relu6激活值，即对input中每个元素x，计算min(max(x，0)，6)。
     *
     * 输入：
     *
     * * input，一个n维输入tensor。
     *
     * 输出：
     *
     * * output，n维Relu6输出tensor，数据类型和shape和input一致。
     */
    OH_NN_OPS_RELU6 = 48,

    /**
     * 对一个tensor从某一axis开始做层归一化。
     *
     * 输入：
     *
     * *  input，一个n维输入tensor。
     * *  gamma，一个m维tensor，gamma维度应该与input做归一化部分的shape一致。
     * *  beta，一个m维tensor，shape与gamma一样。
     *
     * 参数：
     *
     * *  beginAxis，是一个NN_INT32的标量，指定开始做归一化的轴，取值范围是[1，rank(input))。
     * *  epsilon，是一个NN_FLOAT32的标量，是归一化公式中的微小量，常用值是1e-7。
     *
     * 输出：
     *
     * * output，n维输出tensor，数据类型和shape和input一致。
     */
    OH_NN_OPS_LAYER_NORM = 49,

    /**
     * 当keepDims为false时，过乘以维度中的所有元素来减小张量的维度，减少input的维数；当keepDims为true时，过乘以维度中的所有元素来减小张量的维度，保留相应的维度。
     *
     * 输入：
     *
     * *  input，n维输入tensor，n<8。
     * *  axis，1维tensor，指定计算乘的维度，axis中每个元素的取值范围为[-n，n)。
     *
     * 参数：
     *
     * *  keepDims，布尔值，是否保留维度的标志位。
     *
     * 输出：
     *
     * *  output，m维输出tensor，数据类型和input相同。当keepDims为false时，m==n；当keepDims为true时，m<n。
     */
    OH_NN_OPS_REDUCE_PROD = 50,

    /**
     * 当keepDims为false时，计算指定维度上的逻辑与，减少input的维数；当keepDims为true时，计算指定维度上的逻辑与，保留相应的维度。
     *
     * 输入：
     *
     * *  n维输入tensor，n<8。
     * *  1维tensor，指定计算逻辑与的维度，axis中每个元素的取值范围为[-n，n)。
     *
     * 参数：
     *
     * *  keepDims，布尔值，是否保留维度的标志位。
     *
     * 输出：
     * *  output，m维输出tensor，数据类型和input相同。当keepDims为false时，m==n；当keepDims为true时，m<n。
     */
    OH_NN_OPS_REDUCE_ALL = 51,

    /**
     * 数据类型转换。
     *
     * 输入：
     *
     * *  input，n维tensor。
     *
     * 参数：
     *
     * *  src_t，定义输入的数据类型。
     * *  dst_t，定义输出的数据类型。
     *
     * 输出：
     *
     * * output，n维tensor，数据类型由input2决定  输出shape和输入相同。
     */
    OH_NN_OPS_QUANT_DTYPE_CAST = 52,

    /**
     * 查找沿最后一个维度的k个最大条目的值和索引。
     *
     * 输入：
     *
     * *  x，n维tensor。
     * *  input k，指明是得到前k个数据以及其index。
     *
     * 参数：
     *
     * *  sorted，如果为True，按照大到小排序，如果为False，按照小到大排序。
     *
     * 输出：
     *
     * * output0,最后一维的每个切片中的k个最大元素。
     * * output1，输入的最后一个维度内的值的索引。
     */
    OH_NN_OPS_TOP_K = 53,

    /**
     * 返回跨轴的张量最大值的索引。
     *
     * 输入：
     *
     * *  input，n维tensor，输入张量(N，∗)，其中∗意味着任意数量的附加维度。
     *
     * 参数：
     *
     * *  axis，指定求最大值索引的维度。
     * *  keep_dims，bool值，是否维持输入张量维度。
     *
     * 输出：
     * *  output，tensor，轴上输入张量最大值的索引。
     */
    OH_NN_OPS_ARG_MAX = 54,

    /**
     * 根据输入axis的值。增加一个维度。
     *
     * 输入：
     * * x，n维tensor。
     *
     * 参数：
     *
     * * axis，指定增加的维度。axis可以是一个整数或一组整数，整数的取值范围为[-n，n)。
     *
     * 输出：
     * * output，输出tensor。
     */
    OH_NN_OPS_UNSQUEEZE = 55,

    /**
     * 高斯误差线性单元激活函数。output=0.5∗x∗(1+tanh(x/2))，不支持int量化输入。
     *
     * 输入：
     * * 一个n维输入tensor。
     *
     * 输出：
     * * output，n维Relu输出tensor，数据类型和shape和input一致。
     */
    OH_NN_OPS_GELU = 56,
} OH_NN_OperationType;

/**
 * @brief 操作数的类型
 *
 * 操作数通常用于设置模型的输入、输出和算子参数。作为模型（或算子）的输入和输出时，需要将操作数类型设置为{@link OH_NN_TENSOR}；操作数
 * 用于设置算子参数时，需要指定参数类型。假设正在设置{@link OH_NN_OPS_CONV2D}算子的pad参数，则需要将
 * {@link OH_NN_Tensor}实例的type属性设置为{@link OH_NN_CONV2D_PAD}。其他算子参数的设置以此类推，枚举值
 * 的命名遵守 OH_NN_{算子名称}_{属性名} 的格式。
 *
 * @since 9
 * @version 1.0
 */
typedef enum {
    /** Tensor类型 */
    OH_NN_TENSOR = 0,

    /** Add算子的activationType参数 */
    OH_NN_ADD_ACTIVATIONTYPE = 1,

    /** AvgPool算子的kernel_size参数 */
    OH_NN_AVG_POOL_KERNEL_SIZE = 2,
    /** AvgPool算子的stride参数 */
    OH_NN_AVG_POOL_STRIDE = 3,
    /** AvgPool算子的pad_mode参数 */
    OH_NN_AVG_POOL_PAD_MODE = 4,
    /** AvgPool算子的pad参数 */
    OH_NN_AVG_POOL_PAD = 5,
    /** AvgPool算子的activation_type参数 */
    OH_NN_AVG_POOL_ACTIVATION_TYPE = 6,

    /** BatchNorm算子的eosilon参数 */
    OH_NN_BATCH_NORM_EPSILON = 7,

    /** BatchToSpaceND算子的blockSize参数 */
    OH_NN_BATCH_TO_SPACE_ND_BLOCKSIZE = 8,
    /** BatchToSpaceND算子的crops参数 */
    OH_NN_BATCH_TO_SPACE_ND_CROPS = 9,

    /** Concat算子的axis参数 */
    OH_NN_CONCAT_AXIS = 10,

    /** Conv2D算子的strides参数 */
    OH_NN_CONV2D_STRIDES = 11,
    /** Conv2D算子的pad参数 */
    OH_NN_CONV2D_PAD = 12,
    /** Conv2D算子的dilation参数 */
    OH_NN_CONV2D_DILATION = 13,
    /** Conv2D算子的padMode参数 */
    OH_NN_CONV2D_PAD_MODE = 14,
    /** Conv2D算子的activationType参数 */
    OH_NN_CONV2D_ACTIVATION_TYPE = 15,
    /** Conv2D算子的group参数 */
    OH_NN_CONV2D_GROUP = 16,

    /** Conv2DTranspose算子的strides参数 */
    OH_NN_CONV2D_TRANSPOSE_STRIDES = 17,
    /** Conv2DTranspose算子的pad参数 */
    OH_NN_CONV2D_TRANSPOSE_PAD = 18,
    /** Conv2DTranspose算子的dilation参数 */
    OH_NN_CONV2D_TRANSPOSE_DILATION = 19,
    /** Conv2DTranspose算子的outputPaddings参数 */
    OH_NN_CONV2D_TRANSPOSE_OUTPUT_PADDINGS = 20,
    /** Conv2DTranspose算子的padMode参数 */
    OH_NN_CONV2D_TRANSPOSE_PAD_MODE = 21,
    /** Conv2DTranspose算子的activationType参数 */
    OH_NN_CONV2D_TRANSPOSE_ACTIVATION_TYPE = 22,
    /** Conv2DTranspose算子的group参数 */
    OH_NN_CONV2D_TRANSPOSE_GROUP = 23,

    /** DepthwiseConv2dNative算子的strides参数 */
    OH_NN_DEPTHWISE_CONV2D_NATIVE_STRIDES = 24,
    /** DepthwiseConv2dNative算子的pad参数 */
    OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD = 25,
    /** DepthwiseConv2dNative算子的dilation参数 */
    OH_NN_DEPTHWISE_CONV2D_NATIVE_DILATION = 26,
    /** DepthwiseConv2dNative算子的padMode参数 */
    OH_NN_DEPTHWISE_CONV2D_NATIVE_PAD_MODE = 27,
    /** DepthwiseConv2dNative算子的activationType参数 */
    OH_NN_DEPTHWISE_CONV2D_NATIVE_ACTIVATION_TYPE = 28,

    /** Div算子的activationType参数 */
    OH_NN_DIV_ACTIVATIONTYPE = 29,

    /** Eltwise算子的mode参数 */
    OH_NN_ELTWISE_MODE = 30,

    /** FullConnection算子的axis参数 */
    OH_NN_FULL_CONNECTION_AXIS = 31,
    /** FullConnection算子的activationType参数 */
    OH_NN_FULL_CONNECTION_ACTIVATIONTYPE = 32,

    /** Matmul算子的transposeA参数 */
    OH_NN_MATMUL_TRANSPOSE_A = 33,
    /** Matmul算子的transposeB参数 */
    OH_NN_MATMUL_TRANSPOSE_B = 34,
    /** Matmul算子的activationType参数 */
    OH_NN_MATMUL_ACTIVATION_TYPE = 35,

    /** MaxPool算子的kernel_size参数 */
    OH_NN_MAX_POOL_KERNEL_SIZE = 36,
    /** MaxPool算子的stride参数 */
    OH_NN_MAX_POOL_STRIDE = 37,
    /** MaxPool算子的pad_mode参数 */
    OH_NN_MAX_POOL_PAD_MODE = 38,
    /** MaxPool算子的pad参数 */
    OH_NN_MAX_POOL_PAD = 39,
    /** MaxPool算子的activation_type参数 */
    OH_NN_MAX_POOL_ACTIVATION_TYPE = 40,

    /** Mul算子的activationType参数 */
    OH_NN_MUL_ACTIVATION_TYPE = 41,

    /** OneHot算子的axis参数 */
    OH_NN_ONE_HOT_AXIS = 42,

    /** Pad算子的constant_value参数 */
    OH_NN_PAD_CONSTANT_VALUE = 43,

    /** Scale算子的activationType参数*/
    OH_NN_SCALE_ACTIVATIONTYPE = 44,
    /** Scale算子的axis参数*/
    OH_NN_SCALE_AXIS = 45,

    /** Softmax算子的axis参数 */
    OH_NN_SOFTMAX_AXIS = 46,

    /** SpaceToBatchND算子的BlockShape参数 */
    OH_NN_SPACE_TO_BATCH_ND_BLOCK_SHAPE = 47,
    /** SpaceToBatchND算子的Paddings参数 */
    OH_NN_SPACE_TO_BATCH_ND_PADDINGS = 48,

    /** Split算子的Axis参数 */
    OH_NN_SPLIT_AXIS = 49,
    /** Split算子的OutputNum参数 */
    OH_NN_SPLIT_OUTPUT_NUM = 50,
    /** Split算子的SizeSplits参数 */
    OH_NN_SPLIT_SIZE_SPLITS = 51,

    /** Squeeze算子的Axis参数 */
    OH_NN_SQUEEZE_AXIS = 52,

    /** Stack算子的Axis参数 */
    OH_NN_STACK_AXIS = 53,

    /** StridedSlice算子的BeginMask参数 */
    OH_NN_STRIDED_SLICE_BEGIN_MASK = 54,
    /** StridedSlice算子的EndMask参数 */
    OH_NN_STRIDED_SLICE_END_MASK = 55,
    /** StridedSlice算子的EllipsisMask参数 */
    OH_NN_STRIDED_SLICE_ELLIPSIS_MASK = 56,
    /** StridedSlice算子的NewAxisMask参数 */
    OH_NN_STRIDED_SLICE_NEW_AXIS_MASK = 57,
    /** StridedSlice算子的ShrinkAxisMask参数 */
    OH_NN_STRIDED_SLICE_SHRINK_AXIS_MASK = 58,

    /** Sub算子的ActivationType参数 */
    OH_NN_SUB_ACTIVATIONTYPE = 59,

    /** ReduceMean算子的keep_dims参数*/
    OH_NN_REDUCE_MEAN_KEEP_DIMS = 60,

    /** ResizeBilinear算子的new_height参数*/
    OH_NN_RESIZE_BILINEAR_NEW_HEIGHT = 61,
    /** ResizeBilinear算子的new_width参数*/
    OH_NN_RESIZE_BILINEAR_NEW_WIDTH = 62,
    /** ResizeBilinear算子的preserve_aspect_ratio参数*/
    OH_NN_RESIZE_BILINEAR_PRESERVE_ASPECT_RATIO = 63,
    /** ResizeBilinear算子的coordinate_transform_mode参数*/
    OH_NN_RESIZE_BILINEAR_COORDINATE_TRANSFORM_MODE = 64,
    /** ResizeBilinear算子的exclude_outside参数*/
    OH_NN_RESIZE_BILINEAR_EXCLUDE_OUTSIDE = 65,

    /** LayerNorm算子的beginNormAxis参数 */
    OH_NN_LAYER_NORM_BEGIN_NORM_AXIS = 66,
    /** LayerNorm算子的epsilon参数 */
    OH_NN_LAYER_NORM_EPSILON = 67,
    /** LayerNorm算子的beginParamsAxis参数 */
    OH_NN_LAYER_NORM_BEGIN_PARAM_AXIS = 68,
    /** LayerNorm算子的elementwiseAffine参数 */
    OH_NN_LAYER_NORM_ELEMENTWISE_AFFINE = 69,

    /** ReduceProd算子的keep_dims参数*/
    OH_NN_REDUCE_PROD_KEEP_DIMS = 70,

    /** ReduceAll算子的keep_dims参数*/
    OH_NN_REDUCE_ALL_KEEP_DIMS = 71,

    /** QuantDTypeCast算子的src_t参数*/
    OH_NN_QUANT_DTYPE_CAST_SRC_T = 72,
    /** QuantDTypeCast算子的dst_t参数*/
    OH_NN_QUANT_DTYPE_CAST_DST_T = 73,

    /** Topk算子的Sorted参数 */
    OH_NN_TOP_K_SORTED = 74,

    /** ArgMax算子的axis参数 */
    OH_NN_ARG_MAX_AXIS = 75,
    /** ArgMax算子的keepDims参数 */
    OH_NN_ARG_MAX_KEEPDIMS = 76,

    /** Unsqueeze算子的Axis参数 */
    OH_NN_UNSQUEEZE_AXIS = 77,
} OH_NN_TensorType;

/**
 * @brief 自定义的32位无符号整型数组类型
 *
 * 该结构体用于存储32位无符号整型数组，size要求记录数组的长度。
 *
 * @since 9
 * @version 1.0
 */
typedef struct OH_NN_UInt32Array {
    /** 无符号整型数组的指针 */
    uint32_t *data;
    /** 数组长度 */
    uint32_t size;
} OH_NN_UInt32Array;

/**
 * @brief 量化信息
 *
 * 在量化的场景中，32位浮点型数据需要根据量化参数，按公式 `浮点数=scale*(量化值-zeroPoint)` 量化成比特位更少的数据类型，
 * 其中r是浮点数，q是量化后的结果。
 *
 * @since 9
 * @version 1.0
 */
typedef struct OH_NN_QuantParam {
    /** 指定numBits、scale和zeroPoint数组的长度。在per-layer量化的场景下，quantCount通常指定为1，即一个tensor所有通道
     *  共享一套量化参数；在per-channel量化场景下，quantCount通常和tensor通道数一致，每个通道使用自己的量化参数。
     */
    uint32_t quantCount;
    /** 量化位数 */
    const uint32_t *numBits;
    /** 指向scale量化信息的指针 */
    const double *scale;
    /** 指向zero point量化信息的指针 */
    const int32_t *zeroPoint;
} OH_NN_QuantParam;

/**
 * @brief 操作数结构体
 *
 * {@link OH_NN_Tensor}类型通常用于构造模型图中的数据节点和算子参数，在构造操作数时需要明确数据类型、维数、维度信息和量化信息。
 * type成员指定操作数的用途，当操作数用作模型图中的输入、输出，则要求type置为{@link OH_NN_TENSOR}；当操作数用作算子参数，
 * 则需要指定为具体的枚举值，具体参考{@link OH_NN_TensorType}。
 *
 * @since 9
 * @version 1.0
 */
typedef struct OH_NN_Tensor {
    /** 指定操作数的数据类型，要求从{@link OH_NN_DataType}枚举类型中取值。 */
    OH_NN_DataType dataType;
    /** 指定操作数的维数 */
    uint32_t dimensionCount;
    /** 指定操作数的维度信息 */
    const int32_t *dimensions;
    /** 指定操作数的量化信息，数据类型要求为{@link OH_NN_QuantParam}。 */
    const OH_NN_QuantParam *quantParam;
    /** 指定操作数的类型, 要求从{@link OH_NN_TensorType}枚举类型中取值。 */
    OH_NN_TensorType type;
} OH_NN_Tensor;

/**
 * @brief 内存结构体
 *
 * @since 9
 * @version 1.0
 */
typedef struct OH_NN_Memory {
    /** 指向共享内存的指针，该共享内存通常由底层硬件驱动申请 */
    void * const data;
    /** 记录共享内存的字节长度 */
    const size_t length;
} OH_NN_Memory;

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // NEURAL_NETWORK_RUNTIME_TYPE_H