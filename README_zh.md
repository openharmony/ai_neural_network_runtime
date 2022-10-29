# Neural Network Runtime

## 简介

Neural Network Runtime（神经网络运行时）是一套面向AI领域的运行时部件，适配上层AI推理引擎和底层加速芯片，为端侧AI推理引擎提供硬件加速的计算能力。

如架构图所示，在OpenHarmony系统上，AI应用通常要经过AI推理引擎和Neural Network Runtime才能对接底层芯片驱动，进而加速推理计算。Neural Network Runtime和芯片驱动直接通过HDI接口交互，Neural Network Runtime将模型和数据传递给芯片驱动，通过HDI接口在加速芯片上执行推理计算，计算结果通过Neural Network Runtime、AI推理引擎逐层返回至AI应用。

通常，AI应用、AI推理引擎、Neural Network Runtime处在同一个进程下，芯片驱动运行在另一个进程下，两者之间需要借助进程间通信（IPC）传递模型和计算数据。Neural Network Runtime根据HDI接口实现了HDI客户端，相应的，芯片厂商需要根据HDI接口实现并开放HDI服务。

**图1** Neural Network Runtime架构图
!["Neural Network Runtime架构图"](neural_network_runtime_intro.png)

## 目录

```undefined
/foundation/ai/neural_network_runtime
├── common
├── example                        # 开发样例目录
│   ├── deep_learning_framework    # 应用开发样例存放目录
│   └── drivers                    # 设备驱动开发样例存放目录
├── frameworks                     # 框架代码存放目录
│   └── native
│       └── op                     # 算子头文件和实现存放目录
├── interfaces                     # 对外接口存放目录
│   ├── innerkits                  # 对内部子系统暴露的头文件存放目录
│   └── kits                       # 对外开放的头文件存放目录 
└── test                           # 测试用例存放目录
    ├── system_test                # 系统测试用例存放目录
    └── unittest                   # 单元测试用例存放目录
```

## 编译构建

在OpenHarmony源码根目录下，调用以下指令，单独编译Neural Network Runtime。
```shell
./build.sh --product-name name --ccache --build-target neural_network_runtime
```
> **说明：** name为产品名称，例如Hi3516DV300、rk3568等。

## 说明

### 接口说明

完整的接口文档请参考：
- [neural_network_runtime.h](./interfaces/kits/c/neural_network_runtime.h)
- [neural_network_runtime_type.h](./interfaces/kits/c/neural_network_runtime_type.h)

### 使用说明

- AI推理引擎/应用开发请参考：[Neural Network Runtime开发指导](./neural-network-runtime-guidelines.md)
- AI加速芯片驱动/设备开发请参考：[Neural Network Runtime设备开发指导](./example/drivers/README_zh.md)

## 相关仓

- [**neural_network_runtime**](https://gitee.com/openharmony-sig/neural_network_runtime)
- [Mindspore](https://gitee.com/openharmony/third_party_mindspore)
