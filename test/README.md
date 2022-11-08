# 测试用例运行指导

本指导以RK3568为例，介绍Neural Network Runtime测试用例的执行流程：

1. 编译测试用例。

    调用以下命令编译 Neural Network Runtime 单元测试用例和系统测试用例。

    ```shell
    ./build.sh --product-name rk3568 --ccache --build-target nnrt_test_target --jobs 4
    ```

    编译完成后，在`out/rk3568/tests`目录下找到单元测试用例和系统测试用例，如下图所示：

    ```text
    /out/rk3568/tests
    ├── systemtest                         # 系统测试用例存放目录
    │   └── neural_network_runtime         # Neural Network Runtime系统测试用例存放目录
    └── unittest                           # 单元测试用例存放目录
        └── neural_network_runtime         # Neural Network Runtime测试单元用例存放目录
    ```

2. 上传测试用例。

    执行以下代码，将测试用例推送到设备。

    ```shell
    hdc_std shell "mkdir /data/local/tmp/nnrt_test"
    hdc_std file send ./out/rk3568/tests/unittest/neural_network_runtime/. /data/local/tmp/nnrt_test
    hdc_std file send ./out/rk3568/tests/systemtest/neural_network_runtime/. /data/local/tmp/nnrt_test
    ```

3. 执行单元测试用例。

    以`NeuralNetworkRuntimeTest`为例，执行单元测试。

    ```shell
    hdc_std shell "chmod 755 /data/local/tmp/nnrt_test/NeuralNetworkRuntimeTest"
    hdc_std shell "/data/local/tmp/nnrt_test/NeuralNetworkRuntimeTest"
    ```

    如果用例全部通过，应该得到以下输出：

    ```text
    [==========] 106 tests from 1 test suite ran. (101ms total)
    [  PASSED  ] 106 tests.
    ```

4. 执行系统测试用例（可选）。

    以`End2EndTest`为例，执行以下指令，运行系统测试。

    ```shell
    hdc_std shell "chmod 755 /data/local/tmp/nnrt_test/End2EndTest"
    hdc_std shell "/data/local/tmp/nnrt_test/End2EndTest"
    ```

    如果用例全部通过，应该得到以下输出：

    ```text
    [==========] 8 tests from 1 test suite ran. (648ms total)
    [  PASSED  ] 8 tests.
    ```

    > **说明：**
    >
    > 系统测试需要在提供Neural Network Runtime加速芯片驱动的设备上执行，加速芯片驱动的开发请参考[Neural Network Runtime设备开发指导](./example/drivers/README_zh.md)。
