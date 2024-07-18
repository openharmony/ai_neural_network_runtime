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

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "common/log.h"
#include "hdi_prepared_model_v2_1.h"
#include "memory_manager.h"
#include "transform.h"
#include "test/unittest/common/v2_1/mock_idevice.h"
#include "test/unittest/common/file_utils.h"
#include "tensor.h"
#include "nntensor.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;
namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class HDIPreparedModelTest : public testing::Test {
protected:
    void GetBuffer(void*& buffer, size_t length);
    void InitTensor(std::vector<IOTensor>& inputs, void* buffer, size_t length);
    OH_NN_ReturnCode Run(std::vector<IOTensor>& inputs);
    OH_NN_ReturnCode RunFail(std::vector<IOTensor>& inputs);
};

class MockTensor : public Tensor {
public:
    MOCK_METHOD1(SetTensorDesc, OH_NN_ReturnCode(const TensorDesc*));
    MOCK_METHOD0(CreateData, OH_NN_ReturnCode());
    MOCK_METHOD1(CreateData, OH_NN_ReturnCode(size_t));
    MOCK_METHOD3(CreateData, OH_NN_ReturnCode(int, size_t, size_t));
    MOCK_CONST_METHOD0(GetTensorDesc, TensorDesc*());
    MOCK_CONST_METHOD0(GetData, void*());
    MOCK_CONST_METHOD0(GetFd, int());
    MOCK_CONST_METHOD0(GetSize, size_t());
    MOCK_CONST_METHOD0(GetOffset, size_t());
    MOCK_CONST_METHOD0(GetBackendID, size_t());
};

void HDIPreparedModelTest::GetBuffer(void*& buffer, size_t length)
{
    std::string data = "ABCD";
    const size_t dataLength = 100;
    data.resize(dataLength, '-');

    std::string filename = "/data/log/memory-001.dat";
    FileUtils fileUtils(filename);
    fileUtils.WriteFile(data);

    int fd = open(filename.c_str(), O_RDWR);
    EXPECT_NE(-1, fd);

    const auto& memoryManager = MemoryManager::GetInstance();
    buffer = memoryManager->MapMemory(fd, length);
    close(fd);
}

void HDIPreparedModelTest::InitTensor(std::vector<IOTensor>& inputs, void* buffer, size_t length)
{
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_INT8;
    inputTensor.dataType = OH_NN_INT8;
    inputTensor.format = OH_NN_FORMAT_NCHW;
    inputTensor.data = buffer;
    inputTensor.length = length;
    inputs.emplace_back(std::move(inputTensor));
}

OH_NN_ReturnCode HDIPreparedModelTest::Run(std::vector<IOTensor>& inputs)
{
    const int vvPosition = 2;
    std::vector<IOTensor> outputs;
    std::vector<std::vector<int32_t>> outputsDims {{0}};
    std::vector<bool> isOutputBufferEnough {};

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);
    EXPECT_CALL(*sp, Run(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(
            ::testing::SetArgReferee<vvPosition>(outputsDims),
            ::testing::Return(HDF_SUCCESS))
        );

    OH_NN_ReturnCode result = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    return result;
}

/**
 * @tc.name: hidpreparedmodel_constructor_001
 * @tc.desc: Verify the Constructor function validate constructor success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_constructor_001, TestSize.Level0)
{
    OHOS::sptr<V2_1::IPreparedModel> hdiPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(hdiPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(hdiPreparedModel);
    EXPECT_NE(preparedModel, nullptr);
}

/**
 * @tc.name: hidpreparedmodel_exportmodelcache_001
 * @tc.desc: Verify the ExportModelCache function return memory error.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_exportmodelcache_001, TestSize.Level0)
{
    std::vector<V2_1::SharedBuffer> bufferVect = {{100, 100, 0, 100}};
    OHOS::sptr<V2_1::IPreparedModel> hdiPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(hdiPreparedModel);
    std::vector<Buffer> modelCache;
    EXPECT_CALL(*((V2_1::MockIPreparedModel*)hdiPreparedModel.GetRefPtr()),
        ExportModelCache(::testing::_))
        .WillRepeatedly(
            ::testing::DoAll(
                ::testing::SetArgReferee<0>(bufferVect),
                ::testing::Return(HDF_SUCCESS)
            )
        );

    OH_NN_ReturnCode result = preparedModel->ExportModelCache(modelCache);
    EXPECT_EQ(OH_NN_MEMORY_ERROR, result);
}

/**
 * @tc.name: hidpreparedmodel_exportmodelcache_002
 * @tc.desc: Verify the ExportModelCache function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_exportmodelcache_002, TestSize.Level0)
{
    std::vector<V2_1::SharedBuffer> bufferVect;
    OHOS::sptr<V2_1::IPreparedModel> mockPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(mockPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(mockPreparedModel);
    std::vector<Buffer> modelCache;
    EXPECT_CALL(*((V2_1::MockIPreparedModel*)mockPreparedModel.GetRefPtr()),
        ExportModelCache(::testing::_))
        .WillRepeatedly(
            ::testing::DoAll(
                ::testing::SetArgReferee<0>(bufferVect),
                ::testing::Return(HDF_SUCCESS)
            )
        );

    OH_NN_ReturnCode result = preparedModel->ExportModelCache(modelCache);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/**
 * @tc.name: hidpreparedmodel_exportmodelcache_003
 * @tc.desc: Verify the ExportModelCache function return invalid parameter.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_exportmodelcache_003, TestSize.Level0)
{
    OHOS::sptr<V2_1::IPreparedModel> hdiPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(hdiPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(hdiPreparedModel);
    std::vector<Buffer> modelCache;
    OH_NN_ReturnCode result = preparedModel->ExportModelCache(modelCache);
    EXPECT_EQ(OH_NN_SUCCESS, result);
}

/**
 * @tc.name: hidpreparedmodel_exportmodelcache_004
 * @tc.desc: Verify the ExportModelCache function return unvailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_exportmodelcache_004, TestSize.Level0)
{
    std::vector<V2_1::SharedBuffer> bufferVect = {{100, 100, 0, 100}};
    OHOS::sptr<V2_1::IPreparedModel> mockPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(mockPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(mockPreparedModel);
    std::vector<Buffer> modelCache;
    EXPECT_CALL(*((V2_1::MockIPreparedModel*)mockPreparedModel.GetRefPtr()),
        ExportModelCache(::testing::_))
        .WillRepeatedly(
            ::testing::DoAll(
                ::testing::SetArgReferee<0>(bufferVect),
                ::testing::Return(HDF_FAILURE)
            )
        );

    OH_NN_ReturnCode result = preparedModel->ExportModelCache(modelCache);
    EXPECT_EQ(OH_NN_SAVE_CACHE_EXCEPTION, result);
}

/**
 * @tc.name: hidpreparedmodel_exportmodelcache_005
 * @tc.desc: Verify the ExportModelCache function return unvailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_exportmodelcache_005, TestSize.Level0)
{
    LOGE("ExportModelCache hidpreparedmodel_exportmodelcache_005");
    std::vector<V2_1::SharedBuffer> bufferVect = {{100, 100, 0, 100}};
    OHOS::sptr<V2_1::IPreparedModel> mockPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(mockPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(mockPreparedModel);

    std::vector<Buffer> modelCache;
    Buffer buffer;
    modelCache.emplace_back(buffer);
    OH_NN_ReturnCode result = preparedModel->ExportModelCache(modelCache);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_001
 * @tc.desc: Verify the Run function return invalid parameter.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_001, TestSize.Level0)
{
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_INT8;

    IOTensor outputTensor;
    outputTensor.dataType = OH_NN_INT8;
    std::vector<IOTensor> inputs;
    inputs.emplace_back(std::move(inputTensor));
    std::vector<IOTensor> outputs;

    std::vector<V2_1::IOTensor> iOutputTensors;
    V2_1::IOTensor iTensor;
    iOutputTensors.emplace_back(iTensor);
    std::vector<std::vector<int32_t>> outputsDims {{0}};
    std::vector<bool> isOutputBufferEnough {};

    std::shared_ptr<V2_1::MockIPreparedModel> sp = std::make_shared<V2_1::MockIPreparedModel>();
    OHOS::sptr<V2_1::IPreparedModel> hdiPreparedModel =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(hdiPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(hdiPreparedModel);
    OH_NN_ReturnCode result = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_002
 * @tc.desc: Verify the Run function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_002, TestSize.Level0)
{
    const size_t length = 100;
    void* buffer = nullptr;
    GetBuffer(buffer, length);

    std::vector<IOTensor> inputs;
    std::vector<IOTensor> outputs;
    InitTensor(inputs, buffer, length);

    OH_NN_ReturnCode result = Run(inputs);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    const auto& memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}

/**
 * @tc.name: hidpreparedmodel_run_003
 * @tc.desc: Verify the Run function return unavailable device in case of run failure.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_003, TestSize.Level0)
{
    const size_t length = 100;
    void* buffer = nullptr;
    GetBuffer(buffer, length);

    std::vector<IOTensor> inputs;
    std::vector<IOTensor> outputs;
    InitTensor(inputs, buffer, length);

    std::vector<std::vector<int32_t>> outputsDims {};
    std::vector<bool> isOutputBufferEnough {};

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);

    EXPECT_CALL(*sp, Run(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(
            ::testing::DoAll(
                ::testing::SetArgReferee<2>(outputsDims),
                ::testing::Return(HDF_FAILURE)
            )
        );

    OH_NN_ReturnCode result = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, result);
    const auto& memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}

/**
 * @tc.name: hidpreparedmodel_run_004
 * @tc.desc: Verify the Run function return invalid parameter.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_004, TestSize.Level0)
{
    std::vector<IOTensor> inputs;
    InitTensor(inputs, nullptr, 0);
    OH_NN_ReturnCode result = Run(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_005
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_005, TestSize.Level0)
{
    const size_t length = 100;
    void* buffer = nullptr;
    GetBuffer(buffer, length);

    std::vector<IOTensor> inputs;
    std::vector<IOTensor> outputs;
    InitTensor(inputs, buffer, length);
    InitTensor(outputs, nullptr, 0);

    std::vector<std::vector<int32_t>> outputsDims {};
    std::vector<bool> isOutputBufferEnough {};

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);

    OH_NN_ReturnCode result = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
    const auto& memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}

/**
 * @tc.name: hidpreparedmodel_run_006
 * @tc.desc: Verify the Run function return success.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_006, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_006");
    const size_t length = 100;
    void* buffer = nullptr;
    GetBuffer(buffer, length);

    std::vector<IOTensor> inputs;
    std::vector<IOTensor> outputs;
    InitTensor(inputs, buffer, length);
    InitTensor(outputs, buffer, length);

    std::vector<std::vector<int32_t>> outputsDims {{0}};
    std::vector<bool> isOutputBufferEnough {};

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);
    EXPECT_CALL(*sp, Run(::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(
                ::testing::SetArgReferee<2>(outputsDims),
            ::testing::Return(HDF_SUCCESS))
        );

    OH_NN_ReturnCode result = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_SUCCESS, result);
    
    const auto& memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}

OH_NN_ReturnCode HDIPreparedModelTest::RunFail(std::vector<IOTensor>& inputs)
{
    std::vector<IOTensor> outputs;
    std::vector<std::vector<int32_t>> outputsDims {};
    std::vector<bool> isOutputBufferEnough {};

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);

    OH_NN_ReturnCode result = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    return result;
}

/**
 * @tc.name: hidpreparedmodel_run_007
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_007, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_007");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_BOOL;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_008
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_008, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_008");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_INT16;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_009
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_009, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_009");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_INT64;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_010
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_010, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_010");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_UINT8;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_011
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_011, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_011");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_UINT16;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_012
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_012, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_012");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_UINT32;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_013
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_013, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_013");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_UINT64;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_014
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_014, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_014");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_FLOAT16;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_015
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_015, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_015");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_FLOAT32;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_016
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_016, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_016");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_FLOAT64;
    inputTensor.format = OH_NN_FORMAT_NHWC;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_017
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_017, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_017");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_UNKNOWN;
    inputTensor.format = OH_NN_FORMAT_NONE;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_018
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_018, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_018");
    std::vector<IOTensor> inputs;
    IOTensor inputTensor;
    inputTensor.dataType = OH_NN_INT32;
    inputs.emplace_back(std::move(inputTensor));

    OH_NN_ReturnCode result = RunFail(inputs);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_run_019
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_019, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_019");
    std::vector<NN_Tensor*> inputs;
    std::vector<NN_Tensor*> outputs;
    std::vector<std::vector<int32_t>> outputsDims {};
    std::vector<bool> isOutputBufferEnough {};

    inputs.emplace_back(nullptr);

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);
    OH_NN_ReturnCode ret = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/**
 * @tc.name: hidpreparedmodel_run_020
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_020, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_020");
    std::vector<NN_Tensor*> inputs;
    std::vector<NN_Tensor*> outputs;
    std::vector<std::vector<int32_t>> outputsDims {};
    std::vector<bool> isOutputBufferEnough {};

    MockTensor* tensorImpl = new (std::nothrow) MockTensor();
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(tensorImpl);
    inputs.emplace_back(tensor);

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);
    OH_NN_ReturnCode ret = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_FAILED, ret);

    testing::Mock::AllowLeak(tensorImpl);
}

/**
 * @tc.name: hidpreparedmodel_run_021
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_021, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_021");
    std::vector<NN_Tensor*> inputs;
    std::vector<NN_Tensor*> outputs;
    std::vector<std::vector<int32_t>> outputsDims {};
    std::vector<bool> isOutputBufferEnough {};

    size_t deviceId = 1;
    NNTensor2_0* tensorImpl = new (std::nothrow) NNTensor2_0(deviceId);
    TensorDesc TensorDesc;

    tensorImpl->SetTensorDesc(&TensorDesc);
    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(tensorImpl);
    inputs.emplace_back(tensor);

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);
    OH_NN_ReturnCode ret = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_FAILED, ret);
}

/**
 * @tc.name: hidpreparedmodel_run_022
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_run_022, TestSize.Level0)
{
    LOGE("Run hidpreparedmodel_run_022");
    std::vector<NN_Tensor*> inputs;
    std::vector<NN_Tensor*> outputs;
    std::vector<std::vector<int32_t>> outputsDims {};
    std::vector<bool> isOutputBufferEnough {};

    size_t backendId = 1;
    NNTensor2_0* nnTensor = new (std::nothrow) NNTensor2_0(backendId);
    EXPECT_NE(nullptr, nnTensor);

    TensorDesc tensorDesc;
    char name = 'a';
    tensorDesc.SetName(&name);
    tensorDesc.SetDataType(OH_NN_UINT32);
    tensorDesc.SetFormat(OH_NN_FORMAT_NCHW);
    int32_t expectDim[2] = {3, 3};
    int32_t* ptr = expectDim;
    uint32_t dimensionCount = 2;
    tensorDesc.SetShape(ptr, dimensionCount);

    OH_NN_ReturnCode retSetTensorDesc = nnTensor->SetTensorDesc(&tensorDesc);
    EXPECT_EQ(OH_NN_SUCCESS, retSetTensorDesc);

    nnTensor->SetSize(200);
    nnTensor->SetOffset(0);
    float m_dataArry[9] {0, 1, 2, 3, 4, 5, 6, 7, 8};
    void* buffer = m_dataArry;
    nnTensor->SetData(buffer);

    NN_Tensor* tensor = reinterpret_cast<NN_Tensor*>(nnTensor);
    inputs.emplace_back(tensor);

    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);
    OH_NN_ReturnCode ret = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_UNAVAILABLE_DEVICE, ret);
}

/**
 * @tc.name: hidpreparedmodel_getmodelid_001
 * @tc.desc: Verify the Run function return invalid parameter in case of output invalid.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_getmodelid_001, TestSize.Level0)
{
    LOGE("GetModelID hidpreparedmodel_getmodelid_001");
    OHOS::sptr<V2_1::MockIPreparedModel> sp =
        OHOS::sptr<V2_1::MockIPreparedModel>(new (std::nothrow) V2_1::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    uint32_t index = 0;
    std::unique_ptr<HDIPreparedModelV2_1> preparedModel = std::make_unique<HDIPreparedModelV2_1>(sp);
    OH_NN_ReturnCode ret = preparedModel->GetModelID(index);
    EXPECT_EQ(OH_NN_SUCCESS, ret);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
