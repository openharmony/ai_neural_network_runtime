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
#include "frameworks/native/hdi_prepared_model.h"
#include "frameworks/native/memory_manager.h"
#include "frameworks/native/transform.h"
#include "test/unittest/common/mock_idevice.h"
#include "test/unittest/common/file_utils.h"

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
    const int vPosition = 3;
    std::vector<IOTensor> outputs;
    std::vector<std::vector<int32_t>> outputsDims {{0}};
    std::vector<bool> isOutputBufferEnough {};

    OHOS::sptr<V1_0::MockIPreparedModel> sp =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(sp);
    EXPECT_CALL(*sp, Run(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(::testing::DoAll(
            ::testing::SetArgReferee<vvPosition>(outputsDims),
            ::testing::SetArgReferee<vPosition>(isOutputBufferEnough),
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
    OHOS::sptr<V1_0::IPreparedModel> hdiPreparedModel =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    EXPECT_NE(hdiPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(hdiPreparedModel);
    EXPECT_NE(preparedModel, nullptr);
}

/**
 * @tc.name: hidpreparedmodel_exportmodelcache_001
 * @tc.desc: Verify the ExportModelCache function return memory error.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_exportmodelcache_001, TestSize.Level0)
{
    std::vector<V1_0::SharedBuffer> bufferVect = {{100, 100, 0, 100}};
    OHOS::sptr<V1_0::IPreparedModel> hdiPreparedModel =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(hdiPreparedModel);
    std::vector<ModelBuffer> modelCache;
    EXPECT_CALL(*((V1_0::MockIPreparedModel*)hdiPreparedModel.GetRefPtr()),
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
    std::vector<V1_0::SharedBuffer> bufferVect;
    OHOS::sptr<V1_0::IPreparedModel> mockPreparedModel =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    EXPECT_NE(mockPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(mockPreparedModel);
    std::vector<ModelBuffer> modelCache;
    EXPECT_CALL(*((V1_0::MockIPreparedModel*)mockPreparedModel.GetRefPtr()),
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
    OHOS::sptr<V1_0::IPreparedModel> hdiPreparedModel =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    EXPECT_NE(hdiPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(hdiPreparedModel);
    std::vector<ModelBuffer> modelCache {{nullptr, 0}};
    OH_NN_ReturnCode result = preparedModel->ExportModelCache(modelCache);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
}

/**
 * @tc.name: hidpreparedmodel_exportmodelcache_004
 * @tc.desc: Verify the ExportModelCache function return unvailable device.
 * @tc.type: FUNC
 */
HWTEST_F(HDIPreparedModelTest, hidpreparedmodel_exportmodelcache_004, TestSize.Level0)
{
    std::vector<V1_0::SharedBuffer> bufferVect = {{100, 100, 0, 100}};
    OHOS::sptr<V1_0::IPreparedModel> mockPreparedModel =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    EXPECT_NE(mockPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(mockPreparedModel);
    std::vector<ModelBuffer> modelCache;
    EXPECT_CALL(*((V1_0::MockIPreparedModel*)mockPreparedModel.GetRefPtr()),
        ExportModelCache(::testing::_))
        .WillRepeatedly(
            ::testing::DoAll(
                ::testing::SetArgReferee<0>(bufferVect),
                ::testing::Return(HDF_FAILURE)
            )
        );

    OH_NN_ReturnCode result = preparedModel->ExportModelCache(modelCache);
    EXPECT_EQ(OH_NN_UNAVALIDABLE_DEVICE, result);
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

    std::vector<V1_0::IOTensor> iOutputTensors;
    V1_0::IOTensor iTensor;
    iOutputTensors.emplace_back(iTensor);
    std::vector<std::vector<int32_t>> outputsDims {{0}};
    std::vector<bool> isOutputBufferEnough {};

    std::shared_ptr<V1_0::MockIPreparedModel> sp = std::make_shared<V1_0::MockIPreparedModel>();
    OHOS::sptr<V1_0::IPreparedModel> hdiPreparedModel =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    EXPECT_NE(hdiPreparedModel, nullptr);

    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(hdiPreparedModel);
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

    OHOS::sptr<V1_0::MockIPreparedModel> sp =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(sp);

    EXPECT_CALL(*sp, Run(::testing::_, ::testing::_, ::testing::_, ::testing::_))
        .WillRepeatedly(
            ::testing::DoAll(
                ::testing::SetArgReferee<2>(outputsDims),
                ::testing::SetArgReferee<3>(isOutputBufferEnough),
                ::testing::Return(HDF_FAILURE)
            )
        );

    OH_NN_ReturnCode result = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_UNAVALIDABLE_DEVICE, result);
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

    OHOS::sptr<V1_0::MockIPreparedModel> sp =
        OHOS::sptr<V1_0::MockIPreparedModel>(new (std::nothrow) V1_0::MockIPreparedModel());
    EXPECT_NE(sp, nullptr);

    std::unique_ptr<HDIPreparedModel> preparedModel = std::make_unique<HDIPreparedModel>(sp);

    OH_NN_ReturnCode result = preparedModel->Run(inputs, outputs, outputsDims, isOutputBufferEnough);
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, result);
    const auto& memoryManager = MemoryManager::GetInstance();
    memoryManager->UnMapMemory(buffer);
}
} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS
