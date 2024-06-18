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

#include <gtest/gtest.h>

#include "quant_param.h"

using namespace testing;
using namespace testing::ext;
using namespace OHOS::NeuralNetworkRuntime;

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace UnitTest {
class QuantParamsTest : public testing::Test {
public:
    QuantParamsTest() = default;
    ~QuantParamsTest() = default;
};

/**
 * @tc.name: quantparamstest_setscales_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(QuantParamsTest, quantparamstest_setscales_001, TestSize.Level0)
{
    QuantParams quantParams;
    std::vector<double> scales = {1, 2, 3, 4};
    quantParams.SetScales(scales);
    EXPECT_EQ(false, quantParams.GetScales().empty());
}

/**
 * @tc.name: quantparamstest_setscales_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(QuantParamsTest, quantparamstest_setscales_002, TestSize.Level0)
{
    QuantParams quantParams;
    EXPECT_EQ(true, quantParams.GetScales().empty());
}

/**
 * @tc.name: quantparamstest_setzeropoints_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(QuantParamsTest, quantparamstest_setzeropoints_001, TestSize.Level0)
{
    QuantParams quantParams;
    std::vector<int32_t> zeroPoints = {1, 2, 3, 4};
    quantParams.SetZeroPoints(zeroPoints);
    EXPECT_EQ(false, quantParams.GetZeroPoints().empty());
}

/**
 * @tc.name: quantparamstest_setzeropoints_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(QuantParamsTest, quantparamstest_setzeropoints_002, TestSize.Level0)
{
    QuantParams quantParams;
    EXPECT_EQ(true, quantParams.GetZeroPoints().empty());
}

/**
 * @tc.name: quantparamstest_setnumbits_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(QuantParamsTest, quantparamstest_setnumbits_001, TestSize.Level0)
{
    QuantParams quantParams;
    std::vector<uint32_t> numBits = {1, 2, 3, 4};
    quantParams.SetNumBits(numBits);
    EXPECT_EQ(false, quantParams.GetNumBits().empty());
}

/**
 * @tc.name: quantparamstest_setnumbits_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(QuantParamsTest, quantparamstest_setnumbits_002, TestSize.Level0)
{
    QuantParams quantParams;
    EXPECT_EQ(true, quantParams.GetNumBits().empty());
}

/**
 * @tc.name: quantparamstest_copytocompat_001
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(QuantParamsTest, quantparamstest_copytocompat_001, TestSize.Level0)
{
    QuantParams quantParams;
    std::vector<double> scales = {1, 2};
    quantParams.SetScales(scales);
    std::vector<int32_t> zeroPoints = {1, 2, 3};
    quantParams.SetZeroPoints(zeroPoints);
    std::vector<uint32_t> numBits = {1, 2, 3, 4};
    quantParams.SetNumBits(numBits);
    std::vector<OHOS::NeuralNetworkRuntime::QuantParam> compatQuantParams;
    EXPECT_EQ(OH_NN_INVALID_PARAMETER, quantParams.CopyToCompat(compatQuantParams));
}

/**
 * @tc.name: quantparamstest_copytocompat_002
 * @tc.desc: Verify the QuantParams function return nullptr in case of fd -1.
 * @tc.type: FUNC
 */
HWTEST_F(QuantParamsTest, quantparamstest_copytocompat_002, TestSize.Level0)
{
    QuantParams quantParams;
    std::vector<double> scales = {1, 2, 3, 4};
    quantParams.SetScales(scales);
    std::vector<int32_t> zeroPoints = {1, 2, 3, 4};
    quantParams.SetZeroPoints(zeroPoints);
    std::vector<uint32_t> numBits = {1, 2, 3, 4};
    quantParams.SetNumBits(numBits);
    std::vector<OHOS::NeuralNetworkRuntime::QuantParam> compatQuantParams;
    EXPECT_EQ(OH_NN_SUCCESS, quantParams.CopyToCompat(compatQuantParams));
}

} // namespace UnitTest
} // namespace NeuralNetworkRuntime
} // namespace OHOS