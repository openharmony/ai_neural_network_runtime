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
#ifndef CONST_H
#define CONST_H

#include <string>
#include <vector>

namespace OHOS {
namespace NeuralNetworkRuntime {
namespace Test {

const uint32_t ADD_DATA_LENGTH = 4 * sizeof(float);
const uint32_t AVG_INPUT_LENGTH = 9 * sizeof(float);
const std::vector<int32_t> TENSOR_SHAPE = {2, 2, 2, 2};
const std::vector<int32_t> PARAM_INDEX = {2};
const std::vector<int32_t> INPUT_INDEX = {0, 1};
const std::vector<int32_t> OUTPUT_INDEX = {3};
const int32_t ELEMENT_COUNT = 4;

const std::string CACHE_DIR = "/data/local/tmp/";
const std::string CACHE_FILE = "/0.nncache";
const std::string CACHE_INFO_FILE = "/cache_info.nncache";
const uint32_t NO_DEVICE_COUNT = 0;
const int STRESS_COUNT = 10000;
const int PRINT_FREQ = 500;

const size_t MODEL_SIZE = 100;
const size_t ZERO = 0;
const uint32_t CACHEVERSION = 1;
const std::string SUPPORTMODELPATH = "modelPath";
const unsigned short TEST_BUFFER[14] = {
    0x0000, 0x1021, 0x2042, 0x3063, 0x4084, 0x50a5, 0x60c6, 0x70e7, 0x8108, 0x9129, 0xa14a, 0xb16b, 0xc18c, 0xd1ad
};
} // namespace Test
} // namespace NeuralNetworkRuntime
} // namespace OHOS

#endif // CONST_H