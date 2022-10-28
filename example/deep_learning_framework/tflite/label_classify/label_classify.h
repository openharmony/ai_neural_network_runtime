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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_H
#define TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_H

#include <iostream>

#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/c/c_api_types.h"

namespace tflite {
namespace label_classify {
struct Settings {
    tflite::FlatBufferModel* model;
    bool verbose = false;
    bool accel = false;
    bool printResult = false;
    TfLiteType inputType = kTfLiteFloat32;
    int32_t loopCount = 1;
    float inputMean = 127.5f;
    float inputStd = 127.5f;
    string modelName = "./mbv2.tflite";
    string inputBmpName = "./grace_hopper.bmp";
    string labelsFileName = "./labels.txt";
    string inputShape = "";
    int32_t numberOfThreads = 1;
    int32_t numberOfResults = 5;
    int32_t numberOfWarmupRuns = 0;
};
} // namespace label_classify
} // namespace tflite

#endif // TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_H
