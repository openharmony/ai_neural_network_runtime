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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_UTILS_H
#define TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_UTILS_H

#include "../label_classify/label_classify.h"

#include "sys/time.h"

#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/string_type.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/interpreter.h"

#include "neural_network_runtime.h"

namespace tflite {
namespace label_classify {
double GetUs(struct timeval t);
TfLiteStatus ReadLabelsFile(const string& fileName, std::vector<string>& result, size_t& foundLabelCount);
TfLiteStatus FilterDynamicInputs(Settings& settings,
    std::unique_ptr<tflite::Interpreter>& interpreter, std::map<int, std::vector<int>>& neededInputShapes);
bool IsEqualShape(int tensorIndex, const std::vector<int>& dim, std::unique_ptr<tflite::Interpreter>& interpreter);
void GetInputNameAndShape(string &inputShapeString, std::map<string, std::vector<int>>& userInputShapes);
void PrintResult(std::unique_ptr<tflite::Interpreter>& interpreter);
void AnalysisResults(Settings& settings, std::unique_ptr<tflite::Interpreter>& interpreter);
void ImportData(Settings& settings, std::vector<int>& imageSize, std::unique_ptr<tflite::Interpreter>& interpreter);
} // namespace label_classify
} // namespace tflite

#endif // TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_UTILS_H
