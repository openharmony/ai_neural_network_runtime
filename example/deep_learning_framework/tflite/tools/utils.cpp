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

#include "utils.h"

#include <fstream>
#include <sys/time.h>
#include <iostream>

#include "tflite/tools/bitmap_helpers.h"
#include "tflite/tools/get_topn.h"
#include "tflite/tools/log.h"

namespace tflite {
namespace label_classify {
constexpr int32_t DATA_PRINT_NUM = 1000;
constexpr int32_t DATA_EACHLINE_NUM = 1000;
constexpr int32_t SECOND_TO_MICROSECOND_RATIO = 1000000;
constexpr uint8_t WEIGHT_DIMENSION = 2;
constexpr uint8_t CHANNEL_DIMENSION = 3;

double GetUs(struct timeval t)
{
    return (t.tv_sec * SECOND_TO_MICROSECOND_RATIO + t.tv_usec);
}

TfLiteStatus ReadLabelsFile(const string& fileName, std::vector<string>& result, size_t& foundLabelCount)
{
    std::ifstream file(fileName);
    if (!file) {
        LOG(ERROR) << "Labels file " << fileName << " not found";
        return kTfLiteError;
    }
    result.clear();
    string line;
    while (std::getline(file, line)) {
        result.push_back(line);
    }
    foundLabelCount = result.size();
    const int32_t padding = 16;
    while (result.size() % padding) {
        result.emplace_back();
    }

    return kTfLiteOk;
}

void GetInputNameAndShape(string& inputShapeString, std::map<string, std::vector<int>>& userInputShapes)
{
    if (inputShapeString == "") {
        return;
    }
    int pos = inputShapeString.find_last_of(":");
    string userInputName = inputShapeString.substr(0, pos);

    string dimString = inputShapeString.substr(pos + 1);
    int dimPos = dimString.find(",");
    std::vector<int> inputDims;
    while (dimPos != dimString.npos) {
        inputDims.push_back(std::stoi(dimString.substr(0, dimPos)));
        dimString = dimString.substr(dimPos + 1);
        dimPos = dimString.find(",");
    }
    inputDims.push_back(std::stoi(dimString));
    userInputShapes.insert(std::map<string, std::vector<int>>::value_type(userInputName, inputDims));
}

TfLiteStatus FilterDynamicInputs(Settings& settings, std::unique_ptr<tflite::Interpreter>& interpreter,
    std::map<int, std::vector<int>>& neededInputShapes)
{
    std::vector<int> inputIndexes = interpreter->inputs();
    std::map<string, int> nameIndexs;
    for (int i = 0; i < inputIndexes.size(); i++) {
        LOG(INFO) << "input index: " << inputIndexes[i];
        nameIndexs.insert(std::map<string, int>::value_type(interpreter->GetInputName(i), inputIndexes[i]));
    }

    if (settings.inputShape.find(":") == settings.inputShape.npos) {
        LOG(ERROR) << "The format of input shapes string is not supported.";
        return kTfLiteError;
    }

    // Get input names and shapes
    std::map<string, std::vector<int>> userInputShapes;
    string inputShapeString = settings.inputShape;
    int pos = inputShapeString.find(";");
    while (pos != inputShapeString.npos) {
        GetInputNameAndShape(inputShapeString, userInputShapes);
        inputShapeString = inputShapeString.substr(pos + 1);
        pos = inputShapeString.find(";");
    }
    GetInputNameAndShape(inputShapeString, userInputShapes);

    for (const auto& inputShape : userInputShapes) {
        string inputName = inputShape.first;
        auto findName = nameIndexs.find(inputName);
        if (findName == nameIndexs.end()) {
            LOG(ERROR) << "The input name is error: " << inputShape.first << ".";
            return kTfLiteError;
        } else {
            neededInputShapes.insert(std::map<int, std::vector<int>>::value_type(findName->second, inputShape.second));
        }
    }

    return kTfLiteOk;
}

template <class T> void PrintData(T* data, int32_t dataSize, int32_t printSize)
{
    if (printSize > dataSize) {
        printSize = dataSize;
    }
    for (int32_t i = 0; i < printSize; ++i) {
        std::cout << static_cast<float>(*(data + i)) << "\t";
    }
    std::cout << std::endl;
}

void PrintResult(std::unique_ptr<tflite::Interpreter>& interpreter)
{
    for (int32_t index = 0; index < interpreter->outputs().size(); ++index) {
        int32_t output_index = interpreter->outputs()[index];
        TfLiteIntArray* outputsDims = interpreter->tensor(output_index)->dims;
        int32_t dimSize = outputsDims->size;
        int32_t outputTensorSize = 1;
        for (int32_t i = 0; i < dimSize; ++i) {
            outputTensorSize *= outputsDims->data[i];
        }

        TfLiteTensor* outputTensor = interpreter->tensor(output_index);
        switch (outputTensor->type) {
            case kTfLiteFloat32:
                PrintData<float>(interpreter->typed_output_tensor<float>(index), outputTensorSize, DATA_PRINT_NUM);
                break;
            case kTfLiteInt32:
                PrintData<int32_t>(interpreter->typed_output_tensor<int32_t>(index), outputTensorSize, DATA_PRINT_NUM);
                break;
            case kTfLiteUInt8:
                PrintData<uint8_t>(interpreter->typed_output_tensor<uint8_t>(index), outputTensorSize, DATA_PRINT_NUM);
                break;
            case kTfLiteInt8:
                PrintData<int8_t>(interpreter->typed_output_tensor<int8_t>(index), outputTensorSize, DATA_PRINT_NUM);
                break;
            default:
                LOG(ERROR) << "Unsupportted tensor datatype: " << outputTensor->type << "!";
                return;
        }
    }
}

void AnalysisResults(Settings& settings, std::unique_ptr<tflite::Interpreter>& interpreter)
{
    const float threshold = 0.001f;
    std::vector<std::pair<float, int32_t>> topResults;

    if (settings.printResult) {
        LOG(INFO) << "Outputs Data:";
        PrintResult(interpreter);
    }

    int32_t output = interpreter->outputs()[0];
    TfLiteIntArray* outputDims = interpreter->tensor(output)->dims;
    // assume output dims to be something like (1, 1, ... ,size)
    auto outputSize = outputDims->data[outputDims->size - 1];

    auto tfType = interpreter->tensor(output)->type;
    switch (tfType) {
        case kTfLiteFloat32:
            GetTopN<float>(interpreter->typed_output_tensor<float>(0), outputSize, settings.numberOfResults, threshold,
                &topResults, settings.inputType);
            break;
        case kTfLiteInt8:
            GetTopN<int8_t>(interpreter->typed_output_tensor<int8_t>(0), outputSize, settings.numberOfResults,
                threshold, &topResults, settings.inputType);
            break;
        case kTfLiteUInt8:
            GetTopN<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0), outputSize, settings.numberOfResults,
                threshold, &topResults, settings.inputType);
            break;
        case kTfLiteInt64:
            GetTopN<int64_t>(interpreter->typed_output_tensor<int64_t>(0), outputSize, settings.numberOfResults,
                threshold, &topResults, settings.inputType);
            break;
        default:
            LOG(ERROR) << "cannot handle output type " << tfType << " yet";
            return;
    }

    std::vector<string> labels;
    size_t labelCount;

    if (ReadLabelsFile(settings.labelsFileName, labels, labelCount) != kTfLiteOk) {
        return;
    }
    for (const auto& result : topResults) {
        const float confidence = result.first;
        const int32_t index = result.second;
        LOG(INFO) << confidence << ": " << index << " " << labels[index];
    }
}

void ImportData(Settings& settings, std::vector<int>& imageSize, std::unique_ptr<tflite::Interpreter>& interpreter)
{
    ImageInfo inputImageInfo = {imageSize[0], imageSize[1], imageSize[2]};
    std::vector<uint8_t> in;
    ReadBmp(settings.inputBmpName, inputImageInfo, &settings, in);

    int32_t input = interpreter->inputs()[0];
    if (settings.verbose) {
        LOG(INFO) << "input: " << input;
    }

    // get input dimension from the model.
    TfLiteIntArray* dims = interpreter->tensor(input)->dims;
    ImageInfo wantedimageInfo;
    wantedimageInfo.height = dims->data[1];
    wantedimageInfo.width = dims->data[WEIGHT_DIMENSION];
    wantedimageInfo.channels = (dims->size > CHANNEL_DIMENSION) ? dims->data[CHANNEL_DIMENSION] : 1;

    settings.inputType = interpreter->tensor(input)->type;
    switch (settings.inputType) {
        case kTfLiteFloat32:
            Resize<float>(interpreter->typed_tensor<float>(input), in.data(), inputImageInfo, wantedimageInfo,
                &settings);
            break;
        case kTfLiteInt8:
            Resize<int8_t>(interpreter->typed_tensor<int8_t>(input), in.data(), inputImageInfo, wantedimageInfo,
                &settings);
            break;
        case kTfLiteUInt8:
            Resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(), inputImageInfo, wantedimageInfo,
                &settings);
            break;
        case kTfLiteInt64:
            Resize<int64_t>(interpreter->typed_tensor<int64_t>(input), in.data(), inputImageInfo, wantedimageInfo,
                &settings);
            break;
        default:
            LOG(ERROR) << "cannot handle input type " << settings.inputType << " yet";
            return;
    }
}

bool IsEqualShape(int tensorIndex, const std::vector<int>& dims, std::unique_ptr<tflite::Interpreter>& interpreter)
{
    TfLiteTensor* tensor = interpreter->tensor(tensorIndex);
    for (int i = 0; i < tensor->dims->size; ++i) {
        if (tensor->dims->data[i] != dims[i]) {
            return false;
        }
    }
    return true;
}
} // namespace label_classify
} // namespace tflite