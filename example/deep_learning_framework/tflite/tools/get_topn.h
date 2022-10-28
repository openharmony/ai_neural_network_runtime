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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_GET_TOP_N_H
#define TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_GET_TOP_N_H

#include <algorithm>
#include <functional>
#include <queue>

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace label_classify {
template <class T>
void GetTopN(T* prediction, int32_t predictionSize, size_t numResults, float threshold,
    std::vector<std::pair<float, int32_t>>* topResults, TfLiteType inputType)
{
    // Will contain top N results in ascending order.
    std::priority_queue<std::pair<float, int32_t>, std::vector<std::pair<float, int32_t>>,
        std::greater<std::pair<float, int32_t>>>
        topResultPQ;

    const long count = predictionSize; // NOLINT(runtime/int32_t)
    float value = 0.0;
    float intNormalizedFactor = 256.0;
    float uintNormalizedFactor = 255.0;
    uint32_t offsetNumber = 128;

    for (int32_t i = 0; i < count; ++i) {
        switch (inputType) {
            case kTfLiteFloat32:
                value = prediction[i];
                break;
            case kTfLiteInt8:
                value = (prediction[i] + offsetNumber) / intNormalizedFactor;
                break;
            case kTfLiteUInt8:
                value = prediction[i] / uintNormalizedFactor;
                break;
            default:
                break;
        }

        // Only add it if it beats the threshold and has a chance at being in the top N.
        if (value < threshold) {
            continue;
        }

        topResultPQ.push(std::pair<float, int32_t>(value, i));

        // If at capacity, kick the smallest value out.
        if (topResultPQ.size() > numResults) {
            topResultPQ.pop();
        }
    }

    // Copy to output vector and reverse into descending order.
    while (!topResultPQ.empty()) {
        topResults->push_back(topResultPQ.top());
        topResultPQ.pop();
    }

    std::reverse(topResults->begin(), topResults->end());
}

// explicit instantiation so that we can use them otherwhere
template void GetTopN<float>(float*, int32_t, size_t, float, std::vector<std::pair<float, int32_t>>*, TfLiteType);
template void GetTopN<int8_t>(int8_t*, int32_t, size_t, float, std::vector<std::pair<float, int32_t>>*, TfLiteType);
template void GetTopN<uint8_t>(uint8_t*, int32_t, size_t, float, std::vector<std::pair<float, int32_t>>*, TfLiteType);
template void GetTopN<int64_t>(int64_t*, int32_t, size_t, float, std::vector<std::pair<float, int32_t>>*, TfLiteType);
}  // namespace label_classify
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_GET_TOP_N_H
