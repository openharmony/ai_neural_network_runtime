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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_BITMAP_HELPERS_H
#define TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_BITMAP_HELPERS_H

#include "tflite/label_classify/label_classify.h"
#include "tensorflow/lite/builtin_op_data.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/string_util.h"
#include "log.h"


namespace tflite {
namespace label_classify {
const int INPUT_NUMBER = 2;
const int OUPUT_NUMBER = 1;
const int INT8_OFFSET_NUMBER = 128;
const int BIT_TO_BYTE = 8;
const int BLUE_OFFSET = 2;
const int GREEN_OFFSET = 1;
const int ALPHA_OFFSET = 3;
const int HEADER_ADDRESS_OFFSET = 10;
const int WIDTH_ADDRESS_OFFSET = 18;
const int HEIGHT_ADDRESS_OFFSET = 22;
const int BBP_ADDRESS_OFFSET = 28;
enum ChannelDim : int {
    GRAYSCALE_DIM = 1,
    BGR_DIM = 3,
    BGRA_DIM = 4
};

struct BmpAddressOffset {
    int headerAddressOffset = 0;
    int widthAddressOffset = 0;
    int heightAddressOffset = 0;
    int bbpAddressOffset = 0;
};

struct ColorChannelOffset {
    int blueOffset = 0;
    int greenOffset = 0;
    int alphaOffset = 0;
};

struct ImageInfo {
    int32_t width = 0;
    int32_t height = 0;
    int32_t channels = 0;
};

void ReadBmp(const std::string& input_bmp_name, ImageInfo& imageInfo, Settings* s, std::vector<uint8_t>& input);

template <typename T>
void Resize(T* out, uint8_t* in, ImageInfo inputImageInfo, ImageInfo wantedImageInfo, Settings* s)
{
    std::unique_ptr<Interpreter> interpreter = std::make_unique<Interpreter>();

    int32_t baseIndex = 0;
    int32_t outputIndex = 2;

    // two inputs: input and new_sizes
    interpreter->AddTensors(INPUT_NUMBER, &baseIndex);
    // one output
    interpreter->AddTensors(OUPUT_NUMBER, &baseIndex);
    // set input and output tensors
    interpreter->SetInputs({ 0, 1 });
    interpreter->SetOutputs({ 2 });

    // set parameters of tensors
    TfLiteQuantizationParams quant;
    interpreter->SetTensorParametersReadWrite(0, kTfLiteFloat32, "input",
        { 1, inputImageInfo.height, inputImageInfo.width, inputImageInfo.channels }, quant);
    interpreter->SetTensorParametersReadWrite(1, kTfLiteInt32, "new_size", { 2 }, quant);
    interpreter->SetTensorParametersReadWrite(outputIndex, kTfLiteFloat32, "output",
        { 1, wantedImageInfo.height, wantedImageInfo.width, wantedImageInfo.channels }, quant);

    ops::builtin::BuiltinOpResolver resolver;
    const TfLiteRegistration* resizeOp = resolver.FindOp(BuiltinOperator_RESIZE_BILINEAR, 1);
    auto* params = reinterpret_cast<TfLiteResizeBilinearParams*>(malloc(sizeof(TfLiteResizeBilinearParams)));
    if (params == nullptr) {
        LOG(ERROR) << "Malloc memory failed in BitmapHelperslmpl.";
        return;
    }
    params->align_corners = false;
    params->half_pixel_centers = false;
    interpreter->AddNodeWithParameters({ 0, 1 }, { 2 }, nullptr, 0, params, resizeOp, nullptr);
    interpreter->AllocateTensors();

    // fill input image
    // in[] are integers, cannot do memcpy() directly
    auto input = interpreter->typed_tensor<float>(0);

    for (int32_t i = 0; i < inputImageInfo.height * inputImageInfo.width * inputImageInfo.channels; ++i) {
        input[i] = in[i];
    }

    // fill new_sizes
    interpreter->typed_tensor<int32_t>(1)[0] = wantedImageInfo.height;
    interpreter->typed_tensor<int32_t>(1)[1] = wantedImageInfo.width;
    interpreter->Invoke();
    auto output = interpreter->typed_tensor<float>(2);
    for (int32_t i = 0; i < wantedImageInfo.height * wantedImageInfo.width * wantedImageInfo.channels; ++i) {
        switch (s->inputType) {
            case kTfLiteFloat32:
                out[i] = (output[i] - s->inputMean) / s->inputStd;
                break;
            case kTfLiteInt8:
                out[i] = static_cast<int8_t>(output[i] - INT8_OFFSET_NUMBER);
                break;
            case kTfLiteUInt8:
                out[i] = static_cast<uint8_t>(output[i]);
                break;
            default:
                break;
        }
    }
}

// explicit instantiation
template void Resize<float>(float*, uint8_t*, ImageInfo, ImageInfo, Settings*);
template void Resize<int8_t>(int8_t*, uint8_t*, ImageInfo, ImageInfo, Settings*);
template void Resize<uint8_t>(uint8_t*, uint8_t*, ImageInfo, ImageInfo, Settings*);
template void Resize<int64_t>(int64_t*, uint8_t*, ImageInfo, ImageInfo, Settings*);
}  // namespace label_classify
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_LABEL_CLASSIFY_BITMAP_HELPERS_H
