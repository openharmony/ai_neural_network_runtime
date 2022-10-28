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

#include "tflite/tools/bitmap_helpers.h"

#include <fstream>
#include <iostream>

#include "tflite/tools/log.h"

namespace tflite {
namespace label_classify {
void DecodeBmp(const uint8_t* input, int32_t rowSize, ImageInfo imageInfo, bool topDown, std::vector<uint8_t>& output)
{
    ColorChannelOffset colorChannelOffset = { BLUE_OFFSET, GREEN_OFFSET, ALPHA_OFFSET };
    for (int32_t i = 0; i < imageInfo.height; ++i) {
        int32_t srcPos;
        int32_t dstPos;

        for (int32_t j = 0; j < imageInfo.width; j++) {
            if (!topDown) {
                srcPos = ((imageInfo.height - 1 - i) * rowSize) + j * imageInfo.channels;
            } else {
                srcPos = i * rowSize + j * imageInfo.channels;
            }

            dstPos = (i * imageInfo.width + j) * imageInfo.channels;

            switch (imageInfo.channels) {
                case GRAYSCALE_DIM:
                    output[dstPos] = input[srcPos];
                    break;
                case BGR_DIM:
                    // BGR -> RGB
                    output[dstPos] = input[srcPos + colorChannelOffset.blueOffset];
                    output[dstPos + colorChannelOffset.greenOffset] = input[srcPos + colorChannelOffset.greenOffset];
                    output[dstPos + colorChannelOffset.blueOffset] = input[srcPos];
                    break;
                case BGRA_DIM:
                    // BGRA -> RGBA
                    output[dstPos] = input[srcPos + colorChannelOffset.blueOffset];
                    output[dstPos + colorChannelOffset.greenOffset] = input[srcPos + colorChannelOffset.greenOffset];
                    output[dstPos + colorChannelOffset.blueOffset] = input[srcPos];
                    output[dstPos + colorChannelOffset.alphaOffset] = input[srcPos + colorChannelOffset.alphaOffset];
                    break;
                default:
                    LOG(FATAL) << "Unexpected number of channels: " << imageInfo.channels;
                    break;
            }
        }
    }
    return;
}

void ReadBmp(const std::string& inputBmpName, ImageInfo& imageInfo, Settings* s, std::vector<uint8_t>& inputImage)
{
    int32_t begin, end;
    std::ifstream file(inputBmpName, std::ios::in | std::ios::binary);
    if (!file) {
        LOG(FATAL) << "input file " << inputBmpName << " not found";
        return;
    }

    begin = file.tellg();
    file.seekg(0, std::ios::end);
    end = file.tellg();
    size_t len = end - begin;
    if (s->verbose) {
        LOG(INFO) << "len: " << len;
    }

    std::vector<uint8_t> img_bytes(len);
    BmpAddressOffset bmpAddressOffset = { HEADER_ADDRESS_OFFSET, WIDTH_ADDRESS_OFFSET,
        HEIGHT_ADDRESS_OFFSET, BBP_ADDRESS_OFFSET };
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(img_bytes.data()), len);
    const int32_t headerSize =
        *(reinterpret_cast<const int32_t*>(img_bytes.data() + bmpAddressOffset.headerAddressOffset));
    imageInfo.width = *(reinterpret_cast<const int32_t*>(img_bytes.data() + bmpAddressOffset.widthAddressOffset));
    imageInfo.height =
        abs(*(reinterpret_cast<const int32_t*>(img_bytes.data() + bmpAddressOffset.heightAddressOffset)));
    const int32_t bpp = *(reinterpret_cast<const int32_t*>(img_bytes.data() + bmpAddressOffset.bbpAddressOffset));
    imageInfo.channels = bpp / BIT_TO_BYTE;
    inputImage.resize(imageInfo.height * imageInfo.width * imageInfo.channels);

    if (s->verbose) {
        LOG(INFO) << "width, height, channels: " << imageInfo.width << ", " << imageInfo.height << ", "
            << imageInfo.channels;
    }

    // there may be padding bytes when the width is not a multiple of 4 bytes.
    // 8 * channels == bits per pixel
    const int32_t rowSize = ((8 * imageInfo.channels * imageInfo.width + 31) >> 5) << 2;

    // if height is negative, data layout is top down. otherwise, it's bottom up.
    bool topDown = (imageInfo.height < 0);

    // Decode image, allocating tensor once the image size is known.
    const uint8_t* bmpPixels = &img_bytes[headerSize];
    DecodeBmp(bmpPixels, rowSize, imageInfo, topDown, inputImage);
    return;
}
} // namespace label_classify
} // namespace tflite
