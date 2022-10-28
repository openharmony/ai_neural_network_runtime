#!/bin/bash
#
# Copyright (c) 2022 Huawei Device Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e

function help_info() {
    echo "arm64-v8a(armeabi-v7a) means the CPU architecture is 64-bit(32-bit), the compile command like the following:"
    echo "bash build_ohos_tflite.sh arm64-v8a"
}

function build() {
    echo "$1"
    ./tool_chain/native/build-tools/cmake/bin/cmake \
    -DCMAKE_TOOLCHAIN_FILE=./tool_chain/native/build/cmake/ohos.toolchain.cmake \
    -DOHOS_ARCH=$1 \
    -DOHOS_PLATFORM=OHOS \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DBUILD_SHARED_LIBS=true \
    -DOHOS_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Debug \
    ..
}

if [ "$#" != 1 ]; then
    echo "Incorrect command, please pass the correct number of parameters to the compile command."
    help_info
    exit 1;
fi

if [ "$1" == "arm64-v8a" ]; then
    build arm64-v8a
elif [ "$1" == "armeabi-v7a" ]; then
    build armeabi-v7a
else
    echo "Incorrect CPU architecture parameter or missing setting it, please pass the correct compile command."
    help_info
fi

