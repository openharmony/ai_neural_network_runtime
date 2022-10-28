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

# CMake lowest version requirement
cmake_minimum_required(VERSION 3.16.5)

# project information
project(label_classification)

set(CMAKE_C_FLAGS "-pthread -fstack-protector-all -fPIC -D_FORTIFY_SOURCE=2")

# If the CPU architecture is 32-bit, -march=armv7-a
set(CMAKE_CXX_FLAGS "-pthread -fstack-protector-all -fPIC -D_FORTIFY_SOURCE=2 -march=armv7-a")
set(LOCAL_DIRECTORY_PATH ${PROJECT_SOURCE_DIR})

add_subdirectory(${LOCAL_DIRECTORY_PATH}/tflite)

