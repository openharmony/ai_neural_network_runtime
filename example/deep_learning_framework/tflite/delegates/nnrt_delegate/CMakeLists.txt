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

set(TFLITE_PATH ${LOCAL_DIRECTORY_PATH}/lib_3rd_nnrt_tflite)

LINK_DIRECTORIES(${TFLITE_PATH}/com/arm64-v8a/lib/)

# Header path
set(OHOS_INC ${LOCAL_DIRECTORY_PATH}/../../interfaces/kits/c)
set(TOOLS_INC ${LOCAL_DIRECTORY_PATH}/tflite/tools)
set(TFLITE_INC ${LOCAL_DIRECTORY_PATH}/lib_3rd_nnrt_tflite/include)
include_directories(${LOCAL_DIRECTORY_PATH} ${NNRT_DELEGATE_HOME} ${TFLITE_INC} ${OHOS_INC} ${TOOLS_INC})

# Scr path
file(GLOB NNRT_DELEGATE_SRCS "${NNRT_DELEGATE_HOME}/*.cpp")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LOCAL_DIRECTORY_PATH}/lib)

add_library(nnrt_delegate SHARED ${NNRT_DELEGATE_SRCS})
target_link_libraries(nnrt_delegate -ltensorflow-lite nnrt_implementation)


