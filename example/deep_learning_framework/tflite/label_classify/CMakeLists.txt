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

# Header path
set(OHOS_INC ${LOCAL_DIRECTORY_PATH}/../../interfaces/kits/c)
set(TOOLS_INC ${LOCAL_DIRECTORY_PATH}/tflite/tools)
set(TFLITE_INC ${LOCAL_DIRECTORY_PATH}/lib_3rd_nnrt_tflite/include)
set(TFLITE_FLATBUFFER_INC ${LOCAL_DIRECTORY_PATH}/lib_3rd_nnrt_tflite/include/tensorflow/lite)
include_directories(${NNRT_DEMO_HOME} ${TFLITE_INC} ${OHOS_INC} ${TOOLS_INC} ${TFLITE_FLATBUFFER_INC} ${LOCAL_DIRECTORY_PATH})

# Scr path
aux_source_directory(${NNRT_DEMO_HOME} NNRT_DEMO_SRCS)
file(GLOB TOOLS_SRCS "${TOOLS_INC}/*.cpp")

LINK_DIRECTORIES(${TFLITE_LIB_PATH}/com/arm64-v8a/lib/)
add_executable(label_classify ${NNRT_DEMO_SRCS} ${TOOLS_SRCS})

set(CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
target_link_libraries(label_classify ${LOCAL_DIRECTORY_PATH}/lib/libnnrt_implementation.so)
target_link_libraries(label_classify ${LOCAL_DIRECTORY_PATH}/lib/libnnrt_delegate.so)
target_link_libraries(label_classify -ltensorflow-lite)

set (EXECUTABLE_OUTPUT_PATH ${LOCAL_DIRECTORY_PATH}/output)




