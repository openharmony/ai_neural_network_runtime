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
include_directories(${NNRT_INTERFACE_HOME} ${OHOS_INC})

# Scr path
file(GLOB NNRT_SRCS "${NNRT_INTERFACE_HOME}/*.cpp")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${LOCAL_DIRECTORY_PATH}/lib)
add_library(nnrt_implementation SHARED ${NNRT_SRCS})


