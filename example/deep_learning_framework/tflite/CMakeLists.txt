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

set(NNRT_INTERFACE_HOME ${LOCAL_DIRECTORY_PATH}/tflite/nnrt)
set(NNRT_DELEGATE_HOME ${LOCAL_DIRECTORY_PATH}/tflite/delegates/nnrt_delegate)
set(NNRT_DEMO_HOME ${LOCAL_DIRECTORY_PATH}/tflite/label_classify)
set(TFLITE_LIB_PATH ${LOCAL_DIRECTORY_PATH}/lib_3rd_nnrt_tflite)

add_subdirectory(${NNRT_INTERFACE_HOME})
add_subdirectory(${NNRT_DELEGATE_HOME})
add_subdirectory(${NNRT_DEMO_HOME})


