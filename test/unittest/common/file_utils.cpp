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

#include "file_utils.h"

#include <unistd.h>
#include <fstream>

#include "common/log.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
FileUtils::FileUtils(const std::string &filename) :m_filename(filename)
{
}

FileUtils::~FileUtils()
{
    if (!m_filename.empty()) {
        int ret = unlink(m_filename.c_str());
        if (ret != 0) {
            LOGE("Failed to delete file: %s.", m_filename.c_str());
        }
    }
}

bool FileUtils::WriteFile(const std::string &data)
{
    std::ofstream outFile(m_filename);
    if (!outFile.is_open()) {
        LOGE("Failed to open file: %s.", m_filename.c_str());
        return false;
    }
    outFile.write(data.c_str(), data.length());
    outFile.close();
    return true;
}
} // namespace NeuralNetworkRuntime
} // namespace OHOS
