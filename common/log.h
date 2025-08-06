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

#ifndef NEURAL_NETWORK_RUNTIME_LOG_H
#define NEURAL_NETWORK_RUNTIME_LOG_H

#include <cstdarg>
#include "hilog/log.h"

#ifdef __cplusplus
extern "C" {
#endif

#undef LOG_DOMAIN
#define LOG_DOMAIN 0xD002101

#undef LOG_TAG
#define LOG_TAG "NNRt"

#define R_FILENAME (__builtin_strrchr(__FILE_NAME__, '/') ? __builtin_strrchr(__FILE_NAME__, '/') + 1: __FILE_NAME__)

#define LOGD(fmt, ...)                                                                                     \
    ((void)HILOG_IMPL(LOG_CORE, LOG_DEBUG, LOG_DOMAIN, LOG_TAG, "[%{public}s(%{public}s:%{public}d)]" fmt, \
        R_FILENAME, __FUNCTION__, __LINE__, ##__VA_ARGS__))

#define LOGI(fmt, ...)                                                                                     \
    ((void)HILOG_IMPL(LOG_CORE, LOG_INFO, LOG_DOMAIN, LOG_TAG, "[%{public}s(%{public}s:%{public}d)]" fmt, \
        R_FILENAME, __FUNCTION__, __LINE__, ##__VA_ARGS__))

#define LOGW(fmt, ...)                                                                                     \
    ((void)HILOG_IMPL(LOG_CORE, LOG_WARN, LOG_DOMAIN, LOG_TAG, "[%{public}s(%{public}s:%{public}d)]" fmt, \
        R_FILENAME, __FUNCTION__, __LINE__, ##__VA_ARGS__))

#define LOGE(fmt, ...)                                                                                     \
    ((void)HILOG_IMPL(LOG_CORE, LOG_ERROR, LOG_DOMAIN, LOG_TAG, "[%{public}s(%{public}s:%{public}d)]" fmt, \
        R_FILENAME, __FUNCTION__, __LINE__, ##__VA_ARGS__))

#define LOGF(fmt, ...)                                                                                     \
    ((void)HILOG_IMPL(LOG_CORE, LOG_FATAL, LOG_DOMAIN, LOG_TAG, "[%{public}s(%{public}s:%{public}d)]" fmt, \
        R_FILENAME, __FUNCTION__, __LINE__, ##__VA_ARGS__))

#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_RUNTIME_LOG_H
