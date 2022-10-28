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

#include <stdarg.h>
#include <stdbool.h>
#include "hilog/log_c.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NNR_LOG_DOMAIN
#define NNR_LOG_DOMAIN 0xD002101
#endif

#define LOGD(...) HiLogPrint(LOG_CORE, LOG_DEBUG, NNR_LOG_DOMAIN, "NNRt", __VA_ARGS__)
#define LOGI(...) HiLogPrint(LOG_CORE, LOG_INFO, NNR_LOG_DOMAIN, "NNRt", __VA_ARGS__)
#define LOGW(...) HiLogPrint(LOG_CORE, LOG_WARN, NNR_LOG_DOMAIN, "NNRt", __VA_ARGS__)
#define LOGE(...) HiLogPrint(LOG_CORE, LOG_ERROR, NNR_LOG_DOMAIN, "NNRt", __VA_ARGS__)
#define LOGF(...) HiLogPrint(LOG_CORE, LOG_FATAL, NNR_LOG_DOMAIN, "NNRt", __VA_ARGS__)

#ifdef __cplusplus
}
#endif

#endif // NEURAL_NETWORK_RUNTIME_LOG_H
