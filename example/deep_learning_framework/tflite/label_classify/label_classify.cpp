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

#include "label_classify.h"

#include <fcntl.h>
#include <getopt.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <unistd.h>

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/command_line_flags.h"
#include "tensorflow/lite/tools/delegates/delegate_provider.h"

#include "log.h"
#include "utils.h"

namespace tflite {
namespace label_classify {
using TfLiteDelegatePtr = tflite::Interpreter::TfLiteDelegatePtr;
using ProvidedDelegateList = tflite::tools::ProvidedDelegateList;
constexpr int BASE_NUMBER = 10;
constexpr int CONVERSION_RATE = 1000;
static struct option LONG_OPTIONS[] = {
    {"help", no_argument, nullptr, 'h'},
    {"use_nnrt", required_argument, nullptr, 'a'},
    {"count", required_argument, nullptr, 'c'},
    {"image", required_argument, nullptr, 'i'},
    {"labels", required_argument, nullptr, 'l'},
    {"tflite_model", required_argument, nullptr, 'm'},
    {"num_results", required_argument, nullptr, 'n'},
    {"input_mean", required_argument, nullptr, 'b'},
    {"input_std", required_argument, nullptr, 's'},
    {"verbose", required_argument, nullptr, 'v'},
    {"warmup_nums", required_argument, nullptr, 'w'},
    {"print_result", required_argument, nullptr, 'z'},
    {"input_shape", required_argument, nullptr, 'p'},
    {nullptr, 0, nullptr, 0},
};

class DelegateProviders {
public:
    DelegateProviders() : m_delegateListUtil(&params)
    {
        m_delegateListUtil.AddAllDelegateParams();  // Add all registered delegate params to the contained 'params_'.
    }

    ~DelegateProviders() {}

    bool InitFromCmdlineArgs(int32_t* argc, const char** argv)
    {
        std::vector<tflite::Flag> flags;
        m_delegateListUtil.AppendCmdlineFlags(&flags);

        const bool parseResult = Flags::Parse(argc, argv, flags);
        if (!parseResult) {
            std::string usage = Flags::Usage(argv[0], flags);
            LOG(ERROR) << usage;
        }
        return parseResult;
    }

    void MergeSettingsIntoParams(const Settings& settings)
    {
        if (settings.accel) {
            if (!params.HasParam("use_nnrt")) {
                LOG(WARN) << "NNRT deleate execution provider isn't linked or NNRT "
                          << "delegate isn't supported on the platform!";
            } else {
                params.Set<bool>("use_nnrt", true);
            }
        }
    }

    std::vector<ProvidedDelegateList::ProvidedDelegate> CreateAllDelegates() const
    {
        return m_delegateListUtil.CreateAllRankedDelegates();
    }

private:
    // Contain delegate-related parameters that are initialized from command-line flags.
    tflite::tools::ToolParams params;

    // A helper to create TfLite delegates.
    ProvidedDelegateList m_delegateListUtil;
};

void PrepareModel(Settings& settings, std::unique_ptr<tflite::Interpreter>& interpreter,
    DelegateProviders& delegateProviders)
{
    const std::vector<int32_t> inputs = interpreter->inputs();
    const std::vector<int32_t> outputs = interpreter->outputs();

    if (settings.verbose) {
        LOG(INFO) << "number of inputs: " << inputs.size();
        LOG(INFO) << "number of outputs: " << outputs.size();
    }

    std::map<int, std::vector<int>> neededInputShapes;
    if (settings.inputShape != "") {
        if (FilterDynamicInputs(settings, interpreter, neededInputShapes) != kTfLiteOk) {
            return;
        }
    }

    delegateProviders.MergeSettingsIntoParams(settings);
    auto delegates = delegateProviders.CreateAllDelegates();

    for (auto& delegate : delegates) {
        const auto delegateName = delegate.provider->GetName();
        if (interpreter->ModifyGraphWithDelegate(std::move(delegate.delegate)) != kTfLiteOk) {
            LOG(ERROR) << "Failed to apply " << delegateName << " delegate.";
            return;
        } else {
            LOG(INFO) << "Applied " << delegateName << " delegate.";
        }
    }

    if (settings.inputShape != "") {
        for (const auto& inputShape : neededInputShapes) {
            if (IsEqualShape(inputShape.first, inputShape.second, interpreter)) {
                LOG(WARNING) << "The input shape is same as the model shape, not resize.";
                continue;
            }
            if (interpreter->ResizeInputTensor(inputShape.first, inputShape.second) != kTfLiteOk) {
                LOG(ERROR) << "Fail to resize index " << inputShape.first << ".";
                return;
            } else {
                LOG(INFO) << "Susccess to resize index " << inputShape.first << ".";
            }
        }
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        LOG(ERROR) << "Failed to allocate tensors!";
        return;
    }

    if (settings.verbose) {
        PrintInterpreterState(interpreter.get());
    }
}

void LogInterpreterParams(Settings& settings, std::unique_ptr<tflite::Interpreter>& interpreter)
{
    if (!interpreter) {
        LOG(ERROR) << "Failed to construct interpreter";
        return;
    }

    if (settings.verbose) {
        LOG(INFO) << "tensors size: " << interpreter->tensors_size();
        LOG(INFO) << "nodes size: " << interpreter->nodes_size();
        LOG(INFO) << "inputs: " << interpreter->inputs().size();
        LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0);

        size_t tSize = interpreter->tensors_size();
        for (size_t i = 0; i < tSize; ++i) {
            if (interpreter->tensor(i)->name) {
                LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", " << interpreter->tensor(i)->bytes <<
                    ", " << interpreter->tensor(i)->type << ", " << interpreter->tensor(i)->params.scale << ", " <<
                    interpreter->tensor(i)->params.zero_point;
            }
        }
    }
}

void InferenceModel(Settings& settings, DelegateProviders& delegateProviders)
{
    if (!settings.modelName.c_str()) {
        LOG(ERROR) << "no model file name";
        return;
    }
    std::unique_ptr<tflite::FlatBufferModel> model;
    std::unique_ptr<tflite::Interpreter> interpreter;
    model = tflite::FlatBufferModel::BuildFromFile(settings.modelName.c_str());
    if (!model) {
        LOG(ERROR) << "Failed to mmap model " << settings.modelName;
        return;
    }

    settings.model = model.get();
    model->error_reporter();
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        LOG(ERROR) << "Failed to construct interpreter, please check the model.";
        return;
    }

    LogInterpreterParams(settings, interpreter);

    // set settings input type
    PrepareModel(settings, interpreter, delegateProviders);
    std::vector<int> imageSize { 224, 224, 3};
    ImportData(settings, imageSize, interpreter);

    if (settings.loopCount > 0 && settings.numberOfWarmupRuns > 0) {
        LOG(INFO) << "Warm-up for " << settings.numberOfWarmupRuns << " times";
        for (int32_t i = 0; i < settings.numberOfWarmupRuns; ++i) {
            if (interpreter->Invoke() != kTfLiteOk) {
                LOG(ERROR) << "Failed to invoke tflite!";
                return;
            }
        }
    }

    struct timeval startTime, stopTime;
    LOG(INFO) << "Invoke for " << settings.loopCount << " times";
    gettimeofday(&startTime, nullptr);
    for (int32_t i = 0; i < settings.loopCount; ++i) {
        if (interpreter->Invoke() != kTfLiteOk) {
            LOG(ERROR) << "Failed to invoke tflite!";
            return;
        }
    }

    gettimeofday(&stopTime, nullptr);
    LOG(INFO) << "invoked, average time: " <<
        (GetUs(stopTime) - GetUs(startTime)) / (settings.loopCount * CONVERSION_RATE) << " ms";
    AnalysisResults(settings, interpreter);
}

void DisplayUsage()
{
    LOG(INFO) << "label_classify -m xxx.tflite -i xxx.bmp -l xxx.txt -c 1 -a 1\n"
              << "\t--help,         -h: show the usage of the demo\n"
              << "\t--use_nnrt,     -a: [0|1], 1 refers to use NNRT\n"
              << "\t--input_mean,   -b: input mean\n"
              << "\t--count,        -c: loop interpreter->Invoke() for certain times\n"
              << "\t--image,        -i: image_name.bmp\n"
              << "\t--labels,       -l: labels for the model\n"
              << "\t--tflite_model, -m: modelName.tflite\n"
              << "\t--num_results,  -n: number of results to show\n"
              << "\t--input_std,    -s: input standard deviation\n"
              << "\t--verbose,      -v: [0|1] print more information\n"
              << "\t--warmup_nums,  -w: number of warmup runs\n"
              << "\t--print_result, -z: flag to print results\n"
              << "\t--input_shape,  -p: Indicates the specified dynamic input node and the corresponding shape.\n";
}

int32_t InitSettings(int32_t argc, char** argv, Settings& settings)
{
    // getopt_long stores the option index here.
    int32_t optionIndex = 0;
    while ((optionIndex = getopt_long(argc, argv, "a:b:c:h:i:l:m:n:p:s:v:w:z:", LONG_OPTIONS, nullptr)) != -1) {
        switch (optionIndex) {
            case 'a':
                settings.accel = strtol(optarg, nullptr, BASE_NUMBER);
                break;
            case 'b':
                settings.inputMean = strtod(optarg, nullptr);
                break;
            case 'c':
                settings.loopCount = strtol(optarg, nullptr, BASE_NUMBER);
                break;
            case 'i':
                settings.inputBmpName = optarg;
                break;
            case 'l':
                settings.labelsFileName = optarg;
                break;
            case 'm':
                settings.modelName = optarg;
                break;
            case 'n':
                settings.numberOfResults = strtol(optarg, nullptr, BASE_NUMBER);
                break;
            case 'p':
                settings.inputShape = optarg;
                break;
            case 's':
                settings.inputStd = strtod(optarg, nullptr);
                break;
            case 'v':
                settings.verbose = strtol(optarg, nullptr, BASE_NUMBER);
                break;
            case 'w':
                settings.numberOfWarmupRuns = strtol(optarg, nullptr, BASE_NUMBER);
                break;
            case 'z':
                settings.printResult = strtol(optarg, nullptr, BASE_NUMBER);
                break;
            case 'h':
            case '?':
                // getopt_long already printed an error message.
                DisplayUsage();
                return -1;
            default:
                return -1;
        }
    }

    return 0;
}

int32_t Main(int32_t argc, char** argv)
{
    if (argc <= 1) {
        DisplayUsage();
        return EXIT_FAILURE;
    }

    DelegateProviders delegateProviders;
    bool parseResult = delegateProviders.InitFromCmdlineArgs(&argc, const_cast<const char**>(argv));
    if (!parseResult) {
        return EXIT_FAILURE;
    }

    Settings settings;
    if (InitSettings(argc, argv, settings) == -1) {
        return EXIT_FAILURE;
    };

    InferenceModel(settings, delegateProviders);
    return 0;
}
} // namespace label_classify
} // namespace tflite

int32_t main(int32_t argc, char** argv)
{
    return tflite::label_classify::Main(argc, argv);
}
