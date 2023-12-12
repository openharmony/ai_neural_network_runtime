#include "common/utils.h"

namespace OHOS {
namespace NeuralNetworkRuntime {
std::string GenUniqueName(
    const std::string& deviceName, const std::string& vendorName, const std::string& version)
{
    return deviceName + "_" + vendorName + "_" + version;
}

} // namespace NeuralNetworkRuntime
} // namespace OHOS