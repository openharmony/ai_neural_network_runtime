#ifndef OHOS_HDI_NNRT_V2_0_UTILS_H
#define OHOS_HDI_NNRT_V2_0_UTILS_H

#include <hdf_base.h>
#include "v2_0/nnrt_types.h"

namespace OHOS {
namespace HDI {
namespace Nnrt {
namespace V2_0 {
inline int32_t GetHDFReturnCode(NNRT_ReturnCode returnCode)
{
    if (returnCode == NNRT_ReturnCode::NNRT_INVALID_PARAMETER ||
        (returnCode <= NNRT_ReturnCode::NNRT_INVALID_TENSOR &&
         returnCode >= NNRT_ReturnCode::NNRT_INVALID_MODEL_CACHE)) {
        return HDF_ERR_INVALID_PARAM;
    } else if (returnCode == NNRT_ReturnCode::NNRT_OUT_OF_MEMORY) {
        return HDF_ERR_MALLOC_FAIL;
    } 

    return HDF_FAILURE;
}
} // namespace V2_0
} // namespace Nnrt
} // namespace HDI
} // namespace OHOS
#endif // OHOS_HDI_NNRT_UTILS_H