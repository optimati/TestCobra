#pragma once

#include <vector>  // for vector

#include "Analytics/AnalyticsModule.h"
#include "Core/common/pointer.h"             // for shared_ptr
#include "Util/datetime.h"                   // for datetime
#include "Vectorization/terminals/matrix.h"  // for matrix

namespace cobra
{
class day_count_convention;
class parameter_lognormal;
class parameter_markovian_hjm;
}  // namespace cobra

namespace cobra
{
class ANALYTICS_VISIBILITY lognormal_model_with_mhjm_ir
{
public:
    ANALYTICS_API ptr_const<cobra::parameter_lognormal> calibrate(
        const std::vector<cobra::datetime>&    calibration_dates,
        const std::vector<double>&             market_variance,
        const ptr_const<day_count_convention>& day_convention) const;

    ANALYTICS_API virtual ~lognormal_model_with_mhjm_ir();

protected:
    lognormal_model_with_mhjm_ir(
        const cobra::datetime&                           valution_date,
        const ptr_const<cobra::parameter_markovian_hjm>& parameter_ir,
        matrix<double>&&                                 correlation_asset_ir,
        matrix<double>&&                                 correlation_asset_asset);

    const cobra::datetime                     valution_date_;
    ptr_const<cobra::parameter_markovian_hjm> parameter_ir_;
    matrix<double>                            correlation_asset_ir_;
    matrix<double>                            correlation_asset_asset_;
};
}  // namespace cobra
