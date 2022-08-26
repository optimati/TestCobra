#pragma once

#include "Analytics/AnalyticsModule.h"                           // for ANALYTICS_API
#include "Analytics/calibration/lognormal_model_with_mhjm_ir.h"  // for lognormal_mode...
#include "Core/common/pointer.h"                                 // for shared_ptr

namespace cobra
{
class datetime;
class parameter_markovian_hjm;
template <typename value_t>
class matrix;
}  // namespace cobra

namespace cobra
{
class ANALYTICS_VISIBILITY lognormal_fx_with_mhjm_ir : public lognormal_model_with_mhjm_ir
{
public:
    ANALYTICS_API lognormal_fx_with_mhjm_ir(
        const datetime&                           valution_date,
        const matrix<double>&                     correlation_matrix,
        const ptr_const<parameter_markovian_hjm>& parameter_ir_domestic,
        const ptr_const<parameter_markovian_hjm>& parameter_ir_foreign);

    ANALYTICS_API ~lognormal_fx_with_mhjm_ir() override;

private:
    ptr_const<cobra::parameter_markovian_hjm> parameter_ir_domestic_;
    ptr_const<cobra::parameter_markovian_hjm> parameter_ir_foreign_;
};
}  // namespace cobra
