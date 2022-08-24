#pragma once

#include <cstddef>  // for size_t
#include <vector>   // for vector

#include "Analytics/AnalyticsModule.h"
#include "Core/common/aad_state_parameters_manager.h"
#include "Core/common/wrapping_hints.h"
#include "Serialization/common/serialization_macros.h"
#include "Util/datetime.h"
#include "Util/day_count_convention.h"
#include "Vectorization/terminals/matrix.h"

namespace cobra
{
template <typename value_t>
class vector;
}  // namespace cobra

namespace cobra
{
enum class parameter_markovian_hjm_type : int
{
    HULL_WHITE        = 0,
    PICEWISE_CONSTANT = 1
};

class ANALYTICS_VISIBILITY parameter_markovian_hjm : public aad_state_parameters_manager
{
public:
    ANALYTICS_API parameter_markovian_hjm(
        const matrix<double>&                  decays,
        const matrix<double>&                  volatilities,
        const matrix<double>&                  correlation,
        const std::vector<datetime>&           decays_dates,
        const std::vector<datetime>&           volatilities_dates,
        const ptr_const<day_count_convention>& day_convention,
        cobra::parameter_markovian_hjm_type    model_type);

    ANALYTICS_API ~parameter_markovian_hjm() override;

    ANALYTICS_API const std::vector<cobra::datetime>& decays_dates() const;

    ANALYTICS_API const std::vector<cobra::datetime>& volatilities_dates() const;

    ANALYTICS_API size_t number_of_factors() const;

    ANALYTICS_API const cobra::matrix<double>& decays() const;

    ANALYTICS_API const cobra::matrix<double>& volatilities() const;

    ANALYTICS_API const cobra::matrix<double>& correlation() const;

    ANALYTICS_API const ptr_const<day_count_convention>& day_convention() const;

    ANALYTICS_API cobra::parameter_markovian_hjm_type model_type() const;

#ifndef __COBRA_WRAP__

    ANALYTICS_API void decays_aad(
        const vector<double>& value_aad, size_t idx, double* state_parameters_aad) const;

    ANALYTICS_API void volatility_aad(
        const vector<double>& value_aad, size_t idx, double* state_parameters_aad) const;

    ANALYTICS_API void correlation_aad(
        const matrix<double>& value_aad, double* state_parameters_aad) const;

    ANALYTICS_API void finalize_aad(double* /*parameters*/) const override;

    ANALYTICS_API size_t get_state_parameters(double* parameters) const override;

    ANALYTICS_API size_t set_state_parameters(const double* parameters) override;

#endif  // !__COBRA_WRAP__

private:
    friend class calibration_ir_hjm;

    // SERIALIZATION :
    ANALYTICS_API parameter_markovian_hjm();

    COBRA_SERIALIZATION_EXPORT(
        ANALYTICS_API,
        parameter_markovian_hjm,
        decays,
        volatilities,
        correlation,
        decays_dates,
        volatilities_dates,
        day_convention,
        model_type);

    ANALYTICS_API void initialize();

    void validate();

    // members
    matrix<double>                      decays_;
    matrix<double>                      volatilities_;
    matrix<double>                      correlation_;
    std::vector<datetime>               decays_dates_;
    std::vector<datetime>               volatilities_dates_;
    ptr_const<day_count_convention>     day_convention_;
    cobra::parameter_markovian_hjm_type model_type_;

    AAD_STATE_PARAMETERS(decays_);
    AAD_STATE_PARAMETERS(volatilities_);
    AAD_STATE_PARAMETERS(correlation_);
};
}  // namespace cobra
