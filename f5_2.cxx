#include "Analytics/calibration/fx/lognormal_hjm/lognormal_fx_with_mhjm_ir.h"

#include <cstddef>  // for size_t
#include <utility>  // for move
#include <vector>   // for vector

#include "Analytics/parameters/parameter_markovian_hjm.h"  // for parameter_markovian_hjm
#include "Core/common/pointer.h"                           // for make_ptr_const
#include "Util/datetime.h"                                 // for datetime
#include "Util/datetime_helper.h"
#include "Vectorization/terminals/matrix.h"  // for matrix, matrix<>::ve...

namespace cobra
{
namespace details
{
//-----------------------------------------------------------------------------
ptr_const<parameter_markovian_hjm> merge_parameters(
    const matrix<double>&                 correlation_matrix,
    const cobra::parameter_markovian_hjm& parameter_ir_domestic,
    const cobra::parameter_markovian_hjm& parameter_ir_foreign)
{
    size_t ir_dom_num_factors = parameter_ir_domestic.number_of_factors();
    size_t ir_for_num_factors = parameter_ir_foreign.number_of_factors();

    size_t ir_num_factors = ir_dom_num_factors + ir_for_num_factors;

    std::vector<datetime> decays_dates;

    datetime_helper::merges_dates(
        parameter_ir_domestic.decays_dates(), parameter_ir_foreign.decays_dates(), decays_dates);

    matrix<double> decays(decays_dates.size(), ir_num_factors);
    {
        std::vector<size_t> decays_dates_indexs;
        datetime_helper::interpolate_from_subset_dates(
            decays_dates, parameter_ir_domestic.decays_dates(), decays_dates_indexs);

        for (size_t i = 0; i < decays_dates.size(); ++i)
        {
            for (size_t f = 0; f < ir_dom_num_factors; ++f)
                decays[i][f] = parameter_ir_domestic.decays().at(decays_dates_indexs[i], f);
        }
    }
    {
        std::vector<size_t> decays_dates_indexs;
        datetime_helper::interpolate_from_subset_dates(
            decays_dates, parameter_ir_foreign.decays_dates(), decays_dates_indexs);

        for (size_t i = 0; i < decays_dates.size(); ++i)
        {
            for (size_t f = 0; f < ir_for_num_factors; ++f)
                decays[i][f + ir_dom_num_factors] =
                    parameter_ir_foreign.decays().at(decays_dates_indexs[i], f);
        }
    }

    std::vector<datetime> volatilities_dates;

    datetime_helper::merges_dates(
        parameter_ir_domestic.volatilities_dates(),
        parameter_ir_foreign.volatilities_dates(),
        volatilities_dates);

    matrix<double> volatilities(volatilities_dates.size(), ir_num_factors);
    {
        std::vector<size_t> vol_indexs;
        datetime_helper::interpolate_from_subset_dates(
            volatilities_dates, parameter_ir_domestic.volatilities_dates(), vol_indexs);

        for (size_t i = 0; i < volatilities_dates.size(); ++i)
        {
            for (size_t f = 0; f < ir_dom_num_factors; ++f)
                volatilities[i][f] = parameter_ir_domestic.volatilities().at(vol_indexs[i], f);
        }
    }
    {
        std::vector<size_t> vol_indexs;
        datetime_helper::interpolate_from_subset_dates(
            volatilities_dates, parameter_ir_foreign.volatilities_dates(), vol_indexs);

        for (size_t i = 0; i < volatilities_dates.size(); ++i)
        {
            for (size_t f = 0; f < ir_for_num_factors; ++f)
                volatilities[i][f + ir_dom_num_factors] =
                    parameter_ir_foreign.volatilities().at(vol_indexs[i], f);
        }
    }

    matrix<double> correlation_ir_ir(ir_num_factors, ir_num_factors);

    for (size_t f1 = 0; f1 < ir_num_factors; ++f1)
        for (size_t f2 = 0; f2 <= f1; ++f2)
        {
            correlation_ir_ir[f1][f2] = correlation_ir_ir[f2][f1] = correlation_matrix[f1][f2];
        }
    for (size_t f1 = 0; f1 < ir_dom_num_factors; ++f1)
        for (size_t f2 = ir_dom_num_factors; f2 < ir_num_factors; ++f2)
        {
            correlation_ir_ir[f1][f2] = correlation_ir_ir[f2][f1] = -correlation_matrix[f1][f2];
        }

    return util::make_ptr_const<parameter_markovian_hjm>(
        decays,
        volatilities,
        correlation_ir_ir,
        decays_dates,
        volatilities_dates,
        parameter_ir_domestic.day_convention(),
        parameter_ir_domestic.model_type());
}

//-----------------------------------------------------------------------------
matrix<double> correlation_fx_ir(
    const matrix<double>& correlation_matrix, size_t ir_dom_num_factors, size_t ir_num_factors)
{
    const auto     total_number_of_factors = correlation_matrix.rows();
    matrix<double> output(total_number_of_factors - ir_num_factors, ir_num_factors);

    for (size_t f1 = 0; f1 < ir_num_factors; ++f1)
        for (size_t f2 = ir_num_factors; f2 < total_number_of_factors; ++f2)
        {
            output[f2 - ir_num_factors][f1] =
                f1 < ir_dom_num_factors ? -correlation_matrix[f1][f2] : correlation_matrix[f1][f2];
        }

    return output;
}

//-----------------------------------------------------------------------------
matrix<double> correlation_fx_fx(const matrix<double>& correlation_matrix, size_t ir_num_factors)
{
    const auto total_number_of_factors = correlation_matrix.rows();
    const auto num_of_fx_factors       = total_number_of_factors - ir_num_factors;

    matrix<double> output(num_of_fx_factors, num_of_fx_factors);

    for (size_t f1 = ir_num_factors; f1 < total_number_of_factors; ++f1)
        for (size_t f2 = ir_num_factors; f2 < total_number_of_factors; ++f2)
        {
            auto x                                           = correlation_matrix[f1][f2];
            output[f2 - ir_num_factors][f1 - ir_num_factors] = x;
        }

    return output;
}
}  // namespace details

//-----------------------------------------------------------------------------
lognormal_fx_with_mhjm_ir::lognormal_fx_with_mhjm_ir(
    const cobra::datetime&                           valution_date,
    const matrix<double>&                            correlation_matrix,
    const ptr_const<cobra::parameter_markovian_hjm>& parameter_ir_domestic,
    const ptr_const<cobra::parameter_markovian_hjm>& parameter_ir_foreign)
    : lognormal_model_with_mhjm_ir(
          valution_date,
          details::merge_parameters(
              correlation_matrix, *parameter_ir_domestic, *parameter_ir_foreign),
          std::move(details::correlation_fx_ir(
              correlation_matrix,
              parameter_ir_domestic->number_of_factors(),
              parameter_ir_domestic->number_of_factors() +
                  parameter_ir_foreign->number_of_factors())),
          std::move(details::correlation_fx_fx(
              correlation_matrix,
              parameter_ir_domestic->number_of_factors() +
                  parameter_ir_foreign->number_of_factors()))),
      parameter_ir_domestic_(parameter_ir_domestic),
      parameter_ir_foreign_(parameter_ir_foreign)
{
}

//-----------------------------------------------------------------------------
lognormal_fx_with_mhjm_ir::~lognormal_fx_with_mhjm_ir() = default;
}  // namespace cobra
