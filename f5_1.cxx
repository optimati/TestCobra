#include "Analytics/calibration/equity/lognormal_hjm/lognormal_equity_with_mhjm_ir.h"

#include <cstddef>  // for size_t

#include "Analytics/parameters/parameter_markovian_hjm.h"  // for parameter_markovian_hjm
#include "Vectorization/terminals/matrix.h"                // for matrix, matrix<>::ve...

namespace cobra
{
class datetime;
}

namespace cobra
{
namespace details
{
//-----------------------------------------------------------------------------
matrix<double> correlation_equity_ir(
    const matrix<double>& correlation_matrix, size_t ir_num_factors)
{
    const auto     total_number_of_factors = correlation_matrix.rows();
    matrix<double> output(total_number_of_factors - ir_num_factors, ir_num_factors);

    for (size_t f1 = 0; f1 < ir_num_factors; ++f1)
        for (size_t f2 = ir_num_factors; f2 < total_number_of_factors; ++f2)
        {
            output[f2 - ir_num_factors][f1] = -correlation_matrix[f1][f2];
        }

    return output;
}

//-----------------------------------------------------------------------------
matrix<double> correlation_equity_equity(
    const matrix<double>& correlation_matrix, size_t ir_num_factors)
{
    const auto total_number_of_factors = correlation_matrix.rows();
    const auto num_of_equity_factors   = total_number_of_factors - ir_num_factors;

    matrix<double> output(num_of_equity_factors, num_of_equity_factors);

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
lognormal_equity_with_mhjm_ir::lognormal_equity_with_mhjm_ir(
    const cobra::datetime&                           valution_date,
    const matrix<double>&                            correlation_matrix,
    const ptr_const<cobra::parameter_markovian_hjm>& parameter_ir)
    : lognormal_model_with_mhjm_ir(
          valution_date,
          parameter_ir,
          std::move(details::correlation_equity_ir(
              correlation_matrix, parameter_ir->number_of_factors())),
          std::move(details::correlation_equity_equity(
              correlation_matrix, parameter_ir->number_of_factors())))
{
}

//-----------------------------------------------------------------------------
lognormal_equity_with_mhjm_ir::~lognormal_equity_with_mhjm_ir() = default;
}  // namespace cobra
