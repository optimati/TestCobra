#include "Analytics/calibration/lognormal_model_with_mhjm_ir.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <utility>

#include "Analytics/model/model_markovian_hjm.h"
#include "Analytics/parameters/parameter_lognormal.h"
#include "Analytics/parameters/parameter_markovian_hjm.h"
#include "Core/common/pointer.h"
#include "Util/day_count_convention.h"
#include "Vectorization/expressions/expressions.h"
#include "Vectorization/terminals/vector.h"

namespace cobra
{
namespace details
{
//-----------------------------------------------------------------------------
// a*x^2+b*x+c=0;
double solve_quadratique_eqation(double a, double b, double c)
{
    double delta = b * b - 4 * a * c;
    return (delta >= 0.) ? (-b + sqrt(delta)) / (2. * a) : 0.;
}

//-----------------------------------------------------------------------------
double accumulate_ir_covariance(
    const matrix<double>& sigma_sv,
    const matrix<double>& sigma_mm,
    const matrix<double>& rho_ir_ir,
    const matrix<double>& G,
    size_t                T)
{
    double output = 0;
    for (size_t t = 1; t <= T; ++t)
    {
        auto sigma_t = G[T] * sigma_sv[t] - sigma_mm[t];
        output += accumulate(sigma_t * (rho_ir_ir * sigma_t));
    }

    return output;
}

//-----------------------------------------------------------------------------
double accumulate_ir_asset_covariance(
    const matrix<double>& sigma_asset,
    const matrix<double>& sigma_sv,
    const matrix<double>& sigma_mm,
    const matrix<double>& rho_asset_ir,
    const matrix<double>& G,
    size_t                T,
    double&               c)
{
    vector<double> output(rho_asset_ir.rows());
    output = 0;
    for (size_t t = 1; t < T; ++t)
    {
        auto sigma_bond = G[T] * sigma_sv[t] - sigma_mm[t];
        output += sigma_asset[t] * (rho_asset_ir * sigma_bond);
    }
    c += 2. * accumulate(output);

    auto sigma_bond = G[T] * sigma_sv[T] - sigma_mm[T];
    return 2. * accumulate(sigma_asset[T] * (rho_asset_ir * sigma_bond));
}

//-----------------------------------------------------------------------------
double accumulate_asset_covariance(
    const matrix<double>& sigma_asset, const matrix<double>& rho_asset, size_t T, double& c)
{
    vector<double> output(rho_asset.columns());
    output = 0;
    for (size_t t = 1; t < T; ++t)
    {
        output += sigma_asset[t] * (rho_asset * sigma_asset[t]);
    }
    c += accumulate(output);

    return accumulate(sigma_asset[T] * (rho_asset * sigma_asset[T]));
}
}  // namespace details

//-----------------------------------------------------------------------------
lognormal_model_with_mhjm_ir::lognormal_model_with_mhjm_ir(
    const cobra::datetime&                           valution_date,
    const ptr_const<cobra::parameter_markovian_hjm>& parameter_ir,
    matrix<double>&&                                 correlation_asset_ir,
    matrix<double>&&                                 correlation_asset_asset)
    : valution_date_(valution_date),
      parameter_ir_(parameter_ir),
      correlation_asset_ir_(std::move(correlation_asset_ir)),
      correlation_asset_asset_(std::move(correlation_asset_asset))
{
}

//-----------------------------------------------------------------------------
lognormal_model_with_mhjm_ir::~lognormal_model_with_mhjm_ir() = default;

//-----------------------------------------------------------------------------
ptr_const<parameter_lognormal> lognormal_model_with_mhjm_ir::calibrate(
    const std::vector<cobra::datetime>&    calibration_dates,
    const std::vector<double>&             market_variance,
    const ptr_const<day_count_convention>& day_convention) const
{
    COBRA_CHECK_DEBUG(
        market_variance.size() == calibration_dates.size(),
        "market volatilities and calibration dates have different sizes");

    auto ir_num_assets    = parameter_ir_->number_of_factors();
    auto num_model_assets = correlation_asset_asset_.rows();

    matrix<double> sigma(calibration_dates.size(), num_model_assets);
    sigma    = 1.;
    sigma[0] = 0.;

    model_markovian_hjm model(*parameter_ir_, valution_date_, calibration_dates, false);

    matrix<double> sigma_sv(calibration_dates.size(), ir_num_assets);
    matrix<double> sigma_mm(calibration_dates.size(), ir_num_assets);
    matrix<double> G(calibration_dates.size(), ir_num_assets);

    model.incremental_volatility_sv(calibration_dates, sigma_sv);
    model.incremental_volatility_mm(calibration_dates, sigma_mm);

    for (size_t i = 0; i < calibration_dates.size(); ++i)
    {
        auto       G_i = G[i];
        const auto t   = calibration_dates[i];
        model.integral_decay(t, G_i);
    }

    for (size_t i = 1; i < calibration_dates.size(); ++i)
    {
        auto c = details::accumulate_ir_covariance(
                     sigma_sv, sigma_mm, parameter_ir_->correlation(), G, i) -
                 market_variance[i];

        auto b = details::accumulate_ir_asset_covariance(
            sigma, sigma_sv, sigma_mm, correlation_asset_ir_, G, i, c);

        auto a = details::accumulate_asset_covariance(sigma, correlation_asset_asset_, i, c);

        sigma[i] *= details::solve_quadratique_eqation(a, b, c);
    }

    matrix<double> volatility(calibration_dates.size(), num_model_assets);
    volatility[0] = 0.;

    for (size_t i = 1; i < calibration_dates.size(); ++i)
        volatility[i] = sigma[i] / sqrt(day_convention->fraction(
                                       calibration_dates[i - 1], calibration_dates[i]));

    return util::make_ptr_const<parameter_lognormal>(
        volatility, correlation_asset_asset_, calibration_dates, day_convention);
}
}  // namespace cobra
