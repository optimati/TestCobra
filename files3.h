#pragma once

#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <vector>

#include "Math/common/constants.h"
#include "TestingUtilModule.h"

namespace cobra
{
namespace aad_test
{
const double     AAD_BUMP_MIN     = std::sqrt(std::numeric_limits<double>::epsilon());
constexpr double AAD_BUMP_MAX     = 0.5;
constexpr double AAD_TOLERANCE    = 10 * std::numeric_limits<double>::epsilon();
constexpr double AAD_MAX_TOLERANC = 5.e-10;

//-----------------------------------------------------------------------------
TESTINGUTIL_API void print_aad_results(
    const std::vector<int>&    offsets,
    const std::vector<double>& aad_sensi,
    const std::vector<double>& bump_sensi,
    const std::vector<double>& tolerances);

//-----------------------------------------------------------------------------
inline double relative_difference(const double df_dx_bump, const double df_dx_aad)
{
    return std::fabs(df_dx_bump - df_dx_aad) / std::fmax(std::fabs(df_dx_aad), 1.);
}

//-----------------------------------------------------------------------------
inline void fill_aad_results(
    const double df_dx, const double bump, const double tolerance, std::vector<double>& ret)
{
    ret[2] = df_dx;
    ret[3] = bump;
    ret[4] = tolerance;
}

//-----------------------------------------------------------------------------
template <typename funtion, typename params>
double bump_and_reval(
    const funtion& function, const params& parameter, const size_t offset, const double bump)
{
    auto parameter_bumped = parameter;
    parameter_bumped[offset] += bump;

    return function(parameter_bumped);
}

//-----------------------------------------------------------------------------
template <typename funtion, typename params>
void compute_derivatives(
    double               norm,
    const funtion&       function,
    const params&        parameters,
    const size_t         offset,
    const double         bump,
    bool                 normalise,
    std::vector<double>& ret)
{
    auto f_up   = bump_and_reval(function, parameters, offset, bump);
    auto f_down = bump_and_reval(function, parameters, offset, -bump);

    auto f = norm;

    if (!cobra::is_almost_zero(norm) && normalise)
    {
        f_up /= norm;
        f_down /= norm;
        f = 1.;
    }

    ret[0] = (f - f_down) / bump;
    ret[1] = (f_up - f_down) / (2. * bump);
    ret[2] = (f_up - f) / bump;
}

//-----------------------------------------------------------------------------
template <typename funtion, typename params>
bool test_aad_one_sensitivity(
    const funtion&       function,
    const double         norm,
    const params&        parameters,
    const size_t         offset,
    double               parameter_aad,
    bool                 weak_test,
    bool                 normalise,
    const double         bump_min,
    const double         bump_max,
    const double         tolerance,
    const double         max_tolerance,
    std::vector<double>& ret)
{
    ret[0] = static_cast<double>(offset);
    if (std::fabs(norm) > std::numeric_limits<double>::epsilon() && normalise)
        parameter_aad /= norm;

    ret[1] = parameter_aad;

    auto                bump = weak_test ? 1e-15 : bump_min;
    std::vector<double> derivatives_with_bump(3);

    compute_derivatives(norm, function, parameters, offset, bump, normalise, derivatives_with_bump);

    if (cobra::is_almost_zero(derivatives_with_bump[1]) && cobra::is_almost_zero(parameter_aad))
    {
        fill_aad_results(
            derivatives_with_bump[1], bump, std::numeric_limits<double>::epsilon(), ret);
        return true;
    }

    while ((std::isnan(derivatives_with_bump[0]) || std::isnan(derivatives_with_bump[1]) ||
            std::isnan(derivatives_with_bump[2]) || parameter_aad * derivatives_with_bump[2] < 0 ||
            parameter_aad * derivatives_with_bump[0] < 0 ||
            std::fabs(derivatives_with_bump[1] / parameter_aad - 1.) > 0.05) &&
           bump < 0.1 * bump_max)
    {
        bump *= 5;
        compute_derivatives(
            norm, function, parameters, offset, bump, normalise, derivatives_with_bump);

        if (std::fabs(derivatives_with_bump[0] / derivatives_with_bump[1] - 1.) < tolerance &&
            std::fabs(derivatives_with_bump[2] / derivatives_with_bump[1] - 1.) < tolerance)
            break;

        if (relative_difference(parameter_aad, derivatives_with_bump[1]) < max_tolerance)
        {
            fill_aad_results(derivatives_with_bump[1], bump, max_tolerance, ret);
            return true;
        }
    }
    if (bump_max * 0.1 <= bump)
    {
        fill_aad_results(derivatives_with_bump[1], bump, tolerance, ret);
        return false;
    }

    if (std::fabs(derivatives_with_bump[1] / parameter_aad - 1.) > 0.05)
    {
        fill_aad_results(derivatives_with_bump[1], bump, tolerance, ret);
        return false;
    }

    auto   best_bump       = bump;
    auto   local_tolerance = tolerance;
    size_t max_iter        = 0;

    std::vector<double> derivatives_bump_tmp(3);
    std::vector<double> derivatives_2bumps_tmp(3);
    while (bump < bump_max)
    {
        auto compute_3d_derivative = false;

        compute_derivatives(
            norm, function, parameters, offset, bump, normalise, derivatives_bump_tmp);
        for (const auto r : derivatives_bump_tmp)
        {
            if (std::isnan(r) || std::isinf(r))
                return false;
        }

        if (relative_difference(derivatives_bump_tmp[1], parameter_aad) <
            relative_difference(derivatives_with_bump[1], parameter_aad))
        {
            best_bump             = bump;
            derivatives_with_bump = derivatives_bump_tmp;
            compute_3d_derivative = true;
        }

        compute_derivatives(
            norm, function, parameters, offset, 2 * bump, normalise, derivatives_2bumps_tmp);
        for (const auto r : derivatives_2bumps_tmp)
        {
            if (std::isnan(r) || std::isinf(r))
                return false;
        }

        if (relative_difference(derivatives_2bumps_tmp[1], parameter_aad) <
            relative_difference(derivatives_with_bump[1], parameter_aad))
        {
            best_bump             = 2 * bump;
            derivatives_with_bump = derivatives_2bumps_tmp;
            compute_3d_derivative = true;
        }

        auto df_dx_mid   = derivatives_bump_tmp[1];
        auto df_dx_mid_2 = derivatives_2bumps_tmp[1];
        auto df_dx       = (4. * df_dx_mid - df_dx_mid_2) / 3.;

        if (relative_difference(df_dx, parameter_aad) < max_tolerance)
        {
            fill_aad_results(df_dx, best_bump, max_tolerance, ret);
            return true;
        }

        auto third_derivatives_tolerance = (std::fabs(df_dx_mid_2 - df_dx_mid) / 6.);

        if (compute_3d_derivative && std::fabs(df_dx_mid / parameter_aad - 1.) < 0.01 &&
            third_derivatives_tolerance > std::numeric_limits<double>::epsilon())
        {
            local_tolerance = std::fmin(
                max_tolerance,
                std::fmax(third_derivatives_tolerance / (std::fabs(parameter_aad) + 1), tolerance));
        }

        for (size_t i = 0; i < 3; i++)
        {
            auto tmp_tolerance = (i != 1 ? tolerance : local_tolerance);
            if (relative_difference(derivatives_with_bump[i], parameter_aad) < tmp_tolerance)
            {
                fill_aad_results(derivatives_with_bump[i], best_bump, tmp_tolerance, ret);
                return true;
            }
        }

        if (std::fabs(df_dx_mid_2 - parameter_aad) > third_derivatives_tolerance &&
            third_derivatives_tolerance > 10 * max_tolerance * (std::fabs(parameter_aad) + 1))
        {
            bump *= (bump < 0.1 * bump_max ? 10. : 2.);
            max_iter = 0;
        }
        else
        {
            // auto use_2_bump = false;
            if (relative_difference(df_dx_mid_2, parameter_aad) <=
                relative_difference(df_dx_mid, parameter_aad))
            {
                // use_2_bump = true;
                bump *= 2.;
                df_dx_mid = df_dx_mid_2;
                bump *= df_dx_mid_2 / parameter_aad;
            }
            else
            {
                if (max_iter > 4 || third_derivatives_tolerance > std::sqrt(bump) ||
                    std::fabs(df_dx_mid - parameter_aad - third_derivatives_tolerance) >
                        max_tolerance)
                {
                    bump *= 2. * df_dx_mid_2 / parameter_aad;
                    max_iter = 0;
                }
                else
                {
                    bump *= df_dx_mid / parameter_aad;
                    max_iter++;
                }
            }
        }
    }
    for (size_t i = 0; i < 3; i++)
    {
        if (std::fabs(norm) * relative_difference(derivatives_with_bump[i], parameter_aad) <
            max_tolerance)
        {
            fill_aad_results(derivatives_with_bump[i], best_bump, max_tolerance, ret);
            return true;
        }
    }

    fill_aad_results(derivatives_with_bump[1], bump, tolerance, ret);
    return false;
}

//-----------------------------------------------------------------------------
template <
    typename funtion    = std::function<void(std::vector<double> const&, std::vector<double>&)>,
    typename params     = std::vector<double>,
    typename params_aad = std::vector<double>>
bool run_aad_test(
    const funtion&    function,
    const params&     parameters,
    const params_aad& parameters_aad,
    bool              weak_test = false,
    bool              normalise = true,
    bool              debug     = false)
{
    const auto number_of_parameters = parameters.size();

    const auto norm = function(parameters);

    std::vector<int>    offsets;
    std::vector<double> aad;
    std::vector<double> bump_and_reval;
    std::vector<double> tolerances;

    offsets.reserve(number_of_parameters);
    aad.reserve(number_of_parameters);
    bump_and_reval.reserve(number_of_parameters);
    tolerances.reserve(number_of_parameters);

    std::vector<double> outputs(5);
    bool                ret = true;
    for (size_t i = 0; i < number_of_parameters; ++i)
    {
        auto passed = test_aad_one_sensitivity(
            function,
            norm,
            parameters,
            i,
            parameters_aad[i],
            weak_test,
            normalise,
            AAD_BUMP_MIN,
            AAD_BUMP_MAX,
            AAD_TOLERANCE,
            AAD_MAX_TOLERANC,
            outputs);

        double error = std::fabs(outputs[1] / outputs[2] - 1.);
        if (error < 0.001 && weak_test)
            passed = true;

        if ((debug && passed) || !passed)
        {
            offsets.push_back(static_cast<int>(i));
            aad.push_back(outputs[1]);
            bump_and_reval.push_back(outputs[2]);
            tolerances.push_back(outputs[4]);
        }

        if (!passed)
            ret = false;
    }

    if (!ret || debug)
        cobra::aad_test::print_aad_results(offsets, aad, bump_and_reval, tolerances);

    return ret;
}
}  // namespace aad_test
}  // namespace cobra
