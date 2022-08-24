#include "Analytics/model/model_markovian_hjm.h"

#include <cmath>
#include <limits>
#include <vector>

#include "Analytics/change_of_measure/change_of_measure.h"
#include "Analytics/parameters/parameter_markovian_hjm.h"
#include "Analytics/pde/common/pde_model_coefficients.h"
#include "Core/util/exception.h"
#include "Math/common/constants.h"
#include "Math/matrix_operation/cholesky_decomposition.h"
#include "Util/datetime_helper.h"
#include "Util/day_count_convention.h"
#include "Vectorization/expressions/expressions.h"
#include "Vectorization/terminals/vector.h"

//#define RISK_NEUTRAL

namespace cobra
{
namespace details
{
//-----------------------------------------------------------------------------
inline double intermediate_money_market_variance(
    const double dcf, const double mrs, const double alpha, const double beta)
{
    if (cobra::is_almost_zero(mrs))
    {
        const auto f_t = beta * dcf + alpha;
        return dcf * (f_t * f_t + alpha * (f_t + alpha)) / 3.;
    }

    const double x = dcf * mrs;
    return 0.5 * (alpha * ((exp(x) + 1.) * alpha - 4. * beta) * expm1(x) + 2. * beta * beta * x) /
           mrs;
}

//-----------------------------------------------------------------------------
inline void intermediate_money_market_variance_aad(
    const double value_aad,
    const double value,
    const double dcf,
    const double mrs,
    const double alpha,
    const double beta,
    double&      mrs_aad,
    double&      alpha_aad,
    double&      beta_aad)
{
    if (cobra::is_almost_zero(mrs))
    {
        const auto f_t = beta * dcf + alpha;
        // AAD: value = dcf * (f_t * f_t + alpha * (f_t + alpha)) / 3.f;
        const auto tmp_aad = value_aad * dcf / 3.;
        const auto f_t_aad = tmp_aad * (2. * f_t + alpha);
        alpha_aad += tmp_aad * (2. * alpha + f_t);

        // AAD: f_t = beta * dcf + alpha;
        beta_aad += f_t_aad * dcf;
        alpha_aad += f_t_aad;
        return;
    }
    double x = dcf * mrs;
    // AAD: 0.5 / mrs *(alpha * ((exp(x) + 1.) * alpha - 4. * beta) * expm1(x) + 2. * beta * beta *
    // x);
    mrs_aad -= value_aad * value / mrs;

    double tmp_aad = value_aad / mrs;
    double x_aad   = tmp_aad * ((exp(2. * x) * alpha - 2. * beta * exp(x)) * alpha + beta * beta);
    beta_aad += 2. * tmp_aad * (beta * x - expm1(x) * alpha);
    alpha_aad += tmp_aad * (((exp(x) + 1.) * alpha - 2. * beta) * expm1(x));

    // AAD: x = dcf * mrs
    mrs_aad += dcf * x_aad;
}

//-----------------------------------------------------------------------------
template <typename T>
auto expm1_x(const T& x)
{
    return if_else(fabs(x) < std::numeric_limits<double>::epsilon(), 1. + 0.5 * x, expm1(x) / x);
}

//-----------------------------------------------------------------------------
template <typename T>
auto expm1_x_aad(const T& x)
{
    return if_else(
        fabs(x) < std::numeric_limits<double>::epsilon(), 0.5, (x * exp(x) - expm1(x)) / (x * x));
}

//-----------------------------------------------------------------------------
double state_variables_covariance_helper(
    size_t f1, size_t f2, const vector<double>& mrs, double dcf)
{
    double x = (mrs[f1] + mrs[f2]) * dcf;
    return dcf * (cobra::is_almost_zero(x) ? 1. + 0.5 * x : expm1(x) / x);
}

//-----------------------------------------------------------------------------
void state_variables_covariance_helper_aad(
    double                value_aad,
    size_t                f1,
    size_t                f2,
    const vector<double>& mrs,
    double                dcf,
    vector<double>&       mrs_aad)
{
    double x = dcf * (mrs[f1] + mrs[f2]);

    double mrs_sum_aad = value_aad * dcf * dcf *
                         (cobra::is_almost_zero(x) ? 0.5 : (x * exp(x) - expm1(x)) / (x * x));
    mrs_aad[f1] += mrs_sum_aad;
    mrs_aad[f2] += mrs_sum_aad;
}
}  // namespace details

//-----------------------------------------------------------------------------
model_markovian_hjm::model_markovian_hjm(
    const parameter_markovian_hjm& parameter,
    const datetime&                valuation_date,
    const std::vector<datetime>&   dates,
    bool                           calibration_flag,
    const std::vector<datetime>&   extra_dates)
    : parameter_(parameter),
      valution_date_(valuation_date),
      dates_(dates),
      calibration_flag_(calibration_flag),
      extra_dates_(extra_dates)
{
    initialize();
}

//-----------------------------------------------------------------------------
model_markovian_hjm::~model_markovian_hjm() = default;

//-----------------------------------------------------------------------------
void model_markovian_hjm::conditional_price(
    const matrix<double>& states,
    const datetime&       from,
    const datetime&       to,
    vector<double>&       output) const
{
    auto offset = date_offset(from);

    datetime_helper::date_offset(from, dates_);

    auto num_factors = parameter_.number_of_factors();
    auto G_from      = integral_decay_[offset];

    vector<double> G_to(num_factors);
    integral_decay(to, G_to);

    vector<double> G_diff = G_to - G_from;
    vector<double> G_add  = G_to + G_from;

    const auto var_matrix =
        covariance_matrix_.get_matrix(static_cast<int>(datetime_helper::date_offset(from, dates_)));

    const auto stdev = accumulate(G_diff * (var_matrix * G_add));

    output = -0.5 * stdev;

    for (size_t f = 0; f < num_factors; ++f)
        output -= G_diff[f] * states[f];
};

//-----------------------------------------------------------------------------
void model_markovian_hjm::step(
    const change_of_measure& adjuster,
    size_t                   time_idx,
    const matrix<double>&    rng,
    matrix<double>&          current,
    matrix<double>&          next) const
{
    current.deepcopy(next);

    auto num_factors = parameter_.number_of_factors();

    vector<double> sigma_sv = incremental_volatility_sv_[time_idx];
    vector<double> sigma_mm = incremental_volatility_mm_[time_idx];

    const auto& covar_current = covariance_matrix_.get_matrix(static_cast<int>(time_idx - 1));

    vector<double> G_current = integral_decay_[dates_indexs_[time_idx - 1]];
    vector<double> G_next    = integral_decay_[dates_indexs_[time_idx]];

    vector<double> G_diff = G_next - G_current;
    vector<double> G_add  = G_next + G_current;

    auto vol_decay = G_next * sigma_sv;

    double drift_mm = accumulate(
        (sigma_mm * (parameter_.correlation() * sigma_mm)) +
        (vol_decay * (parameter_.correlation() * vol_decay)) + (G_add * (covar_current * G_diff)));

    vector<double> drift_sv = sigma_sv * (parameter_.correlation() * sigma_mm);

    auto log_arrow_debreu_price = next[arrow_debreu_price_index()];

    log_arrow_debreu_price += 0.5 * drift_mm;

    for (size_t f = 0; f < num_factors; ++f)
    {
        const auto z = adjuster.shift_random(time_idx, f);

        next[f] -= drift_sv[f] + sigma_sv[f] * (rng[f] - z);

        log_arrow_debreu_price +=
            G_next[f] * next[f] - G_current[f] * current[f] + sigma_mm[f] * (rng[f] - z);
    }

#ifdef DEBUG_MODE
    const auto& covar_next = covariance_matrix_.get_matrix(static_cast<int>(time_idx));

    auto m   = accumulate(next[0]) / next[0].size();
    auto var = accumulate((next[0] - m) * (next[0] - m)) / next[0].size();

    COBRA_INFO_LOG("Analytics/model " << covar_current[0][0] << " simulated " << var);
#endif
}

//-----------------------------------------------------------------------------
size_t model_markovian_hjm::number_of_factors() const
{
    return parameter_.number_of_factors();
}

//-----------------------------------------------------------------------------
size_t model_markovian_hjm::number_of_states() const
{
    return parameter_.number_of_factors() + 1ULL;
}

//-----------------------------------------------------------------------------
size_t model_markovian_hjm::arrow_debreu_price_index() const
{
    return parameter_.number_of_factors();
}

//-----------------------------------------------------------------------------
// accessor methods
//-----------------------------------------------------------------------------
const matrix<double>& model_markovian_hjm::decay() const
{
    return decay_;
}

//-----------------------------------------------------------------------------
const matrix<double>& model_markovian_hjm::integral_decay() const
{
    return integral_decay_;
};

//-----------------------------------------------------------------------------
const matrix<double>& model_markovian_hjm::incremental_volatility_sv() const
{
    return incremental_volatility_sv_;
};

//-----------------------------------------------------------------------------
const matrix<double>& model_markovian_hjm::incremental_volatility_mm() const
{
    return incremental_volatility_mm_;
};

//-----------------------------------------------------------------------------
// model functions
//-----------------------------------------------------------------------------
void model_markovian_hjm::decay(const datetime& t, size_t offset, vector<double>& output) const
{
    auto from = model_dates_[offset];
    output.deepcopy(decay_[offset]);

    if (t.as_float() > from.as_float())
    {
        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto  dcf = parameter_.day_convention()->fraction(from, t);
            const auto& mrs = parameter_.decays()[decays_dates_indexs_[offset + 1]];

            output *= exp(-dcf * mrs);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            output.deepcopy(parameter_.decays()[decays_dates_indexs_[offset + 1]]);
        }
        break;
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::decay_aad(
    const vector<double>& ret_aad,
    const vector<double>& output,
    const datetime&       t,
    size_t                offset,
    double*               state_parameters_aad) const
{
    switch (parameter_.model_type())
    {
    case parameter_markovian_hjm_type::HULL_WHITE:
        log_decay_aad(ret_aad * output, t, offset, state_parameters_aad);
        return;

    case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        log_decay_aad(ret_aad, t, offset, state_parameters_aad);
        return;
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::log_decay_aad(
    const vector<double>& ret_aad,
    const datetime&       t,
    size_t                offset,
    double*               state_parameters_aad) const
{
    if (std::all_of(
            ret_aad.begin(), ret_aad.end(), [](double x) { return cobra::is_almost_zero(x); }))
        return;

    auto from = model_dates_[offset];
    if (t.as_float() > from.as_float())
    {
        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto     dcf = parameter_.day_convention()->fraction(from, t);
            vector<double> mrs_aad(parameter_.number_of_factors());
            // AAD: output -= dcf * mrs;
            mrs_aad = -dcf * ret_aad;
            parameter_.decays_aad(mrs_aad, decays_dates_indexs_[offset + 1], state_parameters_aad);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            // const auto& tmp = parameter_.decays()[decays_dates_indexs_[offset + 1]];
            parameter_.decays_aad(ret_aad, decays_dates_indexs_[offset + 1], state_parameters_aad);
        }
            return;
        }
    }

    auto num_factors = parameter_.number_of_factors();
    for (size_t i = 0; i < num_factors; ++i)
        update_state_parameters(
            ret_aad[i], AAD_OFFSET(decay_), offset * num_factors + i, state_parameters_aad);
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::integral_decay(const datetime& t, vector<double>& output) const
{
    auto offset = date_offset(t);
    if (model_dates_[offset] == t)
        output.deepcopy(integral_decay_[offset]);
    else
        integral_decay(t, date_offset(t), output);
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::integral_decay_aad(
    const vector<double>& ret_aad, const datetime& t, double* state_parameters_aad) const
{
    if (std::all_of(
            ret_aad.begin(), ret_aad.end(), [](double x) { return cobra::is_almost_zero(x); }))
        return;

    auto offset = date_offset(t);
    if (model_dates_[offset] == t)
    {
        auto num_factors = parameter_.number_of_factors();
        for (size_t i = 0; i < num_factors; ++i)
            update_state_parameters(
                ret_aad[i],
                AAD_OFFSET(integral_decay_),
                offset * num_factors + i,
                state_parameters_aad);
    }
    else
        integral_decay_aad(ret_aad, t, offset, state_parameters_aad);
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::integral_decay(
    const datetime& t, size_t offset, vector<double>& output) const
{
    auto from = model_dates_[offset];
    output.deepcopy(integral_decay_[offset]);

    if (t.as_float() > from.as_float())
    {
        const auto  dcf       = parameter_.day_convention()->fraction(from, t);
        const auto& decay_tmp = decay_[offset];

        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto& mrs = parameter_.decays()[decays_dates_indexs_[offset + 1]];

            auto x = -dcf * mrs;
            output += decay_tmp * dcf * details::expm1_x(x);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
            output += decay_tmp * dcf;
            break;
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::integral_decay_aad(
    const vector<double>& ret_aad,
    const datetime&       t,
    size_t                offset,
    double*               state_parameters_aad) const
{
    if (std::all_of(
            ret_aad.begin(), ret_aad.end(), [](double x) { return cobra::is_almost_zero(x); }))
        return;

    auto from        = model_dates_[offset];
    auto num_factors = parameter_.number_of_factors();
    if (t.as_float() > from.as_float())
    {
        const auto  dcf       = parameter_.day_convention()->fraction(from, t);
        const auto& decay_tmp = decay_[offset];

        vector<double> tmp_aad(parameter_.number_of_factors());

        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto& mrs = parameter_.decays()[decays_dates_indexs_[offset + 1]];
            auto        x   = -dcf * mrs;

            // AAD: output += decay_tmp * dcf *details::expm1_x(x);
            tmp_aad = -ret_aad * (dcf * dcf) * decay_tmp * details::expm1_x_aad(x);
            parameter_.decays_aad(tmp_aad, decays_dates_indexs_[offset + 1], state_parameters_aad);

            tmp_aad = ret_aad * dcf * decay_tmp * details::expm1_x(x);

            for (size_t i = 0; i < num_factors; ++i)
                update_state_parameters(
                    tmp_aad[i], AAD_OFFSET(decay_), offset * num_factors + i, state_parameters_aad);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            // AAD: output += decay_tmp * dcf;
            tmp_aad = ret_aad * dcf;
            for (size_t i = 0; i < num_factors; ++i)
                update_state_parameters(
                    tmp_aad[i], AAD_OFFSET(decay_), offset * num_factors + i, state_parameters_aad);
        }
        break;
        }
    }

    // AAD: output.deepcopy(integral_decay_[offset]);
    for (size_t i = 0; i < num_factors; ++i)
        update_state_parameters(
            ret_aad[i],
            AAD_OFFSET(integral_decay_),
            offset * num_factors + i,
            state_parameters_aad);
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::states_variable_variance(size_t offset, vector<double>& output) const
{
    auto from = model_dates_[offset];
    auto to   = model_dates_[offset + 1];

    if (to.as_float() > from.as_float())
    {
        const auto  dcf = parameter_.day_convention()->fraction(from, to);
        const auto& vol = parameter_.volatilities()[volatilities_dates_indexs_[offset + 1]];

        vector<double> vol_decay(parameter_.number_of_factors());

        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto& mrs       = parameter_.decays()[decays_dates_indexs_[offset + 1]];
            const auto& decay_tmp = decay_[offset];
            vol_decay             = vol / decay_tmp;

            auto x = 2. * dcf * mrs;
            output += vol_decay * vol_decay * dcf * details::expm1_x(x);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            output += vol * vol * dcf;
        }
        break;
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::states_variable_variance_aad(
    const vector<double>& ret_aad, size_t offset, double* state_parameters_aad) const
{
    if (std::all_of(
            ret_aad.begin(), ret_aad.end(), [](double x) { return cobra::is_almost_zero(x); }))
        return;

    auto from = model_dates_[offset];
    auto to   = model_dates_[offset + 1];

    if (to.as_float() > from.as_float())
    {
        const auto  vol_offset = volatilities_dates_indexs_[offset + 1];
        const auto  dcf        = parameter_.day_convention()->fraction(from, to);
        const auto& vol        = parameter_.volatilities()[vol_offset];

        vector<double> tmp_aad(parameter_.number_of_factors());

        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto& decay_tmp = decay_[offset];

            const auto&    mrs       = parameter_.decays()[decays_dates_indexs_[offset + 1]];
            vector<double> vol_decay = dcf * vol / decay_tmp;

            auto x = 2. * dcf * mrs;
            // AAD: output += vol_decay * vol_decay * details::expm1_x(x);
            tmp_aad = 2 * ret_aad * vol_decay * vol_decay * details::expm1_x_aad(x);

            parameter_.decays_aad(tmp_aad, decays_dates_indexs_[offset + 1], state_parameters_aad);

            tmp_aad = 2. * ret_aad * vol_decay * details::expm1_x(x);

            // AAD: vol_decay=vol/decay_tmp
            tmp_aad = tmp_aad / decay_tmp;
            parameter_.volatility_aad(tmp_aad, vol_offset, state_parameters_aad);

            tmp_aad *= -vol;

            auto num_factors = parameter_.number_of_factors();
            for (size_t i = 0; i < num_factors; ++i)
                update_state_parameters(
                    tmp_aad[i], AAD_OFFSET(decay_), offset * num_factors + i, state_parameters_aad);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            // AAD: output += vol * vol * dcf;
            tmp_aad = ret_aad * (2. * dcf) * vol;

            parameter_.volatility_aad(tmp_aad, vol_offset, state_parameters_aad);
        }
        break;
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::money_market_variance(size_t offset, vector<double>& output) const
{
    auto from = model_dates_[offset];
    auto to   = model_dates_[offset + 1];

    if (to.as_float() > from.as_float())
    {
        const auto  vol_offset         = volatilities_dates_indexs_[offset + 1];
        const auto  dcf                = parameter_.day_convention()->fraction(from, to);
        const auto& vol                = parameter_.volatilities()[vol_offset];
        const auto& decay_tmp          = decay_[offset];
        const auto& integral_decay_tmp = integral_decay_[offset];

        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto&    mrs = parameter_.decays()[decays_dates_indexs_[offset + 1]];
            vector<double> vol_decay(parameter_.number_of_factors());
            vol_decay = vol / decay_tmp;

            vector<double> beta = if_else(
                fabs(mrs) < std::numeric_limits<double>::epsilon(), decay_tmp, decay_tmp / mrs);

            vector<double> alpha = if_else(
                fabs(mrs) < std::numeric_limits<double>::epsilon(),
                integral_decay_tmp,
                integral_decay_tmp + beta);

            for (size_t f = 0; f < parameter_.number_of_factors(); f++)
                output[f] +=
                    vol_decay[f] * vol_decay[f] *
                    details::intermediate_money_market_variance(dcf, mrs[f], alpha[f], beta[f]);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            for (size_t f = 0; f < parameter_.number_of_factors(); f++)
                output[f] += vol[f] * vol[f] *
                             details::intermediate_money_market_variance(
                                 dcf, 0., integral_decay_tmp[f], decay_tmp[f]);
        }
        break;
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::money_market_variance_aad(
    const vector<double>& ret_aad, size_t offset, double* state_parameters_aad) const
{
    if (std::all_of(
            ret_aad.begin(), ret_aad.end(), [](double x) { return cobra::is_almost_zero(x); }))
        return;

    auto from = model_dates_[offset];
    auto to   = model_dates_[offset + 1];

    if (to.as_float() > from.as_float())
    {
        auto       num_factors = parameter_.number_of_factors();
        const auto vol_offset  = volatilities_dates_indexs_[offset + 1];

        const auto  dcf                = parameter_.day_convention()->fraction(from, to);
        const auto& vol                = parameter_.volatilities()[vol_offset];
        const auto& decay_tmp          = decay_[offset];
        const auto& integral_decay_tmp = integral_decay_[offset];

        vector<double> vol_aad(num_factors);
        vol_aad = 0.;
        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto& mrs = parameter_.decays()[decays_dates_indexs_[offset + 1]];

            vector<double> vol_decay = vol / decay_tmp;
            vector<double> beta      = if_else(
                fabs(mrs) < std::numeric_limits<double>::epsilon(), decay_tmp, decay_tmp / mrs);

            vector<double> alpha = if_else(
                fabs(mrs) < std::numeric_limits<double>::epsilon(),
                integral_decay_tmp,
                integral_decay_tmp + beta);

            vector<double> vol_decay_aad(num_factors);
            vector<double> beta_aad(num_factors);
            vector<double> alpha_aad(num_factors);
            vector<double> mrs_aad(num_factors);

            vol_decay_aad = 0.;
            beta_aad      = 0.;
            alpha_aad     = 0.;
            mrs_aad       = 0.;

            for (size_t f = 0; f < num_factors; f++)
            {
                double value =
                    details::intermediate_money_market_variance(dcf, mrs[f], alpha[f], beta[f]);
                // output[f] +=vol_decay[f] * vol_decay[f]
                // *details::intermediate_money_market_variance(dcf, mrs[f], alpha[f], beta[f]);
                vol_decay_aad[f] += ret_aad[f] * 2. * vol_decay[f] * value;

                auto value_aad = ret_aad[f] * vol_decay[f] * vol_decay[f];

                details::intermediate_money_market_variance_aad(
                    value_aad,
                    value,
                    dcf,
                    mrs[f],
                    alpha[f],
                    beta[f],
                    mrs_aad[f],
                    alpha_aad[f],
                    beta_aad[f]);
            }
            // AAD: alpha = if_else(fabs(mrs) <
            // std::numeric_limits<double>::epsilon(),integral_decay_tmp, integral_decay_tmp +
            // beta);
            beta_aad += if_else(fabs(mrs) < std::numeric_limits<double>::epsilon(), 0., alpha_aad);

            for (size_t f = 0; f < num_factors; ++f)
                update_state_parameters(
                    alpha_aad[f],
                    AAD_OFFSET(integral_decay_),
                    offset * num_factors + f,
                    state_parameters_aad);

            // AAD: beta = if_else(fabs(mrs) < std::numeric_limits<double>::epsilon(), decay_tmp,
            // decay_tmp / mrs);

            beta_aad *= beta;
            for (size_t f = 0; f < num_factors; ++f)
            {
                update_state_parameters(
                    beta_aad[f],
                    AAD_OFFSET(decay_),
                    offset * num_factors + f,
                    state_parameters_aad);

                if (!cobra::is_almost_zero(mrs[f]))
                    mrs_aad[f] -= beta_aad[f] / mrs[f];
            }

            parameter_.decays_aad(mrs_aad, decays_dates_indexs_[offset + 1], state_parameters_aad);

            // AAD: vol_decay = vol / decay_tmp;
            vol_aad = vol_decay_aad / decay_tmp;
            parameter_.volatility_aad(vol_aad, vol_offset, state_parameters_aad);

            vol_aad *= -vol;
            for (size_t f = 0; f < num_factors; ++f)
                update_state_parameters(
                    vol_aad[f], AAD_OFFSET(decay_), offset * num_factors + f, state_parameters_aad);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            for (size_t f = 0; f < parameter_.number_of_factors(); f++)
            {
                double value = details::intermediate_money_market_variance(
                    dcf, 0., integral_decay_tmp[f], decay_tmp[f]);
                // AAD: output[f] +=vol[f] * vol[f]
                // *details::intermediate_money_market_variance(dcf, 0., integral_decay_tmp[f], decay_tmp[f]);
                vol_aad[f] = ret_aad[f] * 2. * vol[f] * value;

                double value_aad              = ret_aad[f] * vol[f] * vol[f];
                double integral_decay_tmp_aad = 0;
                double decay_tmp_aad          = 0;
                double mrs_aad                = 0.;

                details::intermediate_money_market_variance_aad(
                    value_aad,
                    value,
                    dcf,
                    0.,
                    integral_decay_tmp[f],
                    decay_tmp[f],
                    mrs_aad,
                    integral_decay_tmp_aad,
                    decay_tmp_aad);

                update_state_parameters(
                    decay_tmp_aad,
                    AAD_OFFSET(decay_),
                    offset * num_factors + f,
                    state_parameters_aad);

                update_state_parameters(
                    integral_decay_tmp_aad,
                    AAD_OFFSET(integral_decay_),
                    offset * num_factors + f,
                    state_parameters_aad);
            }

            parameter_.volatility_aad(vol_aad, vol_offset, state_parameters_aad);
            break;
        }
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::incremental_volatility_sv(
    const std::vector<datetime>& dates, matrix<double>& output) const
{
    vector<double> var_prev(parameter_.number_of_factors());
    vector<double> var(parameter_.number_of_factors());
    var_prev  = 0.;
    var       = 0.;
    size_t k  = 1;
    output[0] = 0;
    for (size_t i = 1; i < model_dates_.size(); ++i)
    {
        auto d = model_dates_[i];

        if (d > dates.back())
            return;

        states_variable_variance(i - 1, var);
        if (dates[k] == d)
        {
            output[k] = sqrt(var - var_prev);
            var_prev.deepcopy(var);
            k++;
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::incremental_volatility_sv_aad(
    const matrix<double>&        ret_aad,
    const matrix<double>&        output,
    const std::vector<datetime>& dates,
    double*                      state_parameters_aad) const
{
    if (std::all_of(
            ret_aad.begin(), ret_aad.end(), [](double x) { return cobra::is_almost_zero(x); }))
        return;

    vector<double> var_aad(parameter_.number_of_factors());
    vector<double> var_prev_aad(parameter_.number_of_factors());
    vector<double> tmp_aad(parameter_.number_of_factors());

    var_aad      = 0.;
    var_prev_aad = 0.;
    size_t k     = dates.size() - 1;
    for (size_t i = model_dates_.size() - 1; i >= 1; --i)
    {
        auto d = model_dates_[i];

        if (d > dates.back())
            continue;

        if (dates[k] == d)
        {
            // ret_aad[k]= sqrt(var - var_prev);
            tmp_aad = 0.5 * ret_aad[k] / output[k];
            var_aad += var_prev_aad + tmp_aad;
            var_prev_aad = -tmp_aad;
            k--;
        }
        // states_variable_variance(i - 1, var);
        states_variable_variance_aad(var_aad, i - 1, state_parameters_aad);
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::incremental_volatility_mm(
    const std::vector<datetime>& dates, matrix<double>& output) const
{
    vector<double> var_prev(parameter_.number_of_factors());
    vector<double> var(parameter_.number_of_factors());
    var_prev  = 0.;
    var       = 0.;
    size_t k  = 1;
    output[0] = 0;

    const auto last_date = dates.back();

    for (size_t i = 1; i < model_dates_.size(); ++i)
    {
        auto d = model_dates_[i];

        if (d > last_date)
            return;

        money_market_variance(i - 1, var);
        if (dates[k] == d)
        {
            output[k] = sqrt(var - var_prev);
            var_prev.deepcopy(var);
            k++;
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::incremental_volatility_mm_aad(
    const matrix<double>&        ret_aad,
    const matrix<double>&        output,
    const std::vector<datetime>& dates,
    double*                      state_parameters_aad) const
{
    if (std::all_of(
            ret_aad.begin(), ret_aad.end(), [](double x) { return cobra::is_almost_zero(x); }))
        return;

    vector<double> var_aad(parameter_.number_of_factors());
    vector<double> var_prev_aad(parameter_.number_of_factors());
    vector<double> tmp_aad(parameter_.number_of_factors());

    var_aad      = 0.;
    var_prev_aad = 0.;
    size_t k     = dates.size() - 1;
    for (size_t i = model_dates_.size() - 1; i >= 1; --i)
    {
        auto d = model_dates_[i];

        if (d > dates.back())
            continue;

        if (dates[k] == d)
        {
            // ret_aad[k]= sqrt(var - var_prev);
            tmp_aad = 0.5 * ret_aad[k] / output[k];
            var_aad += var_prev_aad + tmp_aad;
            var_prev_aad = -tmp_aad;
            k--;
        }
        // money_market_variance(i - 1, var);
        money_market_variance_aad(var_aad, i - 1, state_parameters_aad);
    }
}

//-----------------------------------------------------------------------------
// functions used for the calibration
//-----------------------------------------------------------------------------
void model_markovian_hjm::states_variable_covariance(
    const datetime& to, matrix<double>& output) const
{
    states_variable_covariance(to, date_offset(to), output);
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::states_variable_covariance_aad(
    matrix<double>& ret_aad, const datetime& to, double* state_parameters_aad) const
{
    states_variable_covariance_aad(ret_aad, to, date_offset(to), state_parameters_aad);
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::states_variable_covariance(
    const datetime& to, size_t offset, matrix<double>& output) const
{
    auto from = model_dates_[offset];

    output.deepcopy(covariance_matrix_.get_matrix(static_cast<int>(offset)));

    if (to.as_float() > from.as_float())
    {
        auto num_of_factors = parameter_.number_of_factors();

        const auto  dcf = parameter_.day_convention()->fraction(from, to);
        const auto& vol = parameter_.volatilities()[volatilities_dates_indexs_[offset + 1]];

        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto& mrs       = parameter_.decays()[decays_dates_indexs_[offset + 1]];
            const auto& decay_tmp = decay_[offset];

            for (size_t f1 = 0; f1 < num_of_factors; f1++)
            {
                auto covariance_f1 = output[f1];
                auto rho_f1        = parameter_.correlation()[f1];
                auto vol_decay     = vol[f1] / decay_tmp[f1];
                for (size_t f2 = 0; f2 <= f1; f2++)
                {
                    double covar = details::state_variables_covariance_helper(f1, f2, mrs, dcf);
                    covariance_f1[f2] += vol_decay * covar * rho_f1[f2] * vol[f2] / decay_tmp[f2];
                }
            }
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            for (size_t f1 = 0; f1 < num_of_factors; f1++)
            {
                auto covariance_f1 = output[f1];
                auto rho_f1        = parameter_.correlation()[f1];
                for (size_t f2 = 0; f2 <= f1; f2++)
                {
                    covariance_f1[f2] += vol[f1] * rho_f1[f2] * vol[f2] * dcf;
                }
            }
        }
        break;
        }

        for (size_t f1 = 0; f1 < num_of_factors; f1++)
        {
            for (size_t f2 = 0; f2 < f1; f2++)
            {
                output[f2][f1] = output[f1][f2];
            }
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::states_variable_covariance_aad(
    matrix<double>& ret_aad, const datetime& to, size_t offset, double* state_parameters_aad) const
{
    if (std::all_of(
            ret_aad.begin(), ret_aad.end(), [](double x) { return cobra::is_almost_zero(x); }))
        return;

    auto from = model_dates_[offset];

    if (to.as_float() > from.as_float())
    {
        auto num_of_factors = parameter_.number_of_factors();
        bool continue_aad   = false;
        if (num_of_factors > 1)
        {
            for (size_t f1 = 0; f1 < num_of_factors; f1++)
            {
                for (size_t f2 = 0; f2 < f1; f2++)
                {
                    ret_aad[f1][f2] += ret_aad[f2][f1];
                    ret_aad[f2][f1] = 0.;
                    if (!cobra::is_almost_zero(ret_aad[f1][f2]))
                        continue_aad = true;
                }
            }
        }
        else
        {
            if (!cobra::is_almost_zero(ret_aad[0][0]))
                continue_aad = true;
        }
        if (!continue_aad)
            return;

        const auto& correlation = parameter_.correlation();
        const auto  dcf         = parameter_.day_convention()->fraction(from, to);
        const auto& vol         = parameter_.volatilities()[volatilities_dates_indexs_[offset + 1]];

        vector<double> vol_aad(num_of_factors);
        matrix<double> rho_aad(num_of_factors, num_of_factors);
        vol_aad = 0;
        rho_aad = 0.;

        switch (parameter_.model_type())
        {
        case parameter_markovian_hjm_type::HULL_WHITE:
        {
            const auto& mrs       = parameter_.decays()[decays_dates_indexs_[offset + 1]];
            const auto& decay_tmp = decay_[offset];

            vector<double> vol_decay = vol / decay_tmp;

            vector<double> mrs_aad(num_of_factors);
            mrs_aad = 0.;

            for (size_t f1 = 0; f1 < num_of_factors; f1++)
            {
                const auto& ret_f1_aad   = ret_aad[f1];
                const auto& rho_f1       = correlation[f1];
                const auto  vol_decay_f1 = vol_decay[f1];

                vector<double> rho_f1_aad(rho_aad[f1]);

                auto& vol_aad_f1 = vol_aad[f1];

                for (size_t f2 = 0; f2 <= f1; f2++)
                {
                    const auto rho_f1f2     = rho_f1[f2];
                    const auto vol_decay_f2 = vol_decay[f2];
                    auto&      vol_aad_f2   = vol_aad[f2];

                    const auto covar = details::state_variables_covariance_helper(f1, f2, mrs, dcf);

                    // AAD: covariance_f1[f2] += vol_decay * covar * rho_f1[f2] * vol_decay[f2];
                    auto value_aad = ret_f1_aad[f2] * covar * rho_f1f2;
                    vol_aad_f2 += value_aad * vol_decay_f1;
                    vol_aad_f1 += value_aad * vol_decay_f2;

                    value_aad = ret_f1_aad[f2] * vol_decay_f1 * vol_decay_f2;

                    rho_f1_aad[f2] += value_aad * covar;

                    // AAD: double covar = details::state_variables_covariance_helper(f1, f2, mrs,
                    // dcf);
                    const auto covar_aad = value_aad * rho_f1f2;
                    details::state_variables_covariance_helper_aad(
                        covar_aad, f1, f2, mrs, dcf, mrs_aad);
                }
            }

            vol_aad /= decay_tmp;

            parameter_.correlation_aad(rho_aad, state_parameters_aad);
            parameter_.decays_aad(mrs_aad, decays_dates_indexs_[offset + 1], state_parameters_aad);
            parameter_.volatility_aad(
                vol_aad, volatilities_dates_indexs_[offset + 1], state_parameters_aad);

            vol_aad *= -vol;
            for (size_t i = 0; i < num_of_factors; ++i)
                update_state_parameters(
                    vol_aad[i],
                    AAD_OFFSET(decay_),
                    offset * num_of_factors + i,
                    state_parameters_aad);
        }
        break;

        case parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        {
            // AAD: output += vol * vol * dcf;
            for (size_t f1 = 0; f1 < num_of_factors; f1++)
            {
                vector<double> ret_f1_aad(ret_aad[f1]);
                vector<double> rho_f1(correlation[f1]);
                vector<double> rho_f1_aad(rho_aad[f1]);

                double     vol_f1_aad = 0.;
                const auto vol_f1     = vol[f1];
                for (size_t f2 = 0; f2 <= f1; f2++)
                {
                    auto value_aad = dcf * ret_f1_aad[f2];
                    // AAD: covariance_f1[f2] += vol[f1] * rho_f1[f2] * vol[f2] * dcf;
                    vol_f1_aad += value_aad * rho_f1[f2] * vol[f2];
                    value_aad *= vol_f1;
                    rho_f1_aad[f2] += value_aad * vol[f2];
                    vol_aad[f2] += value_aad * rho_f1[f2];
                }
                vol_aad[f1] += vol_f1_aad;
            }
            parameter_.volatility_aad(
                vol_aad, volatilities_dates_indexs_[offset + 1], state_parameters_aad);
            parameter_.correlation_aad(rho_aad, state_parameters_aad);
        }
        break;
        }
    }

    // AAD: output.deepcopy(states_variable_variance_[offset]);
    auto num_factors = parameter_.number_of_factors();
    for (size_t f1 = 0; f1 < num_factors; ++f1)
        for (size_t f2 = 0; f2 <= f1; ++f2)
            update_state_parameters(
                ret_aad[f1][f2],
                AAD_OFFSET(covariance_matrix_),
                offset * num_factors * num_factors + f1 * num_factors + f2,
                state_parameters_aad);
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::cholseky_decomposition_states_covariance(
    const datetime& expiry, matrix<double>& cholseky_matrix) const
{
    auto num_of_factors = parameter_.number_of_factors();

    cholseky_matrix = 0.;

    matrix<double> covariance(num_of_factors, num_of_factors);
    states_variable_covariance(expiry, covariance);

    if (num_of_factors > 1)
    {
        cholesky_decomposition(
            covariance.begin(),
            cholseky_matrix.begin(),
            num_of_factors,
            cobra::cholesky_decomposition_type::LOWER_TRIANGULAR);
    }
    else
    {
        cholseky_matrix[0][0] = sqrt(covariance[0][0]);
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::cholseky_decomposition_states_covariance_aad(
    matrix<double>&       cholesky_matrix_aad,
    const matrix<double>& cholesky_matrix,
    const datetime&       expiry,
    double*               state_parameters_aad) const
{
    if (std::all_of(
            cholesky_matrix_aad.begin(),
            cholesky_matrix_aad.end(),
            [](double x) { return cobra::is_almost_zero(x); }))
        return;

    auto num_of_factors = parameter_.number_of_factors();

    matrix<double> covariance_aad(num_of_factors, num_of_factors);
    covariance_aad = 0.;

    if (num_of_factors > 1)
    {
        cholesky_decomposition_aad(
            cholesky_matrix_aad.begin(),
            cholesky_matrix.begin(),
            num_of_factors,
            cobra::cholesky_decomposition_type::LOWER_TRIANGULAR,
            covariance_aad.begin());
    }
    else
    {
        // AAD: cholseky_matrix = sqrt(covariance);
        covariance_aad[0][0] = 0.5 * cholesky_matrix_aad[0][0] / cholesky_matrix[0][0];
    }

    states_variable_covariance_aad(covariance_aad, expiry, state_parameters_aad);
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::decompose(
    matrix<double>&              integral_decays,
    const datetime&              expiry,
    const std::vector<datetime>& tenors) const
{
    const auto expiry_offset  = date_offset(expiry);
    const auto num_of_factors = parameter_.number_of_factors();

    vector<double> integral_decay_expiry(num_of_factors);
    integral_decay(expiry, expiry_offset, integral_decay_expiry);

    matrix<double> cheolseky_matrix(num_of_factors, num_of_factors);
    cholseky_decomposition_states_covariance(expiry, cheolseky_matrix);

    vector<double> tmp(num_of_factors);

    size_t i = 0;
    for (const auto& t : tenors)
    {
        vector<double> integral_decay_tmp(integral_decays[i]);
        integral_decay(t, integral_decay_tmp);

        integral_decay_tmp = integral_decay_tmp - integral_decay_expiry;

        tmp = 0.;
        for (size_t f = 0; f < num_of_factors; ++f)
            tmp = tmp + integral_decay_tmp[f] * cheolseky_matrix[f];

        integral_decay_tmp.deepcopy(tmp);
        i++;
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::decompose_aad(
    matrix<double>&              integral_decays_aad,
    const datetime&              expiry,
    const std::vector<datetime>& tenors,
    double*                      state_parameters_aad) const
{
    const auto expiry_offset  = date_offset(expiry);
    const auto num_of_factors = parameter_.number_of_factors();

    vector<double> integral_decay_expiry(num_of_factors);
    integral_decay(expiry, expiry_offset, integral_decay_expiry);

    matrix<double> cheolesky_matrix(num_of_factors, num_of_factors);
    cholseky_decomposition_states_covariance(expiry, cheolesky_matrix);

    vector<double> integral_decay_tmp(num_of_factors);
    vector<double> integral_decay_tmp_aad(num_of_factors);

    vector<double> integral_decay_expiry_aad(num_of_factors);
    integral_decay_expiry_aad = 0.;

    matrix<double> cheolesky_matrix_aad(num_of_factors, num_of_factors);
    cheolesky_matrix_aad = 0.;

    for (int i = static_cast<int>(tenors.size()) - 1; i >= 0; --i)
    {
        auto t      = tenors[i];
        auto offset = date_offset(t);

        integral_decay(t, offset, integral_decay_tmp);
        integral_decay_tmp -= integral_decay_expiry;

        const auto value_aad = integral_decays_aad[i];
        for (size_t f = 0; f < num_of_factors; ++f)
        {
            // AAD: tmp = tmp + integral_decay_tmp[f] * cheolseky_matrix[f];
            cheolesky_matrix_aad[f] += value_aad * integral_decay_tmp[f];
            integral_decay_tmp_aad[f] = accumulate(value_aad * cheolesky_matrix[f]);
        }

        // AAD: integral_decay_tmp = integral_decay_tmp - integral_decay_expiry;
        integral_decay_expiry_aad -= integral_decay_tmp_aad;

        integral_decay_aad(integral_decay_tmp_aad, t, offset, state_parameters_aad);
    }

    cholseky_decomposition_states_covariance_aad(
        cheolesky_matrix_aad, cheolesky_matrix, expiry, state_parameters_aad);

    integral_decay_aad(integral_decay_expiry_aad, expiry, expiry_offset, state_parameters_aad);
}

//-----------------------------------------------------------------------------
// helper functions
//-----------------------------------------------------------------------------
const std::vector<datetime>& model_markovian_hjm::model_dates() const
{
    return model_dates_;
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::initialize()
{
    if (!calibration_flag_)
    {
        model_dates_.assign(dates_.begin(), dates_.end());
    }
    else
    {
        model_dates_.push_back(valution_date_);
    }
    datetime_helper::merges_dates(
        parameter_.decays_dates(), parameter_.volatilities_dates(), model_dates_);

    if (!extra_dates_.empty())
        datetime_helper::merges_dates(extra_dates_, model_dates_, model_dates_);

    datetime_helper::interpolate_from_subset_dates(
        model_dates_, parameter_.decays_dates(), decays_dates_indexs_);

    datetime_helper::interpolate_from_subset_dates(
        model_dates_, parameter_.volatilities_dates(), volatilities_dates_indexs_);

    if (!calibration_flag_)
        datetime_helper::interpolate_from_subset_dates(dates_, model_dates_, dates_indexs_);

    const auto num_of_dates   = model_dates_.size();
    const auto num_of_factors = parameter_.number_of_factors();

    decay_          = matrix<double>(num_of_dates, num_of_factors);
    integral_decay_ = matrix<double>(num_of_dates, num_of_factors);

    AAD_REGISTER_PARAMETER(decay_, decay_.size());
    AAD_REGISTER_PARAMETER(integral_decay_, integral_decay_.size());

    if (calibration_flag_)
    {
        covariance_matrix_ = tensor<double>({num_of_dates, num_of_factors, num_of_factors});
        AAD_REGISTER_PARAMETER(covariance_matrix_, covariance_matrix_.size());
    }
    else
    {
        const auto size = dates_.size();

        incremental_volatility_sv_ = matrix<double>(size, num_of_factors);
        incremental_volatility_mm_ = matrix<double>(size, num_of_factors);
        covariance_matrix_         = tensor<double>({size, num_of_factors, num_of_factors});

        AAD_REGISTER_PARAMETER(incremental_volatility_sv_, incremental_volatility_sv_.size());
        AAD_REGISTER_PARAMETER(incremental_volatility_mm_, incremental_volatility_mm_.size());
        AAD_REGISTER_PARAMETER(covariance_matrix_, covariance_matrix_.size());
    }

    fill();
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::fill()
{
    size_t num_of_dates = model_dates_.size();

    decay_             = 0.;
    integral_decay_    = 0.;
    covariance_matrix_ = 0.;

    switch (parameter_.model_type())
    {
    case cobra::parameter_markovian_hjm_type::HULL_WHITE:
        decay_[0] = 1.;
        break;

    case cobra::parameter_markovian_hjm_type::PICEWISE_CONSTANT:
        decay_[0].deepcopy(parameter_.decays()[0]);
        break;
    }

    for (size_t t = 1; t < num_of_dates; ++t)
    {
        auto d = model_dates_[t];

        auto decay_t = decay_[t];
        decay(d, t - 1, decay_t);

        auto integral_decay_t = integral_decay_[t];
        integral_decay(d, t - 1, integral_decay_t);

        if (calibration_flag_)
        {
            auto m = covariance_matrix_.get_matrix(static_cast<int>(t));
            states_variable_covariance(d, t - 1, m);
        }
    }

    size_t size = dates_.size();
    if (!calibration_flag_ && size > 0)
    {
        size_t num_factors = parameter_.number_of_factors();

        incremental_volatility_sv(dates_, incremental_volatility_sv_);
        incremental_volatility_mm(dates_, incremental_volatility_mm_);

        for (size_t i = 0; i < size; ++i)
        {
            vector sigma = incremental_volatility_sv_[i];

            matrix m(covariance_matrix_.get_matrix(static_cast<int>(i)));

            for (size_t f = 0; f < num_factors; ++f)
                m[f] = sigma[f] * parameter_.correlation()[f] * sigma;

            if (i > 0)
            {
                matrix tmp_prev = covariance_matrix_.get_matrix(static_cast<int>(i) - 1);
                m               = m + tmp_prev;
            }
        }
    }
}

//-----------------------------------------------------------------------------
void model_markovian_hjm::fill_aad(double* state_parameters_aad) const
{
    size_t num_of_dates   = model_dates_.size();
    size_t num_of_factors = parameter_.number_of_factors();

    size_t size = dates_.size();
    if (!calibration_flag_ && size > 0)
    {
        size_t num_factors = parameter_.number_of_factors();

        vector<double> sigma_aad(num_of_factors);
        vector<double> tmp_aad(num_of_factors);
        matrix<double> rho_aad(num_of_factors, num_of_factors);

        sigma_aad = 0.;
        rho_aad   = 0.;
        for (int i = static_cast<int>(size - 1); i >= 0; --i)
        {
            matrix<double> m_aad(
                get_state_parameters_aad(
                    AAD_OFFSET(covariance_matrix_),
                    i * num_of_factors * num_of_factors,
                    state_parameters_aad),
                num_of_factors,
                num_of_factors);

            const auto& sigma = incremental_volatility_sv_[i];
            tmp_aad           = 0.;
            for (size_t f = 0; f < num_factors; ++f)
            {
                // m[f] = sigma[f] * parameter_.correlation()[f] * sigma;
                tmp_aad += m_aad[f] * sigma[f] * parameter_.correlation()[f];
                tmp_aad[f] += accumulate(m_aad[f] * parameter_.correlation()[f] * sigma);
                rho_aad[f] += sigma[f] * m_aad[f] * sigma;
            }

            sigma_aad += tmp_aad;

            size_t m = number_of_factors();
            for (size_t j = 0; j < m; ++j)
                update_state_parameters(
                    sigma_aad[j],
                    AAD_OFFSET(incremental_volatility_sv_),
                    i * m + j,
                    state_parameters_aad);
        }

        parameter_.correlation_aad(rho_aad, state_parameters_aad);

        {
            matrix<double> m_aad(
                get_state_parameters_aad(
                    AAD_OFFSET(incremental_volatility_sv_), 0, state_parameters_aad),
                incremental_volatility_sv_.rows(),
                incremental_volatility_sv_.columns());

            incremental_volatility_sv_aad(
                m_aad, incremental_volatility_sv_, dates_, state_parameters_aad);
        }
        {
            matrix<double> m_aad(
                get_state_parameters_aad(
                    AAD_OFFSET(incremental_volatility_mm_), 0, state_parameters_aad),
                incremental_volatility_mm_.rows(),
                incremental_volatility_mm_.columns());

            incremental_volatility_mm_aad(
                m_aad, incremental_volatility_mm_, dates_, state_parameters_aad);
        }
    }
    for (int t = static_cast<int>(num_of_dates) - 1; t >= 1; --t)
    {
        auto d = model_dates_[t];
        if (calibration_flag_)
        {
            matrix<double> m_aad(
                get_state_parameters_aad(
                    AAD_OFFSET(covariance_matrix_),
                    t * num_of_factors * num_of_factors,
                    state_parameters_aad),
                num_of_factors,
                num_of_factors);

            states_variable_covariance_aad(m_aad, d, t - 1, state_parameters_aad);
        }
        {
            vector<double> v_aad(
                get_state_parameters_aad(
                    AAD_OFFSET(integral_decay_), t * num_of_factors, state_parameters_aad),
                num_of_factors);

            integral_decay_aad(v_aad, d, t - 1, state_parameters_aad);
        }
        {
            vector<double> v_aad(
                get_state_parameters_aad(
                    AAD_OFFSET(decay_), t * num_of_factors, state_parameters_aad),
                num_of_factors);

            log_decay_aad(v_aad, d, t - 1, state_parameters_aad);
        }
    }
    if (parameter_.model_type() == cobra::parameter_markovian_hjm_type::PICEWISE_CONSTANT)
    {
        vector<double> v_aad(
            get_state_parameters_aad(AAD_OFFSET(decay_), 0, state_parameters_aad), num_of_factors);

        parameter_.decays_aad(v_aad, 0, state_parameters_aad);
    }
}

//-----------------------------------------------------------------------------
size_t model_markovian_hjm::date_offset(const datetime& t) const
{
    return datetime_helper::date_offset(t, model_dates_);
}

//-----------------------------------------------------------------------------
// AAD state manager functions
//-----------------------------------------------------------------------------
void model_markovian_hjm::finalize_aad([[maybe_unused]] double* parameters) const {};

//-----------------------------------------------------------------------------
size_t model_markovian_hjm::get_state_parameters(double* parameters) const
{
    size_t offset = 0;
    for (auto t : decay_)
    {
        parameters[offset++] = t;
    }
    for (auto t : integral_decay_)
    {
        parameters[offset++] = t;
    }
    if (!calibration_flag_)
    {
        for (auto t : incremental_volatility_sv_)
        {
            parameters[offset++] = t;
        }
        for (auto t : incremental_volatility_mm_)
        {
            parameters[offset++] = t;
        }
    }
    for (auto t : covariance_matrix_)
    {
        parameters[offset++] = t;
    }

    COBRA_CHECK_DEBUG(
        offset == state_parameters_size(),
        "expect size parameter ",
        state_parameters_size(),
        " while provided ",
        offset);

    return offset;
}

//-----------------------------------------------------------------------------
size_t model_markovian_hjm::set_state_parameters(const double* parameters)
{
    size_t offset = 0;
    for (auto& t : decay_)
    {
        t = parameters[offset++];
    }
    for (auto& t : integral_decay_)
    {
        t = parameters[offset++];
    }
    if (!calibration_flag_)
    {
        for (auto& t : incremental_volatility_sv_)
        {
            t = parameters[offset++];
        }
        for (auto& t : incremental_volatility_mm_)
        {
            t = parameters[offset++];
        }
    }
    for (auto& t : covariance_matrix_)
    {
        t = parameters[offset++];
    }

    COBRA_CHECK_DEBUG(
        offset == state_parameters_size(),
        "expect size parameter ",
        state_parameters_size(),
        " while provided ",
        offset);

    return offset;
}

//-----------------------------------------------------------------------------
// PDE
void model_markovian_hjm::conditional_price(
    double*               output,
    std::vector<size_t>   pde_state_dimensions,
    const vector<double>& states,
    const datetime&       from,
    const datetime&       to) const
{
    auto num_factors = parameter_.number_of_factors();

    vector<double> G_from(num_factors);
    integral_decay(from, G_from);

#ifndef RISK_NEUTRAL
    G_from = 0.;
#endif  // !RISK_NEUTRAL

    vector<double> G_to(num_factors);
    integral_decay(to, G_to);

    vector<double> G_diff = G_to - G_from;
    vector<double> G_add  = G_to + G_from;

    const auto var_matrix =
        covariance_matrix_.get_matrix(static_cast<int>(datetime_helper::date_offset(from, dates_)));

    const auto stdev = accumulate(G_diff * (var_matrix * G_add));

    tensor<double> tmp(output, pde_state_dimensions);
    tmp = -0.5 * stdev;

    for (size_t f = 0; f < num_factors; ++f)
    {
        vector<double> X(states.data(), pde_state_dimensions[f]);
        tmp -= G_diff[f] * X;
    }
};

//-----------------------------------------------------------------------------
void model_markovian_hjm::coefficients(
    ptr_const<cobra::pde_model_coefficients>* output,
    [[maybe_unused]] const vector<double>&    states,
    const datetime&                           from,
    [[maybe_unused]] const datetime&          to) const
{
    auto           num_factors = parameter_.number_of_factors();
    const auto     from_index  = date_offset(from);
    vector<double> sigma_sv(incremental_volatility_sv_[from_index]);

#ifdef RISK_NEUTRAL

    const auto to_index = date_offset(to);

    vector<double> sigma_mm(incremental_volatility_mm_[from_index]);

    const auto G_to   = integral_decay()[to_index];
    const auto G_from = integral_decay()[from_index];
    const auto var_matrix =
        covariance_matrix_.get_matrix(static_cast<int>(datetime_helper::date_offset(from, dates_)));

    const auto stdev = accumulate((G_to - G_from) * (var_matrix * (G_to + G_from)));

    vector<double> drift_sv = sigma_sv * (parameter_.correlation() * sigma_mm);

#endif  // RISK_NEUTRAL

    for (size_t f = 0; f < num_factors; ++f)
    {
        vector<double> reaction(output[f]->reaction().data(), output[f]->reaction().size());
        auto&          convection = const_cast<cobra::tensor<double>&>(output[f]->convection());

#ifndef RISK_NEUTRAL
        reaction   = 0.;
        convection = 0.;
#else
        reaction   = -((G_to[f] - G_from[f]) * states + 0.5 * stdev);
        convection = -drift_sv[f];
#endif

        auto& diffusion = const_cast<cobra::tensor<double>&>(output[f]->diffusion());
        diffusion       = 0.5 * std::sqr(sigma_sv[f]);
    }
};
}  // namespace cobra
