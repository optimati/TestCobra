#include "Analytics/parameters/parameter_markovian_hjm.h"

#include "Core/util/exception.h"             // for COBRA_CHECK
#include "Util/calendar.h"                   // IWYU pragma: keep
#include "Vectorization/terminals/vector.h"  // for vector

namespace cobra
{
//-----------------------------------------------------------------------------
parameter_markovian_hjm::parameter_markovian_hjm() = default;

//-----------------------------------------------------------------------------
parameter_markovian_hjm::parameter_markovian_hjm(
    const matrix<double>&                  decays,
    const matrix<double>&                  volatilities,
    const matrix<double>&                  correlation,
    const std::vector<datetime>&           decays_dates,
    const std::vector<datetime>&           volatilities_dates,
    const ptr_const<day_count_convention>& day_convention,
    parameter_markovian_hjm_type           model_type)
    : decays_(decays.rows(), decays.columns()),
      volatilities_(volatilities.rows(), volatilities.columns()),
      correlation_(correlation.rows(), correlation.columns()),
      decays_dates_(decays_dates),
      volatilities_dates_(volatilities_dates),
      day_convention_(day_convention),
      model_type_(model_type)
{
    decays_.deepcopy(decays);
    volatilities_.deepcopy(volatilities);
    correlation_.deepcopy(correlation);

    initialize();
}

//-----------------------------------------------------------------------------
parameter_markovian_hjm::~parameter_markovian_hjm() = default;

//-----------------------------------------------------------------------------
void parameter_markovian_hjm::initialize()
{
    validate();
    AAD_REGISTER_PARAMETER(decays_, decays_.size());
    AAD_REGISTER_PARAMETER(volatilities_, volatilities_.size());
    AAD_REGISTER_PARAMETER(correlation_, number_of_factors() * (number_of_factors() - 1) / 2);
}

//-----------------------------------------------------------------------------
const matrix<double>& parameter_markovian_hjm::decays() const
{
    return decays_;
};

//-----------------------------------------------------------------------------
const matrix<double>& parameter_markovian_hjm::volatilities() const
{
    return volatilities_;
};

//-----------------------------------------------------------------------------
const matrix<double>& parameter_markovian_hjm::correlation() const
{
    return correlation_;
};

//-----------------------------------------------------------------------------
const std::vector<cobra::datetime>& parameter_markovian_hjm::decays_dates() const
{
    return decays_dates_;
};

//-----------------------------------------------------------------------------
const std::vector<cobra::datetime>& parameter_markovian_hjm::volatilities_dates() const
{
    return volatilities_dates_;
};

//-----------------------------------------------------------------------------
const ptr_const<day_count_convention>& parameter_markovian_hjm::day_convention() const
{
    return day_convention_;
};

//-----------------------------------------------------------------------------
cobra::parameter_markovian_hjm_type parameter_markovian_hjm::model_type() const
{
    return model_type_;
};

//-----------------------------------------------------------------------------
size_t parameter_markovian_hjm::number_of_factors() const
{
    return decays_.columns();
};

//-----------------------------------------------------------------------------
void parameter_markovian_hjm::validate()
{
    COBRA_CHECK(
        decays_.rows() == decays_dates_.size(),
        "decays size ",
        decays_.rows(),
        " is differents from decays dates size ",
        decays_dates_.size());

    COBRA_CHECK(
        volatilities_.rows() == volatilities_dates_.size(),
        "volatilities size ",
        volatilities_.rows(),
        " is differents from volatilities dates size ",
        volatilities_dates_.size());

    COBRA_CHECK(
        correlation_.rows() == correlation_.columns(),
        "Analytics/correlation number of rows ",
        correlation_.rows(),
        " is differents from correlation number of columns ",
        correlation_.columns());

    COBRA_CHECK(
        volatilities_.columns() == decays_.columns(),
        "volatilities number of factors ",
        volatilities_.columns(),
        " is differents from decays number of factors ",
        decays_.columns());
}

//-----------------------------------------------------------------------------
void parameter_markovian_hjm::decays_aad(
    const vector<double>& value_aad, size_t idx, double* state_parameters_aad) const
{
    size_t m = number_of_factors();
    for (size_t j = 0; j < m; ++j)
        update_state_parameters(
            value_aad[j], AAD_OFFSET(decays_), idx * m + j, state_parameters_aad);
};

//-----------------------------------------------------------------------------
void parameter_markovian_hjm::volatility_aad(
    const vector<double>& value_aad, size_t idx, double* state_parameters_aad) const
{
    size_t m = number_of_factors();
    for (size_t j = 0; j < m; ++j)
        update_state_parameters(
            value_aad[j], AAD_OFFSET(volatilities_), idx * m + j, state_parameters_aad);
};

//-----------------------------------------------------------------------------
void parameter_markovian_hjm::correlation_aad(
    const matrix<double>& value_aad, double* state_parameters_aad) const
{
    size_t m = number_of_factors();

    size_t offset = 0;
    for (size_t i = 0; i < m; ++i)
        for (size_t j = 0; j < i; ++j, offset++)
            update_state_parameters(
                value_aad[i][j] + value_aad[j][i],
                AAD_OFFSET(correlation_),
                offset,
                state_parameters_aad);
};

//-----------------------------------------------------------------------------
void parameter_markovian_hjm::finalize_aad(double* /*parameters*/) const {};

//-----------------------------------------------------------------------------
size_t parameter_markovian_hjm::get_state_parameters(double* parameters) const
{
    size_t offset = 0;
    for (auto t : decays_)
    {
        parameters[offset++] = t;
    }
    for (auto t : volatilities_)
    {
        parameters[offset++] = t;
    }

    for (size_t i = 0; i < number_of_factors(); ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            parameters[offset++] = correlation_.at(i, j);
        }
    }

    COBRA_CHECK(
        offset == state_parameters_size(),
        "expect size parameter ",
        state_parameters_size(),
        " while provided ",
        offset);

    return offset;
}

//-----------------------------------------------------------------------------
size_t parameter_markovian_hjm::set_state_parameters(const double* parameters)
{
    size_t offset = 0;
    for (auto& t : decays_)
    {
        t = parameters[offset++];
    }
    for (auto& t : volatilities_)
    {
        t = parameters[offset++];
    }

    for (size_t i = 0; i < number_of_factors(); ++i)
    {
        for (size_t j = 0; j < i; ++j)
        {
            correlation_.at(j, i) = correlation_.at(i, j) = parameters[offset++];
        }
        correlation_.at(i, i) = 1.;
    }

    COBRA_CHECK(
        offset == state_parameters_size(),
        "expect size parameter ",
        state_parameters_size(),
        " while provided ",
        offset);

    return offset;
}

//-----------------------------------------------------------------------------
COBRA_SERIALIZATION_METHODES(parameter_markovian_hjm);
}  // namespace cobra
