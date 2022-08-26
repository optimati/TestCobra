

//-----------------------------------------------------------------------------
double calibration_ir_hjm_swaption::price(
    double                strike,
    const matrix<double>& integral_decays,
    const vector<double>& roots,
    const vector<double>& dpv_droots) const
{
    const auto& market_dfs = calibration_swaption_->dfs();

    if (calibration_swaption_->is_caplet())
    {
        const auto pay_index   = calibration_swaption_->fixed_pay_index()[0];
        const auto dcf         = calibration_swaption_->fixed_leg_instrument()->dcf(0);
        const auto start_index = calibration_swaption_->float_start_index()[0];
        const auto ratio       = calibration_swaption_->float_leg_instrument()->dcf(0);

        const auto vol = sqrt(
            accumulate(integral_decays[1] * integral_decays[1]) /
            calibration_swaption_->expiry_double());

        const auto K = ratio * market_dfs[start_index];

        const auto F = (ratio + dcf * strike) * market_dfs[pay_index];

        return black_scholes::price(F, K, calibration_swaption_->expiry_double(), vol, 1., -1.);
    }

    const auto sign = accumulate(roots * dpv_droots) <= 0 ? -1. : 1.;

    const auto norm = sqrt(accumulate(roots * roots));

    if (cobra::is_almost_zero(norm))
        return 0.;

    auto q = sign / norm * roots;

    double output = 0.;
    {
        const auto& pay_indexs = calibration_swaption_->fixed_pay_index();

        for (size_t i = 0, size = pay_indexs.size(); i < size; ++i)
        {
            const auto index = pay_indexs[i];

            auto w = N(-accumulate((integral_decays[index] + roots) * q));

            output += calibration_swaption_->fixed_leg_instrument()->dcf(i) * market_dfs[index] * w;
        }
    }

    output *= -strike;

    {
        const auto& start_indexs = calibration_swaption_->float_start_index();
        const auto& end_indexs   = calibration_swaption_->float_end_index();
        const auto& pay_indexs   = calibration_swaption_->float_pay_index();

        for (size_t i = 0, size = pay_indexs.size(); i < size; ++i)
        {
            const auto& start = start_indexs[i];
            const auto& end   = end_indexs[i];
            const auto& pay   = pay_indexs[i];

            const auto pay_shift = accumulate((integral_decays[pay] + roots) * q);

            if (pay != end)
            {
                const auto dcf =
                    calibration_swaption_->float_leg_instrument()->dcf(i) * market_dfs[pay];

                const auto libor_shift = accumulate(
                    (integral_decays[start] + integral_decays[pay] - integral_decays[end] + roots) *
                    q);

                const auto sum_shift = accumulate(
                    (integral_decays[start] + integral_decays[pay] - integral_decays[end]) * q);

                const auto start_libor_shift = accumulate(integral_decays[start] * q);
                const auto pay_libor_shift   = accumulate(integral_decays[pay] * q);
                const auto end_libor_shift   = accumulate(integral_decays[end] * q);

                auto drift_shift =
                    0.5 * (sum_shift * sum_shift - start_libor_shift * start_libor_shift -
                           pay_libor_shift * pay_libor_shift + end_libor_shift * end_libor_shift);

                drift_shift = std::clamp(drift_shift, min_exponent, max_exponent);

                const auto alpha = exp(drift_shift) * N(-libor_shift);
                const auto beta  = N(-pay_shift);

                output += dcf * (market_dfs[start] / market_dfs[end] * alpha - beta);
            }
            else
            {
                const auto dcf = calibration_swaption_->float_leg_instrument()->dcf(i);

                const auto libor_shift = accumulate((integral_decays[start] + roots) * q);

                output +=
                    dcf * (market_dfs[start] * N(-libor_shift) - market_dfs[pay] * N(-pay_shift));
            }
        }
    }

    return output;
}

//-----------------------------------------------------------------------------
void calibration_ir_hjm_swaption::price_aad(
    double                ret_aad,
    double                strike,
    const matrix<double>& integral_decays,
    const vector<double>& roots,
    const vector<double>& dpv_droots,
    matrix<double>&       integral_decays_aad) const
{
    const auto& market_dfs = calibration_swaption_->dfs();

    if (calibration_swaption_->is_caplet())
    {
        const auto pay_index   = calibration_swaption_->fixed_pay_index()[0];
        const auto dcf         = calibration_swaption_->fixed_leg_instrument()->dcf(0);
        const auto start_index = calibration_swaption_->float_start_index()[0];
        const auto ratio       = calibration_swaption_->float_leg_instrument()->dcf(0);
        const auto sqrt_expiry = sqrt(calibration_swaption_->expiry_double());

        const auto vol = sqrt(accumulate(integral_decays[1] * integral_decays[1]));

        const auto K = ratio * market_dfs[start_index];

        const auto F = (ratio + dcf * strike) * market_dfs[pay_index];

        //
        const auto vol_aad =
            ret_aad / sqrt_expiry *
            black_scholes::vega(F, K, calibration_swaption_->expiry_double(), vol / sqrt_expiry);

        integral_decays_aad[1] += vol_aad * integral_decays[1] / vol;

        return;
    }

    const double   sign = accumulate(roots * dpv_droots) <= 0. ? -1. : 1.;
    const double   norm = sqrt(accumulate(roots * roots));
    vector<double> q    = (sign / norm) * roots;

    {
        const auto& start_indexs = calibration_swaption_->float_start_index();
        const auto& end_indexs   = calibration_swaption_->float_end_index();
        const auto& pay_indexs   = calibration_swaption_->float_pay_index();

        for (int i = static_cast<int>(pay_indexs.size()) - 1; i >= 0; --i)
        {
            const auto& start = start_indexs[i];
            const auto& end   = end_indexs[i];
            const auto& pay   = pay_indexs[i];

            const double pay_shift     = accumulate((integral_decays[pay] + roots) * q);
            double       pay_shift_aad = 0.;
            if (pay != end)
            {
                const double dcf =
                    calibration_swaption_->float_leg_instrument()->dcf(i) * market_dfs[pay];

                const double start_libor_shift = accumulate(integral_decays[start] * q);
                const double pay_libor_shift   = accumulate(integral_decays[pay] * q);
                const double end_libor_shift   = accumulate(integral_decays[end] * q);
                const double sum_shift   = start_libor_shift + pay_libor_shift - end_libor_shift;
                const double libor_shift = sum_shift + sign * norm;

                /* const double sum_shift = accumulate(
                    (integral_decays[start] + integral_decays[pay] - integral_decays[end]) * q);
                    const double libor_shift = accumulate(
                      (integral_decays[start] + integral_decays[pay] - integral_decays[end] + roots)
                   * q);*/

                double drift_shift =
                    0.5 * (sum_shift * sum_shift - start_libor_shift * start_libor_shift -
                           pay_libor_shift * pay_libor_shift + end_libor_shift * end_libor_shift);

                drift_shift = std::clamp(drift_shift, min_exponent, max_exponent);

                const double exp_drift_shift = exp(drift_shift);
                const double alpha           = exp_drift_shift * N(-libor_shift);

                // AAD: output += dcf * (market_dfs[start] / market_dfs[end] * alpha - beta);
                double beta_aad  = -ret_aad * dcf;
                double alpha_aad = -beta_aad * market_dfs[start] / market_dfs[end];

                // AAD: const auto alpha = exp(drift_shift) * N(-libor_shift);
                double drift_shift_aad = (drift_shift > min_exponent && drift_shift < max_exponent)
                                             ? alpha_aad * alpha
                                             : 0.;
                double libor_shift_aad = -alpha_aad * exp_drift_shift * N_aad(-libor_shift);

                // AAD: const auto beta = N(-pay_shift);
                pay_shift_aad -= beta_aad * N_aad(-pay_shift);

                double sum_shift_aad         = 0.;
                double start_libor_shift_aad = 0.;
                double pay_libor_shift_aad   = 0.;
                double end_libor_shift_aad   = 0.;
                // AAD: auto drift_shift =0.5 * (sum_shift * sum_shift - start_libor_shift *
                // start_libor_shift -pay_libor_shift * pay_libor_shift + end_libor_shift *
                // end_libor_shift);
                if (is_almost_zero(drift_shift_aad))
                {
                    sum_shift_aad += drift_shift_aad * sum_shift;
                    start_libor_shift_aad -= drift_shift_aad * start_libor_shift;
                    pay_libor_shift_aad -= drift_shift_aad * pay_libor_shift;
                    end_libor_shift_aad += drift_shift_aad * end_libor_shift;
                }

                const double sum_libor_shift_aad = sum_shift_aad + libor_shift_aad;

                double value_aad = (start_libor_shift_aad + sum_libor_shift_aad);
                {
                    auto z = integral_decays_aad[start];
                    z      = fma(value_aad, q, z);
                }

                value_aad = (pay_libor_shift_aad + sum_libor_shift_aad);
                {
                    auto z = integral_decays_aad[pay];
                    z      = fma(value_aad, q, z);
                }

                value_aad = (end_libor_shift_aad - sum_libor_shift_aad);
                {
                    auto z = integral_decays_aad[end];
                    z      = fma(value_aad, q, z);
                }
            }
            else
            {
                const auto dcf = calibration_swaption_->float_leg_instrument()->dcf(i);

                const auto libor_shift = accumulate((integral_decays[start] + roots) * q);

                // AAD: output +=dcf * (market_dfs[start] * N(-libor_shift) - market_dfs[pay] *
                // N(-pay_shift));
                auto value_aad       = ret_aad * dcf;
                auto libor_shift_aad = -value_aad * market_dfs[start] * N_aad(-libor_shift);
                pay_shift_aad        = value_aad * market_dfs[pay] * N_aad(-pay_shift);

                // AAD: libor_shift = accumulate((integral_decays[start] + roots) *
                // q);{
                {
                    auto z = integral_decays_aad[start];
                    z      = fma(libor_shift_aad, q, z);
                }
            }
            // AAD: pay_shift = accumulate((integral_decays[pay] + roots) * q);
            {
                auto z = integral_decays_aad[pay];
                z      = fma(pay_shift_aad, q, z);
            }
        }
    }
    // AAD: output *= -strike;
    ret_aad *= -strike;

    {
        const auto& pay_indexs = calibration_swaption_->fixed_pay_index();

        for (int i = static_cast<int>(pay_indexs.size()) - 1; i >= 0; --i)
        {
            const auto index = pay_indexs[i];

            // output += calibration_swaption_->fixed_leg_instrument().dcf(i) * market_dfs[index] *
            // w;
            double log_w_aad = -ret_aad * calibration_swaption_->fixed_leg_instrument()->dcf(i) *
                               market_dfs[index] *
                               N_aad(-accumulate((integral_decays[index] + roots) * q));

            // AAD: log_w = accumulate((integral_decays[index] + roots) * q);
            integral_decays_aad[index] += log_w_aad * q;
        }
    }
}

