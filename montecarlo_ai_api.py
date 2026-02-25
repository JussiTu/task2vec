import random, math
import numpy as np

def simulate_customer(
    months=12,
    # pricing (you set these)
    P_sub=99.0, P_var=0.00, take_rate=0.00,
    # base usage
    N_mean=800, N_sd=250,
    # token distributions per request
    t_in_mean=1200, t_in_sd=400,
    t_out_mean=600, t_out_sd=250,
    # token prices ($ per 1K tokens)
    p_in=0.003, p_out=0.006,
    # other per-request costs
    c_cloud=0.0015, c_vec=0.0008, c_obs=0.0003, c_hil=0.0,
    # per-customer fixed ops/month
    C_fixed_ops=6.0,
    # churn
    churn=0.04,
    # platform risk events (per month)
    api_price_hike_prob=0.02, api_price_hike_mult=1.25,
    replication_prob=0.01, volume_shock_mult=0.75,
    # discount rate per month
    disc=0.0,
    seed=None
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    alive = True
    token_mult = 1.0
    vol_mult = 1.0

    cashflows = []
    for m in range(months):
        if not alive:
            cashflows.append(0.0)
            continue

        # events
        if random.random() < api_price_hike_prob:
            token_mult *= api_price_hike_mult
        if random.random() < replication_prob:
            vol_mult *= volume_shock_mult

        # sample usage & tokens (truncate at >=0)
        N = max(0, int(np.random.normal(N_mean, N_sd) * vol_mult))
        t_in = max(0, np.random.normal(t_in_mean, t_in_sd))
        t_out = max(0, np.random.normal(t_out_mean, t_out_sd))

        C_req = (t_in/1000.0)*(p_in*token_mult) + (t_out/1000.0)*(p_out*token_mult) \
                + c_cloud + c_vec + c_obs + c_hil

        R_m = (P_sub + N*P_var) * (1.0 - take_rate)
        C_var_m = N * C_req
        CM_m = R_m - C_var_m - C_fixed_ops

        # discount
        CM_m_disc = CM_m / ((1.0 + disc) ** m)
        cashflows.append(CM_m_disc)

        # churn at end of month
        if random.random() < churn:
            alive = False

    LTV = sum(cashflows)
    return {
        "LTV": LTV,
        "avg_monthly_CM": sum(cashflows)/months,
        "min_monthly_CM": min(cashflows),
        "negative_month_prob": sum(1 for x in cashflows if x < 0)/months
    }

def monte_carlo(n=5000, **kwargs):
    results = [simulate_customer(**kwargs)["LTV"] for _ in range(n)]
    arr = np.array(results)
    return {
        "LTV_mean": float(arr.mean()),
        "LTV_p10": float(np.percentile(arr, 10)),
        "LTV_p50": float(np.percentile(arr, 50)),
        "LTV_p90": float(np.percentile(arr, 90)),
        "prob_LTV_negative": float((arr < 0).mean())
    }

# Example run (edit inputs to match your target service)
summary = monte_carlo(
    n=5000,
    months=24,
    P_sub=99.0,
    N_mean=900, N_sd=300,
    t_in_mean=1400, t_in_sd=500,
    t_out_mean=700, t_out_sd=300,
    p_in=0.003, p_out=0.006,
    c_cloud=0.0018, c_vec=0.0010, c_obs=0.0004,
    C_fixed_ops=8.0,
    churn=0.04,
    api_price_hike_prob=0.02, api_price_hike_mult=1.30,
    replication_prob=0.01, volume_shock_mult=0.75,
    take_rate=0.0
)
print(summary)
