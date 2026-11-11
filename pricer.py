import numpy as np
from scipy.stats import norm

def d1_d2(S, K, r, q, sigma, T):
    if T <= 0 or sigma <= 0:
        raise ValueError(f"Invalid input T or sigma <= 0")
    d1 = (np.log(S/K)+(r-q+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return (d1, d2)

def bsm_price(S ,K, r, q, sigma, T, kind):
    d1 , d2 = d1_d2(S ,K, r, q, sigma, T)
    if kind == "call":
        return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    elif kind == "put":
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*np.exp(-q*T)*norm.cdf(-d1)
    else:
        raise ValueError(f"Invalid option kind: {kind}")

def parity_gap(S, K, r, q, sigma, T):
    C = bsm_price(S, K, r, q, sigma, T, "call")
    P = bsm_price(S, K, r, q, sigma, T, "put")
    return (C-P)-(S*np.exp(-q*T)-K*np.exp(-r*T))

def bsm_greeks(S, K, r, q, sigma, T, kind):
    d1 , d2 = d1_d2(S ,K, r, q, sigma, T)
    if kind == "call":
        delta = np.exp(-q*T)*norm.cdf(d1)
        gamma = (np.exp(-q*T)*norm.pdf(d1))/(S*sigma*np.sqrt(T))
        vega = S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
        theta = -((S*np.exp(-q*T)*norm.pdf(d1)*sigma)/(2*np.sqrt(T)))-r*K*np.exp(-r*T)*norm.cdf(d2)+q*S*np.exp(-q*T)*norm.cdf(d1)
        rho = K*T*np.exp(-r*T)*norm.cdf(d2) 
    elif kind == "put":
        delta = np.exp(-q*T)*(norm.cdf(d1)-1)
        gamma = (np.exp(-q*T)*norm.pdf(d1))/(S*sigma*np.sqrt(T))
        vega = S*np.exp(-q*T)*norm.pdf(d1)*np.sqrt(T)
        theta = -((S*np.exp(-q*T)*norm.pdf(d1)*sigma)/(2*np.sqrt(T)))+r*K*np.exp(-r*T)*norm.cdf(-d2)-q*S*np.exp(-q*T)*norm.cdf(-d1)
        rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
    else:
         raise ValueError(f"Invalid option kind: {kind}")
        
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }

def fd_greeks(S, K, r, q, sigma, T, kind):
    hS = 0.01
    hSigma = hR = hT = 0.0001
    delta = (bsm_price(S+hS, K, r, q, sigma, T, kind)-bsm_price(S-hS, K, r, q, sigma, T, kind))/(2*hS)
    gamma = ((bsm_price(S+hS, K, r, q, sigma, T, kind))-2*(bsm_price(S, K, r, q, sigma, T, kind))+(bsm_price(S-hS, K, r, q, sigma, T, kind)))/(hS**2)
    vega = (bsm_price(S, K, r, q, sigma+hSigma, T, kind)-bsm_price(S, K, r, q, sigma-hSigma, T, kind))/(2*hSigma)
    theta = (bsm_price(S, K, r, q, sigma, T-hT, kind)-bsm_price(S, K, r, q, sigma, T+hT, kind))/(2*hT)
    rho = (bsm_price(S, K, r+hR, q, sigma, T, kind)-bsm_price(S, K, r-hR, q, sigma, T, kind))/(2*hR)
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "rho": rho,
    }

def price_gap(sigma, market_price, S, K, r, q, T, kind):
    return bsm_price(S, K, r, q, sigma, T, kind) - market_price

def iv_bisection(market_price, S, K, r, q, T, kind, lo=1e-4, hi=5.0, tol=1e-8, max_iter=200):
    iterations = 0
    middle = (hi+lo)/2
    while hi-lo > tol and iterations < max_iter:
        middle = (hi+lo)/2
        iterations += 1
        result = price_gap(middle, market_price, S, K, r, q, T, kind)
        if result > 0:
            hi = middle
        else:
            lo = middle
    if (hi-lo) > tol:
        return np.nan, iterations
    else:
        return(middle, iterations)

def iv_newton(market_price, S, K, r, q, T, kind, sigma0=0.2, tol=1e-8, max_iter=50):
    iterations = 0
    for i in range(max_iter):
        iterations += 1
        gap = price_gap(sigma0, market_price, S, K, r, q, T, kind)
        vega = bsm_greeks(S, K, r, q, sigma0, T, kind)["vega"]
        if vega < 1e-8:
            return np.nan, iterations
        else:
            sigma0 = sigma0 - (gap/vega)
        if sigma0 <= 0 or sigma0 > 5:
            return np.nan, iterations
        if abs(gap/vega) < tol:
            return(sigma0, iterations)
    return np.nan, iterations

def newton_path(market_price, S, K, r, q, T, kind, sigma0, n_steps):
    sigma_steps = [sigma0]
    for i in range(n_steps):
        gap = price_gap(sigma0, market_price, S, K, r, q, T, kind)
        vega = bsm_greeks(S, K, r, q, sigma0, T, kind)["vega"]
        if vega < 1e-8:
            return np.array(sigma_steps)
        sigma0 = sigma0 - (gap/vega)
        sigma_steps.append(sigma0)
        if sigma0<=0 or sigma0>5:
            return np.array(sigma_steps)
    return np.array(sigma_steps)

def implied_vol(market_price, S, K, r, q, T, kind):
    if kind == "call":
        lower = max(S * np.exp(-q * T)  - K * np.exp(-r * T), 0)
        upper = S * np.exp(-q * T)
    elif kind == "put":
        lower = max(K * np.exp(-r * T) - S * np.exp(-q * T), 0)
        upper = K * np.exp(-r * T)
    else:
        raise ValueError(f"Invalid option kind: {kind}")

    if market_price <= lower or market_price >= upper:
        return np.nan
    else:
        implied_sigma = iv_newton(market_price, S, K, r, q, T, kind, sigma0=0.2, tol=1e-8, max_iter=50)[0]
        if np.isnan(implied_sigma):
            implied_sigma = iv_bisection(market_price, S, K, r, q, T, kind, lo=1e-4, hi=5.0, tol=1e-8, max_iter=200)[0]
        return implied_sigma

obs_times = np.array([0.5, 1.0, 1.5, 2.0])
coupon, autocall_b, coupon_b, protect_b = 4.0, 1.00, 0.70, 0.60

def simulate_paths(S0, r, q, sigma, T, n_steps, n_paths, seed):
    rng = np.random.default_rng(seed)
    delta_T = T/n_steps
    simulation = rng.normal(0, 1,(n_paths, n_steps))
    price_T = S0*np.exp(np.cumsum((r-q-0.5*(sigma**2))*delta_T+sigma*np.sqrt(delta_T)*simulation, axis=1))
    paths = np.full((n_paths, n_steps + 1), S0, float)
    paths[:,1:] = price_T
    return paths

def mc_call(paths, K, r, T):
    call_payoff = np.exp(-r*T)*np.maximum(paths[:,-1]-K, 0)
    price = np.mean(call_payoff)
    std_error = np.std(call_payoff)/np.sqrt(len(paths))
    return (price, std_error)

def phoenix_pv(perf, coupon, autocall_b, coupon_b, protect_b, r, obs_times, notional=100):
    missed=0
    pay=0

    for i in range(len(obs_times)):
        if perf[i] >= coupon_b:
            pay += coupon*(missed+1)*np.exp(-r*(obs_times[i]))
            missed=0
        else:
            missed+=1
        
        if i < len(obs_times) - 1 and perf[i] >= autocall_b:
            pay += notional*np.exp(-r*(obs_times[i]))
            return pay
        
    if perf[-1]>=protect_b:
        pay += notional*np.exp(-r*(obs_times[i]))
        return pay
    else:
        pay += (notional*perf[i])*np.exp(-r*(obs_times[i]))
        return pay

def price_phoenix(S0, r, q, sigma, n_paths, seed):

    paths = simulate_paths(S0, r, q, sigma, obs_times[-1], len(obs_times), n_paths, seed)
    paths_pv = []
    for i in range(n_paths):
        paths_pv.append(phoenix_pv(paths[i, 1:]/S0, coupon, autocall_b, coupon_b, protect_b, r, obs_times,notional=100) ) 

    price = np.mean(paths_pv)
    std_error = np.std(paths_pv) / np.sqrt(len(paths_pv))
    return (price, std_error)
