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