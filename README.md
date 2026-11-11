# Option pricer (Python)

A small Python library that prices European options with Black-Scholes-Merton, computes their Greeks, backs out implied volatility from market prices, and prices a Phoenix autocallable by Monte Carlo.

I built it to understand how a derivatives desk prices and risk-manages vanilla options, and I checked every result against an independent method before trusting it.

## What it does today

| Function | Purpose |
|---|---|
| `d1_d2` | d1 and d2 terms of Black-Scholes-Merton with a continuous dividend yield `q`; raises `ValueError` if `T <= 0` or `sigma <= 0` |
| `bsm_price` | European call or put price |
| `parity_gap` | put-call parity residual, `(C - P) - (S e^{-qT} - K e^{-rT})` |
| `bsm_greeks` | analytic delta, gamma, vega, theta, rho |
| `fd_greeks` | the same five Greeks by central finite differences (bump-and-reprice) |
| `price_gap` | model price minus market price, for a given volatility |
| `iv_bisection` | implied volatility by bisection; returns `nan` if it has not converged after `max_iter` steps |
| `iv_newton` | implied volatility by Newton-Raphson; returns `nan` when vega is too small or the volatility leaves (0, 5] |
| `newton_path` | every Newton step from a starting guess, to plot how the solver converges (or where it breaks down) |
| `implied_vol` | checks the no-arbitrage price bounds, tries Newton first, falls back to bisection |
| `simulate_paths` | geometric Brownian motion paths under the risk-neutral measure (drift r - q), one column per date |
| `mc_call` | Monte Carlo price and standard error of a European call, from simulated paths |
| `phoenix_pv` | discounted payoff of a Phoenix autocallable on one path of performances (memory coupon, autocall, protection barrier) |
| `price_phoenix` | Monte Carlo price and standard error of the Phoenix below |

The Phoenix priced by `price_phoenix`: 2 years, observed every 6 months, notional 100. At each date it pays a coupon of 4 if the underlying is at or above 70% of its start level, plus any coupons missed earlier (memory). It redeems early at 100 if the underlying is at or above 100% on a date before maturity. At maturity the investor gets 100 back if the underlying is at or above 60% (European protection barrier), otherwise 100 times the final performance.

Conventions: rates and dividend yield are continuously compounded, `T` is in years, vega and rho are per 1.00 change (not per 1%), theta is per year. An option kind other than `"call"` or `"put"` raises `ValueError`.

## How I validated it

Each check compares my code with something independent of it:

1. **Textbook values.** Call and put prices match Hull's worked example (S = 42, K = 40, r = 10%, sigma = 20%, T = 0.5): 4.7594 and 0.8086.
2. **Put-call parity.** The parity residual is below 1e-14 on three parameter sets.
3. **Analytic vs finite-difference Greeks.** On the base case (S = 100, K = 105, r = 4%, q = 2%, sigma = 25%, T = 0.75) the five analytic Greeks agree with bump-and-reprice within 2e-7, for both calls and puts.
4. **Implied-vol round trip.** Price an option with a known volatility, recover the volatility with `implied_vol`, compare: 90 cases (5 strikes x 3 maturities x 3 volatilities x call/put), worst error below 1e-13.
5. **Low-vega strike.** For a far out-of-the-money call (K = 200, T = 0.25) vega at the 20% starting guess is about 1.5e-9. Newton alone gives up, the bisection fallback recovers the true 60%.
6. **Price bounds.** Prices on or outside the no-arbitrage bounds return `nan` instead of a meaningless volatility.
7. **Invalid inputs.** `T <= 0`, `sigma <= 0` or an unknown option kind raise `ValueError` instead of dividing by zero.
8. **Simulated paths.** With 200,000 paths the average terminal price matches the forward S e^{(r-q)T} within 0.5%, and the standard deviation of the 6-month log-returns matches sigma sqrt(0.5).
9. **Monte Carlo vs closed form.** A European call priced by `mc_call` (S = 100, K = 105, r = 3%, q = 2%, sigma = 25%, T = 1, 200,000 paths) gives 8.120 +/- 0.034, against 8.124 from `bsm_price`: within one standard error.
10. **Phoenix payoff by hand.** `phoenix_pv` reproduces four hand-computed paths to 1e-6: autocall on the first date, memory coupons then autocall, loss below the protection barrier, and three coupons paid at once by the memory.
11. **Phoenix price and vega sign.** With r = 3%, q = 2%, sigma = 25% and 200,000 paths the note is worth 100.72 +/- 0.03 per 100 of notional. At 15%, 25% and 35% volatility it is worth 104.75, 100.72 and 95.32: the price falls as volatility rises, because the investor is short a down-and-in put through the protection barrier.

## What I learned

- **Bump size matters.** For gamma, a bump of 1 is too coarse (truncation error) and a bump of 1e-6 is too small (rounding error). The error was smallest around 0.01.
- **Newton is fast but fragile.** On the base case it converged in 3 steps where bisection needed 29, but when vega is close to zero its step (price gap divided by vega) explodes: on the K = 200 call the first step jumps from 20% to about 10^8. That is why `implied_vol` falls back to bisection, which only needs the sign of the price gap.
- **Monte Carlo needs an error bar.** A Monte Carlo price on its own means nothing: comparing it with Black-Scholes only makes sense in units of its standard error, which shrinks like 1 / sqrt(number of paths).
- **Check the bounds first.** A call priced at or below max(S e^{-qT} - K e^{-rT}, 0), or at or above S e^{-qT}, has no implied volatility under Black-Scholes-Merton, so the solver returns `nan` rather than a number.

## Limitations

- European exercise only, constant volatility, rates and dividend yield.
- The Phoenix terms (dates, coupon, barriers) are fixed at the top of `pricer.py`, and `price_phoenix` assumes evenly spaced observation dates.
- `T` and `sigma` must be strictly positive: the formulas divide by `sigma * sqrt(T)`, so the pricer refuses expiry-day and zero-volatility inputs rather than guessing.

## Roadmap

- **Automated test file and demo notebook** with charts (gamma error vs bump size, volatility smile, Newton convergence). Next.
- **CRR binomial tree** with American early exercise. Planned.

## Run it

Requires Python 3.9 or later.

```bash
pip install -r requirements.txt
```

```python
from pricer import bsm_price, bsm_greeks, implied_vol

price = bsm_price(42, 40, 0.10, 0.0, 0.20, 0.5, "call")       # 4.7594 (Hull)
greeks = bsm_greeks(100, 105, 0.04, 0.02, 0.25, 0.75, "call")
vol = implied_vol(price, 42, 40, 0.10, 0.0, 0.5, "call")      # 0.20

from pricer import price_phoenix
note, std_error = price_phoenix(100, 0.03, 0.02, 0.25, 200_000, seed=11)   # about 100.72
```

Argument order everywhere: `S, K, r, q, sigma, T, kind`.
