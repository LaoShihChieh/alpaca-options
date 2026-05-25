"""
utils/black_scholes.py — Closed-form Black-Scholes option pricing and Greeks.

Implements:
    - norm_cdf / norm_ppf  (no scipy dependency; uses math.erf + Newton-Raphson)
    - bs_price             — call or put fair value
    - bs_delta             — first-order spot sensitivity
    - strike_for_delta     — closed-form strike given a target delta

All inputs/outputs are in per-share, annualised-year units:
    S      — current underlying price  (e.g. 580.0 for SPY)
    K      — strike price
    T      — time to expiry in years   (e.g. 0.5 / 252 for a half-day 0DTE)
    r      — risk-free rate            (e.g. 0.05 for 5%)
    sigma  — implied volatility        (e.g. 0.20 for 20% annualised)
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Normal distribution helpers (no external dependency)
# ---------------------------------------------------------------------------

def norm_cdf(x: float) -> float:
    """CDF of the standard normal distribution N(0,1)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _erfinv(y: float) -> float:
    """Inverse error function via Newton-Raphson (converges to machine epsilon)."""
    if y == 0.0:
        return 0.0
    if abs(y) >= 1.0:
        raise ValueError(f"erfinv domain error: y={y!r}, must satisfy |y| < 1")

    # Halley-method initial guess (rational approximation)
    a = 0.147
    sign = 1.0 if y > 0 else -1.0
    ya = abs(y)
    ln_term = math.log(1.0 - ya * ya)
    t1 = 2.0 / (math.pi * a) + ln_term / 2.0
    guess = sign * math.sqrt(math.sqrt(t1 * t1 - ln_term / a) - t1)

    # Newton-Raphson refinement: solve erf(x) = y
    sqrt_pi_inv = 2.0 / math.sqrt(math.pi)
    for _ in range(5):
        fx = math.erf(guess) - y
        fpx = sqrt_pi_inv * math.exp(-(guess * guess))
        if abs(fpx) < 1e-300:
            break
        guess -= fx / fpx

    return guess


def norm_ppf(p: float) -> float:
    """Inverse CDF (percent-point function) of the standard normal distribution.

    Parameters
    ----------
    p:
        Probability in the open interval (0, 1).

    Returns
    -------
    float
        z such that N(z) = p.

    Raises
    ------
    ValueError
        If *p* is outside (0, 1).
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"norm_ppf: p must be in (0, 1), got {p!r}")
    return math.sqrt(2.0) * _erfinv(2.0 * p - 1.0)


# ---------------------------------------------------------------------------
# Black-Scholes core
# ---------------------------------------------------------------------------

def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    """Compute d1 and d2 for the Black-Scholes formula."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes theoretical option price (per share).

    Parameters
    ----------
    S:
        Underlying spot price.
    K:
        Option strike price.
    T:
        Time to expiry in years (≥ 0).
    r:
        Continuously-compounded risk-free rate.
    sigma:
        Annualised implied volatility (> 0).
    is_call:
        ``True`` for a call, ``False`` for a put.

    Returns
    -------
    float
        Theoretical option price ≥ 0.
    """
    if T <= 0.0:
        if is_call:
            return max(S - K, 0.0)
        return max(K - S, 0.0)

    d1, d2 = _d1_d2(S, K, T, r, sigma)
    disc = math.exp(-r * T)

    if is_call:
        return S * norm_cdf(d1) - K * disc * norm_cdf(d2)
    return K * disc * norm_cdf(-d2) - S * norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, sigma: float, is_call: bool) -> float:
    """Black-Scholes delta (∂V/∂S).

    Returns a value in [0, 1] for calls, [-1, 0] for puts.
    """
    if T <= 0.0:
        if is_call:
            return 1.0 if S > K else (0.5 if S == K else 0.0)
        return -1.0 if S < K else (-0.5 if S == K else 0.0)

    d1, _ = _d1_d2(S, K, T, r, sigma)
    if is_call:
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0


def strike_for_delta(
    S: float,
    T: float,
    r: float,
    sigma: float,
    delta: float,
    is_call: bool,
) -> float:
    """Return the strike K such that BS delta equals *delta* (closed-form).

    Parameters
    ----------
    S:
        Underlying spot price.
    T:
        Time to expiry in years.
    r:
        Risk-free rate.
    sigma:
        Annualised implied volatility.
    delta:
        Target delta.  Calls use (0, 1), puts use (-1, 0).
    is_call:
        ``True`` for a call, ``False`` for a put.

    Returns
    -------
    float
        Strike K such that bs_delta(S, K, T, r, sigma, is_call) ≈ delta.

    Raises
    ------
    ValueError
        If *delta* is outside the valid range for the option type.
    """
    if T <= 0.0:
        return S  # degenerate case

    if is_call:
        if not (0.0 < delta < 1.0):
            raise ValueError(f"Call delta must be in (0, 1), got {delta}")
        d1 = norm_ppf(delta)
    else:
        if not (-1.0 < delta < 0.0):
            raise ValueError(f"Put delta must be in (-1, 0), got {delta}")
        d1 = norm_ppf(delta + 1.0)

    # d1 = (ln(S/K) + (r + σ²/2)T) / (σ√T)
    # → ln(S/K) = d1 * σ√T − (r + σ²/2)T
    # → K = S * exp(−(d1 * σ√T − (r + σ²/2)T))
    sqrt_T = math.sqrt(T)
    log_ratio = d1 * sigma * sqrt_T - (r + 0.5 * sigma * sigma) * T
    return S * math.exp(-log_ratio)
