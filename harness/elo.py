"""Elo ratings by maximum likelihood (Bradley-Terry with draws counted as half a win), with
bootstrap confidence intervals over games."""
import numpy as np
from scipy.optimize import minimize

K = np.log(10) / 400.0


def fit_elo(games, names, anchor=None, anchor_value=0.0, prior_sd=800.0):
    """games: list of (i, j, score_i) with indices into names. Returns ratings (np.array, Elo)."""
    n = len(names)
    gi = np.array([g[0] for g in games])
    gj = np.array([g[1] for g in games])
    gs = np.array([g[2] for g in games], dtype=float)

    def f(r):
        d = K * (r[gi] - r[gj])
        # negative log-likelihood of score s with p = sigmoid(d): -(s log p + (1-s) log(1-p))
        nll = np.sum(np.logaddexp(0, -d) * gs + np.logaddexp(0, d) * (1 - gs))
        nll += 0.5 * np.sum((r / prior_sd) ** 2)
        p = 1 / (1 + np.exp(-d))
        g = np.zeros(n)
        coef = K * (p - gs)
        np.add.at(g, gi, coef)
        np.add.at(g, gj, -coef)
        g += r / prior_sd ** 2
        return nll, g

    res = minimize(f, np.zeros(n), jac=True, method="L-BFGS-B")
    r = res.x
    if anchor is not None:
        r = r - r[anchor] + anchor_value
    else:
        r = r - r.mean()
    return r


def bootstrap_elo(games, names, anchor=None, anchor_value=0.0, n_boot=200, seed=0):
    rng = np.random.default_rng(seed)
    base = fit_elo(games, names, anchor, anchor_value)
    arr = np.array(games, dtype=float)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), len(arr))
        samp = [(int(a), int(b), float(c)) for a, b, c in arr[idx]]
        boots.append(fit_elo(samp, names, anchor, anchor_value))
    boots = np.array(boots)
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    return base, lo, hi


def score_ci(wins, draws, losses, n_boot=2000, seed=0):
    """Mean score and 95% bootstrap CI for a match."""
    s = np.array([1.0] * wins + [0.5] * draws + [0.0] * losses)
    rng = np.random.default_rng(seed)
    m = [rng.choice(s, len(s)).mean() for _ in range(n_boot)]
    return s.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
