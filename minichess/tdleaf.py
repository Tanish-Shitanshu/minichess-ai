"""TDLeaf(lambda) -- temporal-difference reinforcement learning of the evaluation weights.

WHERE REINFORCEMENT LEARNING IS USED (for the viva):
  * state          the board position
  * action         the move chosen by alpha-beta search (epsilon-greedy for exploration)
  * reward         only at the end of a self-play game: +1 White wins, 0 draw, -1 Black wins
  * value function V(s) = tanh( w . phi(s) / 400 ), a linear evaluation squashed into [-1, 1]
                   (White's point of view); `w` are the weights being learned
  * learning rule  temporal-difference error between the values of successive positions

  Why "Leaf": the agent does not act on V(s) of the board itself but on the result of a search.
  TDLeaf therefore applies the TD update to the value of the *principal-variation leaf* l_t of the
  search at position t (the position at the end of the best line found), not to the root position:

        d_t   = V(l_{t+1}) - V(l_t)                 (V(l_T) := game result z)
        dw    = alpha * sum_t  grad_w V(l_t) * sum_{j>=t} lambda^(j-t) d_j

  lambda=0 uses only one-step errors, lambda=1 would use the whole game outcome (Monte-Carlo).
  (Baxter, Tridgell, Weaver, "Learning to play chess using temporal differences", 2000.)

Implementation notes: self-play uses a fixed-depth alpha-beta search without transposition table so
that the principal variation is complete; each coordinate of the update is divided by the running
RMS of its feature so that features on very different scales (mobility vs piece counts) learn at a
comparable rate.
"""
import argparse
import csv
import ctypes
import multiprocessing as mp
import os
import random
import time

import numpy as np

from ._lib import CABOut, NF, lib
from .agents import FEATURE_NAMES, HAND_WEIGHTS
from .core import State

SCALE = 400.0
MATE = 30000


def init_weights(mode):
    w = np.zeros(NF)
    if mode == "hand":
        return HAND_WEIGHTS.copy()
    if mode == "equal":      # every piece counts the same; all positional weights unknown (zero)
        w[:5] = 100.0
        return w
    if mode == "material":   # textbook piece values, nothing else
        w[:5] = [100, 300, 320, 500, 900]
        return w
    raise ValueError(mode)


def search_root(state, w, depth, budget=10 ** 12, use_tt=False):
    out = CABOut()
    wa = (ctypes.c_double * NF)(*[float(x) for x in w])
    h = (ctypes.c_uint64 * max(1, len(state.hist)))(*state.hist)
    lib.mc_alphabeta(ctypes.byref(state.c), wa, budget, depth, 1 if use_tt else 0, h, len(state.hist), ctypes.byref(out))
    return out


def selfplay_game(args):
    """One self-play game. Returns (list of (phi_leaf, forced_value_or_None), z)."""
    w, depth, eps, seed, max_plies = args
    rng = random.Random(seed)
    lib.mc_clear_tt()
    s = State.start()
    recs = []
    while True:
        code, _ = s.status()
        if code != 0:
            z = 0.0 if code == 2 else float(code)
            break
        if s.ply() >= max_plies:
            z = 0.0
            break
        out = search_root(s, w, depth)
        phi = np.array(out.leaf[:], dtype=np.float64)
        forced = None
        if abs(out.score) >= MATE - 200:
            forced = float(np.sign(out.score) * s.player())   # White's view
        recs.append((phi, forced))
        m = out.move
        if rng.random() < eps:
            m = rng.choice(s.legal_moves())
        s = s.play(m)
    return recs, z


def td_update(w, games, lam, rms):
    """Sum of TDLeaf(lambda) updates over a list of games (before the learning rate)."""
    delta = np.zeros(NF)
    n_pos = 0
    for recs, z in games:
        T = len(recs)
        if T == 0:
            continue
        phis = np.array([r[0] for r in recs])
        V = np.tanh(phis @ w / SCALE)
        grad_mask = np.ones(T)
        for t, (_, forced) in enumerate(recs):
            if forced is not None:
                V[t] = forced
                grad_mask[t] = 0.0
        Vn = np.append(V[1:], z)
        d = Vn - V
        G = np.zeros(T)
        acc = 0.0
        for t in range(T - 1, -1, -1):
            acc = d[t] + lam * acc
            G[t] = acc
        grads = ((1 - V ** 2) * grad_mask / SCALE)[:, None] * phis
        delta += (grads * G[:, None]).sum(axis=0)
        n_pos += T
    return delta / max(1, n_pos), n_pos


def evaluate(w, budget, openings, procs):
    from harness.tournament import match
    sc, wn, dr, ls, _ = match({"kind": "ab", "budget": budget, "weights": list(map(float, w))},
                              {"kind": "ab", "budget": budget}, openings, procs=procs)
    return sc, wn, dr, ls


def train(games, depth=3, eps=0.15, lam=0.7, alpha=3000.0, batch=8, init="equal", eval_every=200,
          eval_budget=2000, seed=0, out_prefix="results/tdleaf", procs=2, log=print):
    from harness.openings import make_openings
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    openings = make_openings(30)
    w = init_weights(init)
    rms = np.ones(NF)
    rng = random.Random(seed)
    rows = []
    done = 0
    t0 = time.time()
    pool = mp.get_context("fork").Pool(procs)
    try:
        while done < games:
            if done % eval_every == 0:
                sc, wn, dr, ls = evaluate(w, eval_budget, openings, procs)
                rows.append([done, sc, wn, dr, ls] + list(w))
                log("games %5d  score vs hand-crafted %.3f (W%d D%d L%d)  %.0fs" % (done, sc, wn, dr, ls, time.time() - t0))
                with open(out_prefix + "_log.csv", "w", newline="") as f:
                    cw = csv.writer(f)
                    cw.writerow(["games", "score", "wins", "draws", "losses"] + FEATURE_NAMES)
                    cw.writerows(rows)
                np.save(out_prefix + "_weights.npy", w)
            args = [(w.copy(), depth, eps, rng.randrange(1 << 30), 120) for _ in range(batch)]
            res = pool.map(selfplay_game, args)
            for recs, _ in res:
                for phi, _f in recs:
                    rms = 0.999 * rms + 0.001 * np.maximum(np.abs(phi), 1.0)
            delta, _ = td_update(w, res, lam, rms)
            w = w + alpha * delta / rms
            done += batch
    finally:
        pool.close()
    sc, wn, dr, ls = evaluate(w, eval_budget, openings, procs)
    rows.append([done, sc, wn, dr, ls] + list(w))
    log("games %5d  score vs hand-crafted %.3f (W%d D%d L%d)  %.0fs" % (done, sc, wn, dr, ls, time.time() - t0))
    with open(out_prefix + "_log.csv", "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["games", "score", "wins", "draws", "losses"] + FEATURE_NAMES)
        cw.writerows(rows)
    np.save(out_prefix + "_weights.npy", w)
    return w


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=2000)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--lam", type=float, default=0.7)
    ap.add_argument("--alpha", type=float, default=3000.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--init", default="equal")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--out", default="results/tdleaf")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    train(a.games, a.depth, a.eps, a.lam, a.alpha, a.batch, a.init, a.eval_every, seed=a.seed, out_prefix=a.out)
