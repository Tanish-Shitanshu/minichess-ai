"""AlphaZero-style self-play reinforcement learning for Gardner 5x5 chess.

WHERE REINFORCEMENT LEARNING IS USED (for the viva):
  The agent is a policy-value neural network f(s) = (p, v) wrapped in Monte-Carlo tree search.
    state   board seen from the side to move (12 one-hot planes x 25 squares)
    action  a move; the network outputs one logit per (from, to) pair, masked to the legal moves
    reward  only at the end of a self-play game: +1 win / 0 draw / -1 loss for the side to move
  Training loop (this file, `train`):
    1. SELF-PLAY      the current network plays against itself. At every position MCTS (PUCT) is run
                      and the move is sampled from the visit counts (temperature 1 early in the game).
                      Dirichlet noise is added to the root prior to force exploration.
    2. TARGETS        policy target pi = normalised root visit counts (the search result),
                      value target z  = final game outcome from that position's side to move.
    3. SUPERVISED STEP minimise   (z - v)^2  -  sum_a pi(a) log p(a)  +  c ||theta||^2   with Adam.
  In RL terms the search is the policy-IMPROVEMENT operator (visit counts are a better policy than
  the raw network output) and regression onto z is policy EVALUATION; repeating the two is a form of
  policy iteration. Nothing but the rules of the game and the win/draw/loss result is given.

  The tree search, network forward pass and self-play loop run in C++ (engine/engine.cpp);
  the network weights and the training step (backpropagation + Adam) are in numpy here.

Network: input 300 -> H relu -> H relu -> {policy: 625 logits, value: tanh}. (A small MLP is enough
for a 5x5 board and keeps self-play fast on CPU.)
"""
import argparse
import csv
import ctypes
import multiprocessing as mp
import os
import time

import numpy as np

from ._lib import CAgentCfg, lib
from .core import planes_from_canonical


# --------------------------------------------------------------------------------------------
# parameters: one flat float32 vector (the layout is shared with the C++ forward pass)
# --------------------------------------------------------------------------------------------
def param_shapes(H):
    return [("W1", (300, H)), ("b1", (H,)), ("W2", (H, H)), ("b2", (H,)),
            ("Wp", (625, H)), ("bp", (625,)), ("Wv", (H,)), ("bv", (1,))]


def n_params(H):
    return sum(int(np.prod(s)) for _, s in param_shapes(H))


def unpack(flat, H):
    out, i = {}, 0
    for name, shape in param_shapes(H):
        n = int(np.prod(shape))
        out[name] = flat[i:i + n].reshape(shape)
        i += n
    return out


def init_params(H, seed=0):
    rng = np.random.default_rng(seed)
    flat = np.zeros(n_params(H), dtype=np.float32)
    P = unpack(flat, H)
    P["W1"][:] = rng.normal(0, 0.15, P["W1"].shape)
    P["W2"][:] = rng.normal(0, np.sqrt(2.0 / H), P["W2"].shape)
    P["Wp"][:] = rng.normal(0, 0.02, P["Wp"].shape)
    P["Wv"][:] = rng.normal(0, 0.05, P["Wv"].shape)
    return flat


# --------------------------------------------------------------------------------------------
# forward / backward in numpy
# --------------------------------------------------------------------------------------------
def forward(P, X, masks=None):
    z1 = X @ P["W1"] + P["b1"]
    h1 = np.maximum(z1, 0)
    z2 = h1 @ P["W2"] + P["b2"]
    h2 = np.maximum(z2, 0)
    logits = h2 @ P["Wp"].T + P["bp"]
    if masks is not None:
        logits = np.where(masks, logits, -1e9)
    v = np.tanh(h2 @ P["Wv"] + P["bv"][0])
    return z1, h1, z2, h2, logits, v


def loss_and_grads(flat, H, X, pi, masks, z, c_l2=1e-4, c_v=1.0):
    P = unpack(flat, H)
    z1, h1, z2, h2, logits, v = forward(P, X, masks)
    B = X.shape[0]
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m) * masks
    sm = e / e.sum(axis=1, keepdims=True)
    logsm = np.log(np.maximum(sm, 1e-12))
    lp = -(pi * logsm).sum() / B
    lv = ((z - v) ** 2).mean()
    l2 = c_l2 * sum((P[k] ** 2).sum() for k in ("W1", "W2", "Wp", "Wv"))
    loss = lp + c_v * lv + l2
    g = np.zeros_like(flat)
    G = unpack(g, H)
    dlogits = (sm - pi) / B * masks
    G["Wp"][:] = dlogits.T @ h2
    G["bp"][:] = dlogits.sum(axis=0)
    dv_pre = c_v * 2 * (v - z) * (1 - v ** 2) / B
    G["Wv"][:] = h2.T @ dv_pre
    G["bv"][0] = dv_pre.sum()
    dh2 = dlogits @ P["Wp"] + np.outer(dv_pre, P["Wv"])
    dz2 = dh2 * (z2 > 0)
    G["W2"][:] = h1.T @ dz2
    G["b2"][:] = dz2.sum(axis=0)
    dh1 = dz2 @ P["W2"].T
    dz1 = dh1 * (z1 > 0)
    G["W1"][:] = X.T @ dz1
    G["b1"][:] = dz1.sum(axis=0)
    for k in ("W1", "W2", "Wp", "Wv"):
        G[k] += 2 * c_l2 * P[k]
    return loss, lp, lv, g


class Adam:
    def __init__(self, n, lr=2e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(n, dtype=np.float32)
        self.v = np.zeros(n, dtype=np.float32)
        self.t, self.lr, self.b1, self.b2, self.eps = 0, lr, b1, b2, eps

    def step(self, flat, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        flat -= self.lr * mh / (np.sqrt(vh) + self.eps)


# --------------------------------------------------------------------------------------------
# self-play (runs inside the C++ engine)
# --------------------------------------------------------------------------------------------
def selfplay_worker(args):
    params, H, sims, ngames, seed, max_plies, temp_moves, rand_open, cpuct = args
    params = np.ascontiguousarray(params, dtype=np.float32)
    cfg = CAgentCfg()
    cfg.budget = sims
    cfg.seed = seed
    cfg.cpuct = cpuct
    cfg.dir_alpha = 0.3
    cfg.dir_eps = 0.25
    cfg.net = params.ctypes.data_as(ctypes.c_void_p)
    cfg.H = H
    cfg.temp_moves = temp_moves
    cfg.rand_open = rand_open
    max_pos = ngames * max_plies
    boards = np.zeros((max_pos, 25), dtype=np.int8)
    pols = np.zeros((max_pos, 625), dtype=np.float32)
    masks = np.zeros((max_pos, 625), dtype=np.int8)
    zs = np.zeros(max_pos, dtype=np.float32)
    stats = (ctypes.c_int * 4)()
    n = lib.mc_selfplay(ctypes.byref(cfg), ngames, max_plies,
                        boards.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        pols.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        masks.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
                        zs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), max_pos, stats)
    return boards[:n].copy(), pols[:n].copy(), masks[:n].astype(bool), zs[:n].copy(), list(stats)


def save_net(path, params, H, meta=None):
    np.savez(path, params=params, H=H, **(meta or {}))


def train(iterations=60, H=128, games_per_iter=48, sims=60, steps_per_iter=60, batch=256, lr=2e-3,
          buffer_size=60000, max_plies=120, temp_moves=12, rand_open=1, cpuct=1.5, seed=0,
          out_prefix="results/azero", procs=2, eval_every=1, resume=None, log=print):
    from harness.openings import make_openings
    from harness.tournament import match
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
    if resume:
        z = np.load(resume)
        params, H = z["params"].astype(np.float32).copy(), int(z["H"])
    else:
        params = init_params(H, seed)
    opt = Adam(params.size, lr=lr)
    buf_b = np.zeros((0, 25), np.int8)
    buf_p = np.zeros((0, 625), np.float32)
    buf_m = np.zeros((0, 625), bool)
    buf_z = np.zeros(0, np.float32)
    eval_ops = make_openings(10)
    rows, total_games, t0 = [], 0, time.time()
    pool = mp.get_context("fork").Pool(procs)
    try:
        for it in range(1, iterations + 1):
            t_sp = time.time()
            per = max(1, games_per_iter // procs)
            args = [(params, H, sims, per, int(rng.integers(1 << 30)), max_plies, temp_moves, rand_open, cpuct)
                    for _ in range(procs)]
            outs = pool.map(selfplay_worker, args)
            st = np.sum([o[4] for o in outs], axis=0)
            nb = np.concatenate([o[0] for o in outs]); npl = np.concatenate([o[1] for o in outs])
            nm = np.concatenate([o[2] for o in outs]); nz = np.concatenate([o[3] for o in outs])
            total_games += per * procs
            buf_b = np.concatenate([buf_b, nb])[-buffer_size:]
            buf_p = np.concatenate([buf_p, npl])[-buffer_size:]
            buf_m = np.concatenate([buf_m, nm])[-buffer_size:]
            buf_z = np.concatenate([buf_z, nz])[-buffer_size:]
            t_sp = time.time() - t_sp
            # ---- training steps on the replay buffer
            t_tr = time.time()
            X_all_cache = None
            lps, lvs = [], []
            N = len(buf_z)
            for _ in range(steps_per_iter):
                idx = rng.integers(0, N, size=min(batch, N))
                X = planes_from_canonical(buf_b[idx]).reshape(len(idx), 300)
                loss, lp, lv, g = loss_and_grads(params, H, X, buf_p[idx], buf_m[idx], buf_z[idx])
                opt.step(params, g)
                lps.append(lp); lvs.append(lv)
            t_tr = time.time() - t_tr
            row = dict(iteration=it, games=total_games, positions=N, policy_loss=float(np.mean(lps)),
                       value_loss=float(np.mean(lvs)), avg_len=st[3] / max(1, per * procs),
                       sp_white=int(st[0]), sp_draw=int(st[1]), sp_black=int(st[2]), t_selfplay=t_sp, t_train=t_tr)
            if it % eval_every == 0:
                spec = {"kind": "az", "budget": sims, "net": out_prefix + "_current.npz"}
                save_net(out_prefix + "_current.npz", params, H)
                s_r = match(spec, {"kind": "random"}, eval_ops, procs=procs)[0]
                s_ab = match(spec, {"kind": "ab", "budget": 200}, eval_ops, procs=procs)[0]
                s_mc = match(spec, {"kind": "mcts", "budget": 400}, eval_ops, procs=procs)[0]
                row.update(score_random=s_r, score_ab200=s_ab, score_mcts400=s_mc)
            rows.append(row)
            log("it %3d games %5d  Lp %.3f Lv %.3f len %.0f  W/D/B %d/%d/%d  vsRand %.2f vsAB200 %.2f vsMCTS400 %.2f  [sp %.0fs tr %.0fs, total %.0fs]"
                % (it, total_games, row["policy_loss"], row["value_loss"], row["avg_len"], st[0], st[1], st[2],
                   row.get("score_random", float("nan")), row.get("score_ab200", float("nan")),
                   row.get("score_mcts400", float("nan")), t_sp, t_tr, time.time() - t0))
            with open(out_prefix + "_log.csv", "w", newline="") as f:
                keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[-1]).index(k) if k in rows[-1] else 99)
                cw = csv.DictWriter(f, fieldnames=keys)
                cw.writeheader()
                cw.writerows(rows)
            if it % 10 == 0 or it == iterations:
                save_net("%s_iter%03d.npz" % (out_prefix, it), params, H, {"games": total_games})
    finally:
        pool.close()
    save_net(out_prefix + "_final.npz", params, H, {"games": total_games})
    return params, H


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--games", type=int, default=48)
    ap.add_argument("--sims", type=int, default=60)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/azero")
    ap.add_argument("--resume", default=None)
    a = ap.parse_args()
    train(a.iterations, a.H, a.games, a.sims, a.steps, lr=a.lr, seed=a.seed, out_prefix=a.out, resume=a.resume)
