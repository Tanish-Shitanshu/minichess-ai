"""Runs every experiment of the report and caches the raw numbers in results/data/*.json.

    python -m harness.run_all                 # everything (stages already on disk are skipped)
    python -m harness.run_all --stages e1 e4  # selected stages
    python -m harness.run_all --force         # recompute
    python -m harness.run_all --quick         # tiny version, only to test the pipeline

Stages:
  weights  learned evaluation weights table (hand-crafted vs TDLeaf vs GA)
  e1       round robin of all agents at one compute budget (400 nodes / simulations), Elo + bootstrap CI
  scaling  Elo vs compute budget (25 ... 1600) against a fixed reference
  e2       agents vs Fairy-Stockfish at fixed depths 1..6 (independent external anchor)
  e3       learning curves: every logged checkpoint re-evaluated with the same protocol
  e4       move quality: evaluation lost by each agent's move vs a depth-9 Fairy-Stockfish search
  timing   milliseconds per move at the common budget
"""
import argparse
import csv
import json
import multiprocessing as mp
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
DATA = os.path.join(RES, "data")
os.makedirs(DATA, exist_ok=True)
sys.path.insert(0, ROOT)

from harness.elo import bootstrap_elo, score_ci            # noqa: E402
from harness.openings import make_openings                 # noqa: E402
from harness.tournament import match, run_jobs             # noqa: E402
from minichess.agents import FEATURE_NAMES, HAND_WEIGHTS  # noqa: E402

B = 400            # common budget: alpha-beta nodes, MCTS simulations, AlphaZero PUCT simulations


def az_path():
    for p in ("azero2_final.npz", "azero_final.npz"):
        if os.path.exists(os.path.join(RES, p)):
            return os.path.join(RES, p)
    raise FileNotFoundError("no AlphaZero network found in results/")


def players(budget=B):
    P = {
        "Random": {"kind": "random"},
        "Greedy-1ply": {"kind": "greedy"},
        "AB hand-crafted": {"kind": "ab", "budget": budget},
        "MCTS (random rollouts)": {"kind": "mcts", "budget": budget},
    }
    P["AB equal weights (TDLeaf start)"] = {"kind": "ab", "budget": budget, "weights": [100.0] * 5 + [0.0] * 9}
    f = os.path.join(RES, "tdleaf_weights.npy")
    if os.path.exists(f):
        P["AB + TDLeaf (RL)"] = {"kind": "ab", "budget": budget, "weights": list(map(float, np.load(f)))}
    f = os.path.join(RES, "ga_weights.npy")
    if os.path.exists(f):
        P["AB + GA (evolution)"] = {"kind": "ab", "budget": budget, "weights": list(map(float, np.load(f)))}
    f = os.path.join(RES, "gac_weights.npy")
    if os.path.exists(f):
        P["AB + GA (constrained)"] = {"kind": "ab", "budget": budget, "weights": list(map(float, np.load(f)))}
    try:
        P["AlphaZero-style (RL)"] = {"kind": "az", "budget": budget, "net": az_path()}
    except FileNotFoundError:
        pass
    return P


def save(name, obj):
    with open(os.path.join(DATA, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


def have(name, force):
    return (not force) and os.path.exists(os.path.join(DATA, name + ".json"))


def elo_of_score(s):
    s = min(max(s, 0.01), 0.99)
    return 400 * np.log10(s / (1 - s))


# ---------------------------------------------------------------------------------------------
def stage_weights(a):
    rows = {"feature": FEATURE_NAMES, "hand-crafted": list(map(float, HAND_WEIGHTS))}
    for nm, f in (("TDLeaf", "tdleaf_weights.npy"), ("GA", "ga_weights.npy"), ("GA-constrained", "gac_weights.npy")):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            rows[nm] = list(map(float, np.load(p)))
    save("weights", rows)


def stage_e1(a):
    P = players()
    names = list(P)
    ops = make_openings(10 if a.quick else a.openings, seed=99)
    games, table = [], {}
    t0 = time.time()
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            sc, w, d, l, res = match(P[names[i]], P[names[j]], ops, procs=2)
            table["%s|%s" % (names[i], names[j])] = [sc, w, d, l]
            games += [(i, j, r[0]) for r in res]
            print("  E1 %-24s vs %-24s %.3f (W%d D%d L%d)  %.0fs" % (names[i], names[j], sc, w, d, l, time.time() - t0), flush=True)
    anchor = names.index("AB hand-crafted")
    elo, lo, hi = bootstrap_elo(games, names, anchor=anchor, anchor_value=0.0, n_boot=100 if a.quick else 300)
    save("e1", {"names": names, "budget": B, "pairs": table, "elo": list(map(float, elo)),
                "lo": list(map(float, lo)), "hi": list(map(float, hi)), "n_openings": len(ops)})


def stage_scaling(a):
    methods = {"AB hand-crafted": lambda b: {"kind": "ab", "budget": b},
               "MCTS (random rollouts)": lambda b: {"kind": "mcts", "budget": b}}
    P = players()
    if "AB + GA (evolution)" in P:
        methods["AB + GA (evolution)"] = lambda b, w=P["AB + GA (evolution)"]["weights"]: {"kind": "ab", "budget": b, "weights": w}
    if "AB + GA (constrained)" in P:
        methods["AB + GA (constrained)"] = lambda b, w=P["AB + GA (constrained)"]["weights"]: {"kind": "ab", "budget": b, "weights": w}
    if "AlphaZero-style (RL)" in P:
        methods["AlphaZero-style (RL)"] = lambda b, n=P["AlphaZero-style (RL)"]["net"]: {"kind": "az", "budget": b, "net": n}
    budgets = [25, 50, 100, 200, 400, 800, 1600] if not a.quick else [50, 400]
    ref = {"kind": "ab", "budget": B}
    ops = make_openings(10 if a.quick else 30, seed=77)
    out = {}
    for m, mk in methods.items():
        out[m] = []
        for b in budgets:
            sc, w, d, l, _ = match(mk(b), ref, ops, procs=2)
            out[m].append({"budget": b, "score": sc, "w": w, "d": d, "l": l, "elo": elo_of_score(sc)})
            print("  scaling %-24s budget %5d  score vs AB-hand@400 %.3f" % (m, b, sc), flush=True)
    save("scaling", {"budgets": budgets, "ref": "AB hand-crafted @400", "curves": out})


def _fs_task(t):
    from harness.fairy import play_vs_fs
    name, spec, depth, ops = t
    sc, w, d, l = play_vs_fs(spec, depth, ops)
    return name, depth, sc, w, d, l


def stage_e2(a):
    P = players()
    pick = [n for n in P if n not in ("Random",)]
    depths = [1, 2, 3, 4, 6, 8, 10, 12] if not a.quick else [1, 3]
    ops = make_openings(4 if a.quick else a.fs_openings, seed=55)
    tasks = [(n, P[n], d, ops) for n in pick for d in depths]
    t0 = time.time()
    with mp.get_context("fork").Pool(2) as pool:
        rows = []
        for r in pool.imap_unordered(_fs_task, tasks):
            rows.append(r)
            print("  E2 %-24s vs FS depth %d: %.3f (W%d D%d L%d)  %.0fs" % (r[0], r[1], r[2], r[3], r[4], r[5], time.time() - t0), flush=True)
    out = {n: {} for n in pick}
    for n, d, sc, w, dr, l in rows:
        out[n][str(d)] = [sc, w, dr, l]
    save("e2", {"depths": depths, "n_games": 2 * len(ops), "results": out})


def _eval_weights_task(t):
    kind, key, w, ops = t
    sc, wn, d, l, _ = match({"kind": "ab", "budget": B, "weights": w}, {"kind": "ab", "budget": B}, ops, procs=1)
    return key, sc


def _eval_net_task(t):
    key, path, ops = t
    sc, *_ = match({"kind": "az", "budget": B, "net": path}, {"kind": "ab", "budget": B}, ops, procs=1)
    return key, sc


def stage_e3(a):
    ops = make_openings(10 if a.quick else 40, seed=33)
    out = {}
    nf = len(FEATURE_NAMES)
    with mp.get_context("fork").Pool(2) as pool:
        for nm, fn in (("AB + TDLeaf (RL)", "tdleaf_log.csv"), ("AB + GA (evolution)", "ga_log.csv"),
                       ("AB + GA (constrained)", "gac_log.csv")):
            p = os.path.join(RES, fn)
            if not os.path.exists(p):
                continue
            rows = list(csv.DictReader(open(p)))
            keys = FEATURE_NAMES
            tasks = [("w", float(r["games"]), [float(r[k]) for k in keys], ops) for r in rows]
            res = pool.map(_eval_weights_task, tasks)
            out[nm] = sorted([[k, s] for k, s in res])
            print("  E3 %s: %d checkpoints" % (nm, len(res)), flush=True)
        for nm, pat in (("AlphaZero-style (RL)", "azero2") if os.path.exists(os.path.join(RES, "azero2_final.npz")) else ("AlphaZero-style (RL)", "azero"),):
            ck = sorted(f for f in os.listdir(RES) if f.startswith(pat + "_iter") and f.endswith(".npz"))
            tasks = []
            for f in ck:
                z = np.load(os.path.join(RES, f))
                games = float(z["games"]) if "games" in z.files else 0.0
                tasks.append((games, os.path.join(RES, f), ops))
            res = pool.map(_eval_net_task, tasks)
            out[nm] = sorted([[k, s] for k, s in res])
            print("  E3 %s: %d checkpoints" % (nm, len(res)), flush=True)
    save("e3", out)


# ---- E4 --------------------------------------------------------------------------------------
def e4_positions(n, seed=11):
    from minichess.core import State
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        s = State.start()
        k = rng.randint(6, 28)
        ok = True
        for _ in range(k):
            ms = s.legal_moves()
            if not ms:
                ok = False
                break
            s = s.play(rng.choice(ms))
            if s.is_terminal():
                ok = False
                break
        if ok and len(s.legal_moves()) >= 2:
            out.append(s)
    return out


_E4 = {}


def _e4_chunk(args):
    from harness.fairy import FS
    from minichess.specs import build_agent
    chunk, P = args
    fs = FS()
    agents = {n: build_agent(s) for n, s in P.items()}
    rows = []
    for s in chunk:
        best_cp, best_mv = fs.analyse(s.fen(), 9)
        r = {"best": best_mv, "best_cp": best_cp, "loss": {}, "agree": {}}
        for n, ag in agents.items():
            m = ag.choose_move(s)
            u = s.to_uci(m)
            if u == best_mv:
                loss = 0.0
            else:
                child = s.play(m)
                if child.is_terminal():
                    v = {1: 10000, -1: -10000, 0: 0}[child.result()]
                    v = v if s.player() > 0 else -v
                else:
                    v = -fs.analyse(child.fen(), 8)[0]
                loss = float(min(1000, max(0, best_cp - v)))
            r["loss"][n] = loss
            r["agree"][n] = int(u == best_mv)
        rows.append(r)
    fs.close()
    return rows


def stage_e4(a):
    P = players()
    pos = e4_positions(30 if a.quick else a.positions)
    chunks = [pos[i::8] for i in range(8)]
    with mp.get_context("fork").Pool(2) as pool:
        parts = pool.map(_e4_chunk, [(c, P) for c in chunks])
    rows = [r for p in parts for r in p]
    out = {}
    for n in P:
        L = np.array([r["loss"][n] for r in rows])
        A = np.array([r["agree"][n] for r in rows])
        out[n] = {"mean_loss": float(L.mean()), "median_loss": float(np.median(L)),
                  "best_move_rate": float(A.mean()), "blunder_rate_200": float((L >= 200).mean()),
                  "losses": L.tolist()}
        print("  E4 %-24s mean loss %6.1f cp  best-move %.2f  blunders(>=200cp) %.2f" % (n, L.mean(), A.mean(), (L >= 200).mean()), flush=True)
    save("e4", {"n_positions": len(rows), "fs_depth": 9, "results": out})


def _ms_per_move(spec, positions, reps=3):
    """Mean milliseconds per move, after a warm-up (the first call allocates the transposition table)."""
    from minichess.specs import build_agent
    from minichess._lib import lib
    ag = build_agent(spec)
    lib.mc_clear_tt()
    for st in positions[:5]:
        ag.choose_move(st, keep_tt=True)          # warm-up: the first search allocates the transposition table
    best = float("inf")
    for _ in range(reps):
        t = time.time()
        for st in positions:
            ag.choose_move(st, keep_tt=True)      # as in a real game, the table persists between moves
        best = min(best, 1000 * (time.time() - t) / len(positions))
    return best


def stage_timing(a):
    P = players()
    pos = e4_positions(40, seed=5)
    out, extra = {}, {}
    for n, s in P.items():
        out[n] = _ms_per_move(s, pos)
        print("  timing %-34s %.3f ms/move" % (n, out[n]), flush=True)
    for n, b in (("AB hand-crafted", 100), ("AB hand-crafted", 1600), ("AB hand-crafted", 6400), ("AB hand-crafted", 25600),
                 ("AlphaZero-style (RL)", 100), ("AlphaZero-style (RL)", 1600), ("MCTS (random rollouts)", 1600)):
        spec = dict(P[n]); spec["budget"] = b
        extra["%s@%d" % (n, b)] = _ms_per_move(spec, pos)
        print("  timing %-34s @%d %.3f ms/move" % (n, b, extra["%s@%d" % (n, b)]), flush=True)
    save("timing", {"budget": B, "ms_per_move": out, "extra": extra})


def stage_equaltime(a):
    """Equal WALL-CLOCK comparison: alpha-beta is given as many nodes as it can search in the time the other agent needs."""
    P = players()
    pos = e4_positions(40, seed=5)
    ab = P["AB hand-crafted"]
    npms = 25600 / _ms_per_move(dict(ab, budget=25600), pos)             # alpha-beta nodes per millisecond
    ops = make_openings(10 if a.quick else a.openings, seed=99)
    out = {"nodes_per_ms": npms, "rows": []}
    for name in ("AlphaZero-style (RL)", "MCTS (random rollouts)", "AB + TDLeaf (RL)"):
        if name not in P:
            continue
        ms = _ms_per_move(P[name], pos)
        nb = int(round(ms * npms))
        if name.startswith("AB"):
            continue
        sc, w, d, l, _ = match(P[name], dict(ab, budget=max(nb, 25)), ops, procs=2)
        out["rows"].append({"agent": name, "ms": ms, "ab_budget": nb, "score": sc, "w": w, "d": d, "l": l})
        print("  equal-time %-24s (%.2f ms) vs AB hand-crafted @%d nodes: %.3f (W%d D%d L%d)" % (name, ms, nb, sc, w, d, l), flush=True)
    from minichess.agents import AlphaBetaAgent
    az_budget = next((r["ab_budget"] for r in out["rows"] if r["agent"].startswith("AlphaZero")), 6400)

    def mean_depth(b):
        ag = AlphaBetaAgent(budget=b)
        return float(np.mean([ag.choose_with_info(st)[3] for st in pos]))
    out["depth_400"], out["depth_eq"] = mean_depth(B), mean_depth(az_budget)
    print("  mean alpha-beta depth: %.1f at %d nodes, %.1f at %d nodes" % (out["depth_400"], B, out["depth_eq"], az_budget))
    save("equaltime", out)


STAGES = {"weights": stage_weights, "e1": stage_e1, "scaling": stage_scaling, "e2": stage_e2,
          "e3": stage_e3, "e4": stage_e4, "timing": stage_timing, "equaltime": stage_equaltime}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--openings", type=int, default=60)
    ap.add_argument("--fs-openings", dest="fs_openings", type=int, default=20)
    ap.add_argument("--positions", type=int, default=300)
    a = ap.parse_args()
    if a.quick:
        DATA = os.path.join(RES, "data_quick")
        os.makedirs(DATA, exist_ok=True)
    for s in a.stages:
        if have(s, a.force) and not a.quick:
            print("[skip] %s (cached)" % s)
            continue
        print("[run] %s" % s, flush=True)
        t = time.time()
        STAGES[s](a)
        print("[done] %s in %.0fs" % (s, time.time() - t), flush=True)
