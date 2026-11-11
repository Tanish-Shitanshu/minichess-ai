"""Figures for the report, drawn from results/data/*.json and the training logs.
    python -m harness.plots            -> results/figures/*.png (and .pdf)
Colour = entity (same colour for the same agent in every figure); markers give a second encoding.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
DATA = os.path.join(RES, os.environ.get("MC_DATA", "data"))
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

from minichess.agents import FEATURE_NAMES  # noqa: E402

COL = {"Random": "#8a8983", "Greedy-1ply": "#eda100", "AB hand-crafted": "#2a78d6", "AB equal weights (TDLeaf start)": "#9bbfe8",
       "MCTS (random rollouts)": "#eb6834", "AB + TDLeaf (RL)": "#1baf7a",
       "AB + GA (evolution)": "#e87ba4", "AB + GA (constrained)": "#c98500", "AlphaZero-style (RL)": "#6250d6"}
MK = {"Random": "x", "Greedy-1ply": "+", "AB hand-crafted": "o", "AB equal weights (TDLeaf start)": "*", "MCTS (random rollouts)": "s",
      "AB + TDLeaf (RL)": "^", "AB + GA (evolution)": "D", "AB + GA (constrained)": "P", "AlphaZero-style (RL)": "v"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 200, "legend.frameon": False,
})


def load(name):
    p = os.path.join(DATA, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def out(fig, name):
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote", name)


def csv_rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def fig_e1():
    d = load("e1")
    if not d:
        return
    names, elo, lo, hi = d["names"], np.array(d["elo"]), np.array(d["lo"]), np.array(d["hi"])
    order = np.argsort(elo)
    fig, ax = plt.subplots(figsize=(5.2, 2.9))
    for k, i in enumerate(order):
        ax.barh(k, elo[i], color=COL[names[i]], height=0.62)
        ax.errorbar(elo[i], k, xerr=[[elo[i] - lo[i]], [hi[i] - elo[i]]], color=INK2, capsize=2, lw=0.9)
        ax.text(max(hi[i], elo[i]) + 25, k, "%+.0f" % elo[i], va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[i] for i in order])
    ax.axvline(0, color=INK2, lw=0.8)
    ax.set_xlabel("Elo relative to hand-crafted alpha-beta (95% bootstrap CI)")
    ax.set_title("E1: strength at an equal budget of %d nodes / simulations" % d["budget"])
    ax.grid(axis="y", visible=False)
    out(fig, "fig_e1_elo")
    # heat map of pairwise scores
    n = len(names)
    M = np.full((n, n), np.nan)
    for key, (sc, w, dr, l) in d["pairs"].items():
        a, b = key.split("|")
        i, j = names.index(a), names.index(b)
        M[i, j], M[j, i] = sc, 1 - sc
    o = list(order[::-1])
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(M[np.ix_(o, o)], cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([names[i] for i in o], rotation=40, ha="right")
    ax.set_yticklabels([names[i] for i in o])
    for r in range(n):
        for c in range(n):
            v = M[o[r], o[c]]
            if not np.isnan(v):
                ax.text(c, r, "%.2f" % v, ha="center", va="center", fontsize=7, color="white" if v > 0.6 else INK)
    ax.grid(False)
    ax.set_title("Score of row agent against column agent")
    fig.colorbar(im, ax=ax, fraction=0.04)
    out(fig, "fig_e1_heatmap")


def fig_scaling():
    d = load("scaling")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for m, rows in d["curves"].items():
        b = [r["budget"] for r in rows]
        e = [r["elo"] for r in rows]
        ax.plot(b, e, color=COL[m], marker=MK[m], lw=1.6, ms=5, label=m)
    ax.set_xscale("log", base=2)
    ax.axhline(0, color=INK2, lw=0.8, ls="--")
    ax.set_xlabel("search budget (nodes or simulations per move)")
    ax.set_ylabel("Elo vs alpha-beta @ %d nodes" % B_REF)
    ax.set_title("Strength grows with compute")
    ax.legend(fontsize=7.5, loc="lower right")
    out(fig, "fig_scaling")


B_REF = 400


def fig_e2():
    d = load("e2")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for n, r in d["results"].items():
        ds = sorted(int(k) for k in r)
        ax.plot(ds, [r[str(k)][0] for k in ds], color=COL[n], marker=MK[n], lw=1.6, ms=5, label=n)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Fairy-Stockfish search depth")
    ax.set_ylabel("score of our agent")
    ax.set_title("E2: against an independent engine (%d games per point)" % d["n_games"])
    ax.legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    out(fig, "fig_e2_fairy")


def fig_e3():
    d = load("e3")
    if not d:
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for n, rows in d.items():
        r = np.array(rows)
        r = r[r[:, 0] > 0] if (r[:, 0] > 0).any() else r
        ax.plot(r[:, 0], r[:, 1], color=COL[n], marker=MK[n], lw=1.4, ms=3.5, label=n)
    ax.axhline(0.5, color=INK2, lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("self-play / fitness games played")
    ax.set_ylabel("score vs hand-crafted alpha-beta")
    ax.set_title("E3: learning curves (same test for every method)")
    ax.legend(fontsize=7.5, loc="lower right")
    out(fig, "fig_e3_learning")


def fig_e4():
    d = load("e4")
    if not d:
        return
    res = d["results"]
    names = sorted(res, key=lambda n: res[n]["mean_loss"])
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
    for ax, key, ttl, fmt in ((axs[0], "mean_loss", "mean evaluation loss per move (centipawns)", "%.0f"),
                              (axs[1], "blunder_rate_200", "share of moves losing >= 200 cp", "%.0f%%")):
        for k, n in enumerate(names):
            v = res[n][key] * (100 if key == "blunder_rate_200" else 1)
            ax.barh(k, v, color=COL[n], height=0.62)
            ax.text(v + 0.01 * max(1, ax.get_xlim()[1]), k, fmt % v, va="center", fontsize=7.5, color=INK)
        ax.set_title(ttl, fontsize=8.5)
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(range(len(names)))
    axs[0].set_yticklabels(names)
    axs[0].invert_yaxis()
    fig.suptitle("E4: move quality against Fairy-Stockfish depth 9 (%d positions)" % d["n_positions"],
                 x=0.01, y=1.06, ha="left", fontsize=10, fontweight="bold")
    out(fig, "fig_e4_quality")


def fig_weights():
    d = load("weights")
    if not d:
        return
    f = d["feature"]
    keys = [k for k in ("hand-crafted", "TDLeaf", "GA", "GA-constrained") if k in d]
    cols = {"hand-crafted": COL["AB hand-crafted"], "TDLeaf": COL["AB + TDLeaf (RL)"], "GA": COL["AB + GA (evolution)"], "GA-constrained": COL["AB + GA (constrained)"]}
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1, 2.2]})
    piece = list(range(5))
    w = 0.2
    for k, key in enumerate(keys):
        vals = np.array(d[key])
        base = vals[0] if vals[0] != 0 else 1
        axs[0].bar(np.arange(5) + k * w, vals[:5] / base, width=w, color=cols[key], label=key)
    axs[0].set_xticks(np.arange(5) + w)
    axs[0].set_xticklabels(["P", "N", "B", "R", "Q"])
    axs[0].set_title("piece value (pawn = 1)")
    axs[0].legend(fontsize=7)
    rest = list(range(5, len(f)))
    for k, key in enumerate(keys):
        vals = np.array(d[key])
        axs[1].bar(np.arange(len(rest)) + k * w, vals[rest], width=w, color=cols[key])
    axs[1].set_xticks(np.arange(len(rest)) + w)
    axs[1].set_xticklabels([f[i] for i in rest], rotation=45, ha="right", fontsize=7)
    axs[1].set_title("positional weights (centipawns)")
    axs[1].axhline(0, color=INK2, lw=0.8)
    out(fig, "fig_weights")


def fig_training():
    az = [p for p in ("azero2_log.csv", "azero_log.csv") if os.path.exists(os.path.join(RES, p))]
    if not az:
        return
    rows = [r for r in csv_rows(os.path.join(RES, az[0])) if int(r["iteration"]) <= 200]  # final network = iteration 200
    g = np.array([float(r["games"]) for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.5))
    axs[0].plot(g, [float(r["policy_loss"]) for r in rows], color=COL["AlphaZero-style (RL)"], lw=1.4)
    axs[0].set_title("policy loss"); axs[0].set_xlabel("self-play games")
    axs[1].plot(g, [float(r["value_loss"]) for r in rows], color=COL["AlphaZero-style (RL)"], lw=1.4)
    axs[1].set_title("value loss"); axs[1].set_xlabel("self-play games")
    for key, lab, c in (("score_random", "vs random", "#8a8983"), ("score_ab200", "vs alpha-beta 200", COL["AB hand-crafted"]),
                        ("score_mcts400", "vs MCTS 400", COL["MCTS (random rollouts)"])):
        axs[2].plot(g, [float(r[key]) if r.get(key) not in (None, "") else np.nan for r in rows], lw=1.3, color=c, label=lab)
    axs[2].set_title("score during training"); axs[2].set_xlabel("self-play games"); axs[2].legend(fontsize=7)
    out(fig, "fig_azero_training")


SHORT = {"Random": "Random", "Greedy-1ply": "Greedy 1-ply", "AB hand-crafted": "AB hand-crafted",
         "AB equal weights (TDLeaf start)": "AB equal weights", "MCTS (random rollouts)": "UCT MCTS",
         "AB + TDLeaf (RL)": "AB + TDLeaf (RL)", "AB + GA (evolution)": "AB + GA", "AB + GA (constrained)": "AB + GA (constr.)",
         "AlphaZero-style (RL)": "AlphaZero-style (RL)"}


def fig_report():
    """Two wide, multi-panel figures for the paper."""
    e1, e2, e3, e4, sc, wt = (load(n) for n in ("e1", "e2", "e3", "e4", "scaling", "weights"))
    if not (e1 and e2 and e3):
        return
    # ---- figure A: E1 Elo | E2 vs Fairy-Stockfish | E3 learning curves
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.55), gridspec_kw={"width_ratios": [1.15, 1, 1]})
    names, elo, lo, hi = e1["names"], np.array(e1["elo"]), np.array(e1["lo"]), np.array(e1["hi"])
    order = np.argsort(elo)
    ax = axs[0]
    for k, i in enumerate(order):
        ax.barh(k, elo[i], color=COL[names[i]], height=0.64)
        ax.errorbar(elo[i], k, xerr=[[elo[i] - lo[i]], [hi[i] - elo[i]]], color=INK2, capsize=1.5, lw=0.8)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([SHORT[names[i]] for i in order], fontsize=7)
    ax.axvline(0, color=INK2, lw=0.8); ax.grid(axis="y", visible=False)
    ax.set_xlabel("Elo vs AB hand-crafted (95% CI)", fontsize=7.5)
    ax.set_title("(a) E1: equal-budget round robin", fontsize=8.5)
    ax = axs[1]
    for n, r in e2["results"].items():
        ds = sorted(int(k) for k in r)
        ax.plot(range(len(ds)), [r[str(k)][0] for k in ds], color=COL[n], marker=MK[n], lw=1.2, ms=3.5)
    ax.set_xticks(range(len(ds))); ax.set_xticklabels(ds)
    ax.axhline(0.5, color=INK2, lw=0.7, ls="--"); ax.set_ylim(-0.02, 0.8)
    ax.set_xlabel("Fairy-Stockfish depth", fontsize=7.5); ax.set_ylabel("score of our agent", fontsize=7.5)
    ax.set_title("(b) E2: external opponent", fontsize=8.5)
    ax = axs[2]
    for n, rows in e3.items():
        r = np.array(rows); r = r[r[:, 0] > 0]
        ax.plot(r[:, 0], r[:, 1], color=COL[n], marker=MK[n], lw=1.1, ms=2.8)
    ax.axhline(0.5, color=INK2, lw=0.7, ls="--"); ax.set_xscale("log")
    ax.set_xlabel("training games", fontsize=7.5); ax.set_ylabel("score vs AB hand-crafted", fontsize=7.5)
    ax.set_title("(c) E3: learning curves", fontsize=8.5)
    for a in axs[1:]:
        a.tick_params(labelsize=7)
    axs[0].tick_params(axis="x", labelsize=7)
    handles = [plt.Line2D([0], [0], color=COL[n], marker=MK[n], lw=1.2, ms=4, label=SHORT[n]) for n in names if n != "Random"]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7, bbox_to_anchor=(0.62, -0.17), handlelength=1.6)
    fig.tight_layout(w_pad=1.0)
    out(fig, "fig_report_a")
    # ---- figure B: scaling | move quality | learned piece values
    if not (sc and e4 and wt):
        return
    fig, axs = plt.subplots(1, 3, figsize=(7.3, 2.35), gridspec_kw={"width_ratios": [1, 1.1, 1]})
    ax = axs[0]
    for m, rows in sc["curves"].items():
        if m.startswith("MCTS"):      # always below -400 Elo: omitted from this panel, stated in the text
            continue
        ax.plot([r["budget"] for r in rows], [r["elo"] for r in rows], color=COL[m], marker=MK[m], lw=1.2, ms=3.5, label=SHORT[m])
    ax.set_xscale("log", base=2); ax.axhline(0, color=INK2, lw=0.7, ls="--")
    ax.set_xlabel("budget (nodes / simulations)", fontsize=7.5); ax.set_ylabel("Elo vs AB @ 400", fontsize=7.5)
    ax.set_title("(a) strength vs compute", fontsize=8.5); ax.tick_params(labelsize=7)
    ax = axs[1]
    res = e4["results"]; nm = sorted(res, key=lambda n: res[n]["mean_loss"])
    for k, n in enumerate(nm):
        v = res[n]["mean_loss"]
        ax.barh(k, v, color=COL[n], height=0.64)
        ax.text(v + 8, k, "%.0f" % v, va="center", fontsize=6.5, color=INK)
    ax.set_yticks(range(len(nm))); ax.set_yticklabels([SHORT[n] for n in nm], fontsize=6.5); ax.invert_yaxis()
    ax.grid(axis="y", visible=False); ax.set_xlabel("mean centipawn loss per move", fontsize=7.5)
    ax.set_title("(b) E4: move quality", fontsize=8.5); ax.tick_params(axis="x", labelsize=7)
    ax = axs[2]
    keys = [k for k in ("hand-crafted", "TDLeaf", "GA", "GA-constrained") if k in wt]
    cols = {"hand-crafted": COL["AB hand-crafted"], "TDLeaf": COL["AB + TDLeaf (RL)"], "GA": COL["AB + GA (evolution)"], "GA-constrained": COL["AB + GA (constrained)"]}
    w = 0.2
    for k, key in enumerate(keys):
        v = np.array(wt[key]); ax.bar(np.arange(4) + k * w, v[1:5] / v[0], width=w, color=cols[key], label=key)
    ax.set_xticks(np.arange(4) + 1.5 * w); ax.set_xticklabels(["N", "B", "R", "Q"], fontsize=7)
    ax.set_ylabel("value (pawn = 1)", fontsize=7.5); ax.set_title("(c) learned piece values", fontsize=8.5)
    ax.legend(fontsize=6, loc="upper left"); ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(w_pad=1.0)
    out(fig, "fig_report_b")


def main():
    for f in (fig_e1, fig_scaling, fig_e2, fig_e3, fig_e4, fig_weights, fig_training, fig_report):
        f()


if __name__ == "__main__":
    main()
