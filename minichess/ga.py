"""Genetic algorithm that evolves the same 14 evaluation weights that TDLeaf learns.

NOT reinforcement learning: there are no gradients and no per-move learning signal. A population of
weight vectors ("genomes") is evolved; the only feedback is the score each genome earns in games.

    fitness      average score of the genome (alpha-beta agent with those weights) in games against
                 other genomes of the current population (co-evolution, like self-play)
    selection    tournament selection (best of 3 random genomes)
    crossover    blend crossover (BLX-0.3): each gene of the child is drawn uniformly from an
                 interval slightly wider than the one spanned by the two parents' genes
    mutation     each gene is perturbed with probability 0.3 by Gaussian noise
    elitism      the 2 best genomes survive unchanged

Using the same feature set and the same search as TDLeaf means any difference between the two
comes from the learning rule, not from the model.
"""
import argparse
import csv
import os
import random
import time

import numpy as np

from ._lib import NF
from .agents import FEATURE_NAMES, HAND_WEIGHTS

GENE_SCALE = np.array([20, 20, 20, 20, 20, 5, 5, 5, 5, 5, 5, 5, 3, 3], dtype=float)


# Optional box constraints ("prior knowledge"): a piece is worth at least 1.5 pawns, mobility is never a
# penalty, and so on. Without them the population can find degenerate weights (e.g. a worthless rook)
# that win against the other genomes but are not chess knowledge.
LO = np.array([80, 150, 150, 150, 150, -40, -40, -40, -40, -40, -40, -40, 0, -10], dtype=float)
HI = np.array([120, 700, 700, 1000, 1500, 60, 80, 60, 60, 60, 60, 60, 30, 40], dtype=float)


def random_genome(rng, constrain=False):
    if constrain:
        return rng.uniform(LO, HI)
    g = np.zeros(NF)
    g[:5] = rng.uniform(20, 500, 5)
    g[5:] = rng.normal(0, 15, NF - 5)
    g[12] = rng.normal(0, 8)
    g[13] = rng.normal(0, 8)
    return g


def fitness_round(pop, budget, openings, k_rounds, rng, procs):
    from harness.tournament import run_jobs
    n = len(pop)
    jobs, owners = [], []
    for _ in range(k_rounds):
        perm = list(range(n))
        rng.shuffle(perm)
        for a, b in zip(perm[0::2], perm[1::2]):
            op = rng.choice(openings)
            sa = {"kind": "ab", "budget": budget, "weights": list(map(float, pop[a]))}
            sb = {"kind": "ab", "budget": budget, "weights": list(map(float, pop[b]))}
            jobs.append((sa, sb, op, True, 120)); owners.append((a, b))
            jobs.append((sa, sb, op, False, 120)); owners.append((a, b))
    res = run_jobs(jobs, procs)
    tot = np.zeros(n)
    cnt = np.zeros(n)
    for (a, b), (score_a, _, _) in zip(owners, res):
        tot[a] += score_a; cnt[a] += 1
        tot[b] += 1 - score_a; cnt[b] += 1
    return tot / np.maximum(cnt, 1), len(jobs)


def evolve(constrain=False, generations=40, pop_size=24, budget=400, k_rounds=6, eval_every=2, eval_budget=2000, seed=0,
           out_prefix="results/ga", procs=2, log=print):
    from harness.openings import make_openings
    from harness.tournament import match
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    pool = make_openings(300, plies=(4, 5, 6), seed=7)
    eval_openings = make_openings(30)
    pop = [random_genome(nrng, constrain) for _ in range(pop_size)]
    rows, games_played, t0 = [], 0, time.time()
    best = pop[0]
    for gen in range(generations + 1):
        fit, ng = fitness_round(pop, budget, pool, k_rounds, rng, procs)
        games_played += ng
        order = np.argsort(-fit)
        best = pop[order[0]].copy()
        if gen % eval_every == 0 or gen == generations:
            sc, w, d, l, _ = match({"kind": "ab", "budget": eval_budget, "weights": list(map(float, best))},
                                   {"kind": "ab", "budget": eval_budget}, eval_openings, procs=procs)
            rows.append([games_played, gen, sc, w, d, l, float(fit.mean()), float(fit[order[0]])] + list(best))
            log("gen %3d games %6d  best vs hand-crafted %.3f (W%d D%d L%d)  mean fit %.3f  %.0fs"
                % (gen, games_played, sc, w, d, l, fit.mean(), time.time() - t0))
            os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)
            with open(out_prefix + "_log.csv", "w", newline="") as f:
                cw = csv.writer(f)
                cw.writerow(["games", "generation", "score", "wins", "draws", "losses", "mean_fitness", "best_fitness"] + FEATURE_NAMES)
                cw.writerows(rows)
            np.save(out_prefix + "_weights.npy", best)
        if gen == generations:
            break
        sigma = 1.0 - 0.6 * gen / generations   # mutation strength decays over time
        new = [pop[order[0]].copy(), pop[order[1]].copy()]   # elitism
        while len(new) < pop_size:
            def pick():
                c = rng.sample(range(pop_size), 3)
                return pop[max(c, key=lambda i: fit[i])]
            p1, p2 = pick(), pick()
            lo, hi = np.minimum(p1, p2), np.maximum(p1, p2)
            span = hi - lo
            child = nrng.uniform(lo - 0.3 * span, hi + 0.3 * span)
            mask = nrng.random(NF) < 0.3
            child = child + mask * nrng.normal(0, GENE_SCALE * sigma)
            if constrain:
                child = np.clip(child, LO, HI)
            new.append(child)
        pop = new
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--budget", type=int, default=400)
    ap.add_argument("--constrain", action="store_true")
    ap.add_argument("--procs", type=int, default=2)
    ap.add_argument("--kr", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/ga")
    a = ap.parse_args()
    evolve(a.constrain, a.generations, a.pop, a.budget, k_rounds=a.kr, seed=a.seed, out_prefix=a.out, procs=a.procs)
