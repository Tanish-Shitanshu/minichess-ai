"""Plays many games in parallel (one process per CPU core). Agents are described by spec dicts."""
import multiprocessing as mp
import os

from minichess.agents import play_game
from minichess.specs import build_agent

_CACHE = {}


def _agent(spec):
    key = repr(sorted(spec.items(), key=lambda kv: kv[0]))
    a = _CACHE.get(key)
    if a is None:
        if len(_CACHE) > 64:
            _CACHE.clear()
        a = _CACHE[key] = build_agent(spec)
    return a


def play_job(job):
    """job = (specA, specB, opening, a_is_white, max_plies). Returns (score for A in {1,.5,0}, plies, reason)."""
    sa, sb, opening, a_white, max_plies = job
    a, b = _agent(sa), _agent(sb)
    res, moves, reason = play_game(a if a_white else b, b if a_white else a, opening, max_plies)
    score_white = {1: 1.0, 0: 0.5, -1: 0.0}[res]
    return (score_white if a_white else 1.0 - score_white), len(moves), reason


def run_jobs(jobs, procs=None):
    procs = procs or os.cpu_count() or 1
    if procs <= 1 or len(jobs) < 2:
        return [play_job(j) for j in jobs]
    with mp.get_context("fork").Pool(procs) as pool:
        return pool.map(play_job, jobs, chunksize=max(1, len(jobs) // (procs * 8)))


def match(spec_a, spec_b, openings, max_plies=150, procs=None):
    """A vs B on every opening, both colours. Returns (score of A in [0,1], wins, draws, losses, results list)."""
    jobs = []
    for op in openings:
        jobs.append((spec_a, spec_b, op, True, max_plies))
        jobs.append((spec_a, spec_b, op, False, max_plies))
    res = run_jobs(jobs, procs)
    scores = [r[0] for r in res]
    w = sum(1 for x in scores if x == 1.0)
    d = sum(1 for x in scores if x == 0.5)
    l = sum(1 for x in scores if x == 0.0)
    return sum(scores) / len(scores), w, d, l, res
