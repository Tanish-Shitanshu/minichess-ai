"""Agent specifications: small picklable dicts that can be turned into agents inside worker processes.

    {"kind": "ab",  "budget": 2000, "weights": [...14 floats...], "name": "Alpha-beta"}
    {"kind": "mcts","budget": 400,  "mode": 1}
    {"kind": "az",  "budget": 100,  "net": "results/azero_final.npz"}
    {"kind": "random"} / {"kind": "greedy"}
"""
import os

import numpy as np

from .agents import (AlphaBetaAgent, AZeroAgent, GreedyAgent, HAND_WEIGHTS, MCTSAgent, RandomAgent)

_NET_CACHE = {}


def load_net(path):
    key = (path, os.path.getmtime(path))
    if key not in _NET_CACHE:
        z = np.load(path)
        _NET_CACHE[key] = (z["params"].astype(np.float32), int(z["H"]))
    return _NET_CACHE[key]


def build_agent(spec):
    k = spec["kind"]
    name = spec.get("name")
    if k == "random":
        return RandomAgent(seed=spec.get("seed", 1))
    if k == "greedy":
        return GreedyAgent(weights=spec.get("weights"))
    if k == "ab":
        w = spec.get("weights")
        return AlphaBetaAgent(budget=spec.get("budget", 1000), weights=HAND_WEIGHTS if w is None else w,
                              name=name or "Alpha-beta", maxdepth=spec.get("maxdepth", 0),
                              use_tt=spec.get("use_tt", True), seed=spec.get("seed", 1))
    if k == "mcts":
        return MCTSAgent(budget=spec.get("budget", 400), cexp=spec.get("cexp", 0.7),
                         rollout_cap=spec.get("rollout_cap", 40), mode=spec.get("mode", 1),
                         name=name or "MCTS", seed=spec.get("seed", 1))
    if k == "az":
        params, H = load_net(spec["net"])
        return AZeroAgent(params, H, budget=spec.get("budget", 100), cpuct=spec.get("cpuct", 1.5),
                          name=name or "AlphaZero-style", seed=spec.get("seed", 1))
    raise ValueError(k)
