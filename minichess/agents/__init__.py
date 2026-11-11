"""Agents. Every agent has the same interface:

    agent.name                       readable name
    agent.choose_move(state)         -> move (int) for the position `state` (a core.State)
    agent.cfg()                      -> ctypes AgentCfg so the C++ engine can play it directly

The `budget` is the compute knob used for the compute-matched comparison:
nodes for alpha-beta based agents, simulations for MCTS and AlphaZero-style agents.
"""
import ctypes

import numpy as np

from .._lib import CAgentCfg, NF, lib

A_RANDOM, A_ALPHABETA, A_MCTS, A_AZERO, A_GREEDY = 0, 1, 2, 3, 4


def hand_weights():
    w = (ctypes.c_double * NF)()
    lib.mc_hand_weights(w)
    return np.array(w[:], dtype=np.float64)


HAND_WEIGHTS = hand_weights()
FEATURE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "pawn@3rd", "pawn@4th",
                 "cent.N", "cent.B", "cent.R", "cent.Q", "cent.K", "mobility", "king shield"]


class Agent:
    type = A_RANDOM
    name = "agent"
    budget = 1
    seed = 1

    def _fill(self, c):
        pass

    def cfg(self):
        c = CAgentCfg()
        c.type = self.type
        c.budget = int(self.budget)
        c.seed = int(self.seed)
        c.maxdepth = 0
        c.use_tt = 0
        c.cpuct = 0.0
        c.rollout_cap = 0
        c.rollout_mode = 0
        c.H = 0
        c.temp_moves = 0
        c.rand_open = 0
        w = getattr(self, "weights", HAND_WEIGHTS)
        for i in range(NF):
            c.w[i] = float(w[i])
        self._fill(c)
        return c

    def choose_move(self, state, budget=None, keep_tt=False):
        """Best move for `state`. The transposition table is cleared first unless keep_tt=True (use keep_tt
        inside one game; clearing keeps independent queries, and different agents, from sharing cached scores)."""
        if not keep_tt:
            lib.mc_clear_tt()
        old = self.budget
        if budget is not None:
            self.budget = budget
        try:
            c = self.cfg()
            h = (ctypes.c_uint64 * max(1, len(state.hist)))(*state.hist)
            return lib.mc_choose(ctypes.byref(c), ctypes.byref(state.c), h, len(state.hist), None, None, None)
        finally:
            self.budget = old

    def choose_with_info(self, state, keep_tt=False):
        if not keep_tt:
            lib.mc_clear_tt()
        c = self.cfg()
        h = (ctypes.c_uint64 * max(1, len(state.hist)))(*state.hist)
        score, nodes, depth = ctypes.c_double(), ctypes.c_int64(), ctypes.c_int()
        m = lib.mc_choose(ctypes.byref(c), ctypes.byref(state.c), h, len(state.hist),
                          ctypes.byref(score), ctypes.byref(nodes), ctypes.byref(depth))
        return m, score.value, nodes.value, depth.value


class RandomAgent(Agent):
    type = A_RANDOM

    def __init__(self, seed=1):
        self.name = "Random"
        self.seed = seed


class GreedyAgent(Agent):
    """One-ply greedy on the hand-crafted evaluation (a sanity-check baseline)."""
    type = A_GREEDY

    def __init__(self, weights=None, seed=1):
        self.name = "Greedy-1ply"
        self.weights = HAND_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
        self.seed = seed


class AlphaBetaAgent(Agent):
    """Iterative-deepening alpha-beta with transposition table, move ordering and quiescence search.
    The evaluation is linear: V(s) = w . phi(s). With the default weights it is the hand-crafted
    baseline; TDLeaf and the genetic algorithm only change `weights`."""
    type = A_ALPHABETA

    def __init__(self, budget=1000, weights=None, name="Alpha-beta", maxdepth=0, use_tt=True, seed=1):
        self.budget = budget
        self.weights = HAND_WEIGHTS if weights is None else np.asarray(weights, dtype=np.float64)
        self.name = name
        self.maxdepth = maxdepth
        self.use_tt = use_tt
        self.seed = seed

    def _fill(self, c):
        c.maxdepth = int(self.maxdepth)
        c.use_tt = 1 if self.use_tt else 0


class MCTSAgent(Agent):
    """UCT with uniformly random playouts. mode 0: playouts to the end (draw at the cap);
    mode 1: playouts to the cap followed by a soft material adjudication."""
    type = A_MCTS

    def __init__(self, budget=400, cexp=0.7, rollout_cap=40, mode=1, name="MCTS", seed=1):
        self.budget = budget
        self.cexp = cexp
        self.rollout_cap = rollout_cap
        self.mode = mode
        self.name = name
        self.seed = seed

    def _fill(self, c):
        c.cpuct = self.cexp
        c.rollout_cap = self.rollout_cap
        c.rollout_mode = self.mode


class AZeroAgent(Agent):
    """AlphaZero-style agent: PUCT search guided by a policy-value network (flat float32 weights)."""
    type = A_AZERO

    def __init__(self, params, H, budget=100, cpuct=1.5, name="AlphaZero-style", seed=1):
        self.params = np.ascontiguousarray(params, dtype=np.float32)
        self.H = int(H)
        self.budget = budget
        self.cpuct = cpuct
        self.name = name
        self.seed = seed

    def _fill(self, c):
        c.cpuct = self.cpuct
        c.H = self.H
        c.net = self.params.ctypes.data_as(ctypes.c_void_p)


def play_game(white, black, opening=(), max_plies=150):
    """Plays one game inside the C++ engine. Returns (result for White, list of moves, reason)."""
    cw, cb = white.cfg(), black.cfg()
    op = (ctypes.c_int * max(1, len(opening)))(*opening)
    moves = (ctypes.c_int * 512)()
    n = ctypes.c_int()
    reason = ctypes.c_int()
    res = lib.mc_play_game(ctypes.byref(cw), ctypes.byref(cb), op, len(opening), max_plies, moves,
                           ctypes.byref(n), ctypes.byref(reason))
    return res, [moves[i] for i in range(n.value)], reason.value
