"""Tests for the learning code: backprop gradients, Elo fitting, self-play data, agent legality."""
import numpy as np

from harness.elo import fit_elo
from minichess.agents import AlphaBetaAgent, MCTSAgent, RandomAgent, play_game
from minichess.azero import init_params, loss_and_grads, selfplay_worker
from minichess.core import State


def test_azero_gradient_matches_finite_differences():
    H = 6
    rng = np.random.default_rng(1)
    flat = init_params(H, 1).astype(np.float64) + rng.normal(0, 0.05, 1)  # float64 for a clean check
    flat += rng.normal(0, 0.05, flat.shape)
    B = 5
    X = (rng.random((B, 300)) < 0.07).astype(np.float64)
    masks = rng.random((B, 625)) < 0.1
    masks[:, 0] = True
    pi = rng.random((B, 625)) * masks
    pi /= pi.sum(1, keepdims=True)
    z = rng.choice([-1.0, 0.0, 1.0], B)
    _, _, _, g = loss_and_grads(flat, H, X, pi, masks, z)
    worst = 0.0
    for k in rng.choice(flat.size, 30, replace=False):
        e = 1e-5
        f1, f2 = flat.copy(), flat.copy()
        f1[k] += e
        f2[k] -= e
        num = (loss_and_grads(f1, H, X, pi, masks, z)[0] - loss_and_grads(f2, H, X, pi, masks, z)[0]) / (2 * e)
        worst = max(worst, abs(num - g[k]) / (abs(num) + abs(g[k]) + 1e-8))
    assert worst < 1e-3


def test_elo_fit_recovers_known_difference():
    # A beats B 75% of the time -> about +191 Elo
    games = [(0, 1, 1.0)] * 75 + [(0, 1, 0.0)] * 25
    r = fit_elo(games, ["A", "B"], prior_sd=5000.0)
    assert abs((r[0] - r[1]) - 400 * np.log10(3)) < 5


def test_selfplay_outputs_are_consistent():
    H = 16
    params = init_params(H, 0)
    boards, pols, masks, zs, stats = selfplay_worker((params, H, 10, 2, 5, 60, 8, 1, 1.5))
    n = len(zs)
    assert n > 0 and boards.shape == (n, 25) and pols.shape == (n, 625) and masks.shape == (n, 625)
    assert np.allclose(pols.sum(1), 1.0, atol=1e-4)
    assert np.all(pols[~masks] == 0)           # the search policy never puts mass on illegal moves
    assert set(np.unique(zs)) <= {-1.0, 0.0, 1.0}
    assert stats[0] + stats[1] + stats[2] == 2  # white wins + draws + black wins = games


def test_every_agent_plays_only_legal_moves():
    for ag in (RandomAgent(), AlphaBetaAgent(budget=200), MCTSAgent(budget=50)):
        s = State.start()
        for _ in range(12):
            if s.is_terminal():
                break
            m = ag.choose_move(s)
            assert m in s.legal_moves(underpromo=True)
            s = s.play(m)


def test_play_game_returns_consistent_result():
    res, moves, reason = play_game(AlphaBetaAgent(budget=200), RandomAgent(), (), 150)
    assert res in (-1, 0, 1) and len(moves) > 0
    s = State.start()
    for m in moves:
        s = s.play(m)
    if reason != 6:
        assert s.is_terminal()
