"""Rules tests. Perft numbers were produced by Fairy-Stockfish (variant gardner) and match our engine."""
import pytest
from minichess.core import State, perft

START = [7, 53, 506, 4775, 52512, 572874]


def test_start_position():
    s = State.start()
    assert s.fen() == "rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1"
    assert len(s.legal_moves()) == 7


@pytest.mark.parametrize("depth,expected", list(enumerate(START, 1)))
def test_perft_start(depth, expected):
    assert perft(State.start(), depth, underpromo=True) == expected


@pytest.mark.parametrize("fen,depth,expected", [
    ("rnbqk/pP1pp/2p2/1PPPP/RNBQK b - - 0 2", 4, 30257),   # promotion-captures
    ("4k/P4/5/5/K4 w - - 0 1", 6, 33618),                  # pawn about to promote
    ("r3k/5/2n2/5/R2QK w - - 0 1", 5, 550912),
    ("2k2/5/5/5/2K1R w - - 0 1", 6, 115150),
    ("rnbqk/p1ppp/5/1p3/RNBQK w - - 0 1", 5, 781161),
])
def test_perft_positions(fen, depth, expected):
    assert perft(State.from_fen(fen), depth, underpromo=True) == expected


def test_mate_and_stalemate():
    mate = State.from_fen("k4/Q4/1K3/5/5 b - - 0 1")
    assert mate.status() == (1, 1)          # White wins by checkmate
    stale = State.from_fen("k4/2Q2/1K3/5/5 b - - 0 1")
    assert stale.status() == (2, 2)         # draw by stalemate


def test_insufficient_material_and_fifty_moves():
    assert State.from_fen("k4/5/5/5/KN3 w - - 0 1").status() == (2, 4)
    assert State.from_fen("k4/5/5/r4/KN3 w - - 100 90").status()[0] != 0


def test_threefold_repetition():
    s = State.start()
    for m in ["b1c3", "b5a3", "c3b1", "a3b5"] * 2:
        s = s.play(s.from_uci(m))
    assert s.status() == (2, 5)
