"""Python face of the Gardner 5x5 game core (the rules live in engine/engine.cpp).

This is the `State` interface from the project contract:

    State.start()            -> standard Gardner start position, White to move
    s.legal_moves()          -> list[int]   (queen-only promotions, see below)
    s.play(move)             -> new State   (immutable style)
    s.is_terminal()          -> bool
    s.result()               -> +1 White wins, 0 draw, -1 Black wins (only if terminal)
    s.player()               -> +1 White to move, -1 Black to move
    s.key()                  -> Zobrist hash
    s.planes()               -> numpy array (12,5,5): own pieces first, from the mover's view
    s.to_uci(move) / State.from_uci(s, str)

Gardner chess: 5x5 board, start position RNBQK / PPPPP, no castling, no pawn double step and no
en passant. A pawn reaching the last rank promotes. Our agents only generate queen promotions
(a documented simplification); the rules code can generate all four.
Draws: stalemate, 50-move rule (100 plies without capture or pawn move), insufficient material
(K vs K, K+minor vs K) and threefold repetition.
"""
import ctypes

import numpy as np

from ._lib import CState, NF, lib

PIECES = " PNBRQK"
START_FEN = "rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1"


def move_to_uci(m):
    buf = ctypes.create_string_buffer(8)
    lib.mc_move_to_str(int(m), buf)
    return buf.value.decode()


class State:
    __slots__ = ("c", "hist")

    def __init__(self, c=None, hist=()):
        if c is None:
            c = CState()
            lib.mc_start(ctypes.byref(c))
        self.c = c
        self.hist = tuple(hist)  # hashes of the positions before this one

    # ---- constructors -------------------------------------------------
    @staticmethod
    def start():
        return State()

    @staticmethod
    def from_fen(fen):
        c = CState()
        if lib.mc_from_fen(ctypes.byref(c), fen.encode()) != 0:
            raise ValueError("bad FEN: " + fen)
        return State(c)

    def copy(self):
        c = CState()
        ctypes.memmove(ctypes.byref(c), ctypes.byref(self.c), ctypes.sizeof(CState))
        return State(c, self.hist)

    # ---- basic queries --------------------------------------------------
    def fen(self):
        buf = ctypes.create_string_buffer(96)
        lib.mc_to_fen(ctypes.byref(self.c), buf)
        return buf.value.decode()

    def board(self):
        """25 ints, index = rank*5+file, +1..+6 White, -1..-6 Black."""
        return list(self.c.b)

    def player(self):
        return int(self.c.stm)

    def key(self):
        return int(self.c.hash)

    def ply(self):
        return int(self.c.ply)

    def in_check(self):
        return bool(lib.mc_in_check(ctypes.byref(self.c)))

    # ---- moves ------------------------------------------------------------
    def legal_moves(self, underpromo=False):
        out = (ctypes.c_int * 256)()
        n = lib.mc_legal_moves(ctypes.byref(self.c), out, 1 if underpromo else 0)
        return [out[i] for i in range(n)]

    def play(self, move):
        c = CState()
        ctypes.memmove(ctypes.byref(c), ctypes.byref(self.c), ctypes.sizeof(CState))
        lib.mc_make(ctypes.byref(c), int(move))
        return State(c, self.hist + (int(self.c.hash),))

    def to_uci(self, move):
        return move_to_uci(move)

    def from_uci(self, text):
        m = lib.mc_str_to_move(ctypes.byref(self.c), text.encode())
        if m < 0:
            raise ValueError("illegal move %r in %s" % (text, self.fen()))
        return m

    # ---- game status --------------------------------------------------------
    def status(self):
        """(code, reason): code 0 ongoing, +1 White wins, -1 Black wins, 2 draw."""
        h = (ctypes.c_uint64 * max(1, len(self.hist)))(*self.hist)
        reason = ctypes.c_int(0)
        code = lib.mc_status(ctypes.byref(self.c), h, len(self.hist), ctypes.byref(reason))
        return code, reason.value

    def is_terminal(self):
        return self.status()[0] != 0

    def result(self):
        code = self.status()[0]
        return 0 if code == 2 else code

    # ---- features -------------------------------------------------------------
    def features(self):
        out = (ctypes.c_double * NF)()
        lib.mc_features(ctypes.byref(self.c), out)
        return np.array(out[:], dtype=np.float64)

    def planes(self):
        cb = (ctypes.c_int8 * 25)()
        lib.mc_canonical_board(ctypes.byref(self.c), cb)
        return planes_from_canonical(np.array(cb[:], dtype=np.int8)[None, :])[0]

    def __str__(self):
        rows = []
        b = self.board()
        for r in range(4, -1, -1):
            row = []
            for f in range(5):
                p = b[r * 5 + f]
                ch = PIECES[abs(p)] if p else "."
                row.append(ch if p > 0 else ch.lower())
            rows.append(str(r + 1) + " " + " ".join(row))
        rows.append("  a b c d e   " + ("White" if self.player() > 0 else "Black") + " to move")
        return "\n".join(rows)


def planes_from_canonical(cb):
    """cb: (N,25) int8 canonical boards (side to move = positive) -> (N,12,25) float32 one-hot planes."""
    n = cb.shape[0]
    out = np.zeros((n, 12, 25), dtype=np.float32)
    for p in range(1, 7):
        out[:, p - 1, :] = (cb == p)
        out[:, 6 + p - 1, :] = (cb == -p)
    return out


def perft(state, depth, underpromo=False):
    return int(lib.mc_perft(ctypes.byref(state.c), depth, 1 if underpromo else 0))
