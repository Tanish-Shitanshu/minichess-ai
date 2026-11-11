"""UCI link to Fairy-Stockfish (variant 'gardner') used as an EXTERNAL reference opponent / evaluator.

 - play_vs_fs(spec, depth, openings): our agent vs Fairy-Stockfish at a fixed search depth
   -> gives an Elo anchor: FS depth d is an independent, well-tested engine.
 - fs_eval(fen, depth): centipawn score (side-to-move view) of the best line, plus best move.
   Used by E4 (move quality = evaluation lost by the move our agent chose vs FS's best move).
"""
import os
import subprocess

from minichess._lib import lib
from minichess.core import State
from minichess.specs import build_agent

FS_PATH = os.path.join(os.path.dirname(__file__), "..", "tools", "fairy-stockfish")


class FS:
    def __init__(self, path=FS_PATH):
        self.p = subprocess.Popen([path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self._send("uci"); self._wait("uciok")
        self._send("setoption name UCI_Variant value gardner")
        self._send("setoption name Threads value 1")
        self._send("isready"); self._wait("readyok")

    def _send(self, s):
        self.p.stdin.write(s + "\n"); self.p.stdin.flush()

    def _wait(self, token):
        lines = []
        while True:
            ln = self.p.stdout.readline()
            if not ln:
                raise RuntimeError("fairy-stockfish died")
            lines.append(ln.strip())
            if ln.strip().startswith(token):
                return lines

    def analyse(self, fen, depth):
        """returns (score_cp from side-to-move view, best move uci). mate scores mapped to +-10000."""
        self._send("ucinewgame")
        self._send("position fen " + fen)
        self._send("go depth %d" % depth)
        lines = self._wait("bestmove")
        score = 0
        for ln in lines:
            if ln.startswith("info") and " score " in ln and " depth %d " % depth in (ln + " "):
                t = ln.split()
                i = t.index("score")
                kind, val = t[i + 1], int(t[i + 2])
                score = val if kind == "cp" else (10000 - abs(val)) * (1 if val > 0 else -1)
        best = lines[-1].split()[1]
        return score, best

    def close(self):
        try:
            self._send("quit")
        except Exception:
            pass
        self.p.kill()


def play_vs_fs(spec, depth, openings, max_plies=200):
    """Our agent plays every opening as both colours against FS at `depth`.
    Returns (score, W, D, L): score = (W + 0.5 D)/N from our agent's view."""
    fs = FS()
    agent = build_agent(spec)
    W = D = L = 0
    try:
        for op in openings:
            for ours_white in (True, False):
                st = State.start()
                lib.mc_clear_tt()
                for m in op:
                    st = st.play(m)
                while not st.is_terminal() and st.ply() < max_plies:
                    ours = (st.player() > 0) == ours_white
                    if ours:
                        mv = agent.choose_move(st, keep_tt=True)
                    else:
                        _, uci = fs.analyse(st.fen(), depth)
                        mv = st.from_uci(uci)
                    st = st.play(mv)
                r = st.result() if st.is_terminal() else 0   # +1 white win, -1 black win, 0 draw
                if r == 0:
                    D += 1
                elif (r > 0) == ours_white:
                    W += 1
                else:
                    L += 1
    finally:
        fs.close()
    n = W + D + L
    return (W + 0.5 * D) / n, W, D, L
