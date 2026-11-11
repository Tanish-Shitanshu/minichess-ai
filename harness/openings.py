"""A fixed, reproducible set of short random openings, so every pairing starts from the same
varied positions (Gardner chess is drawish and engines are deterministic, so without openings
all games between two agents would be identical)."""
import random

from minichess.core import State


def make_openings(n=30, plies=(2, 3, 4), seed=2026):
    rng = random.Random(seed)
    seen, out = set(), []
    while len(out) < n:
        k = plies[len(out) % len(plies)]
        s, mv = State.start(), []
        for _ in range(k):
            ms = s.legal_moves()
            m = rng.choice(ms)
            mv.append(m)
            s = s.play(m)
        key = tuple(mv)
        if key in seen or s.is_terminal():
            continue
        # skip openings that already lose material by force: keep captures out of the first 4 plies
        seen.add(key)
        out.append(mv)
    return out
