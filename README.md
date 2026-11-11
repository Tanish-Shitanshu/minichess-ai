# Mini Chess AI Lab (Gardner 5x5)

Course project, CSD311 Artificial Intelligence, Shiv Nadar University.

We solve/play **Gardner's minichess** (5x5 board, `RNBQK` / `PPPPP`, no castling, no double pawn step, no
en passant; weakly solved as a draw by Mhalla and Prost, 2013) with five techniques and compare them at the
**same compute budget**:

| Agent | Idea | Learning? |
|---|---|---|
| `AB hand-crafted` | iterative-deepening alpha-beta, transposition table, quiescence, linear hand-written evaluation | none |
| `MCTS` | UCT with random rollouts | none |
| `AB + TDLeaf` | same search, evaluation weights learned by **TDLeaf(lambda) self-play** | **reinforcement learning** |
| `AB + GA` | same search, evaluation weights found by a **genetic algorithm** (plain and box-constrained) | evolutionary (not RL) |
| `AlphaZero-style` | policy-value network + PUCT search, trained by **self-play** | **reinforcement learning** |

Everything is validated against an independent engine (Fairy-Stockfish, variant `gardner`): perft counts match,
and Fairy-Stockfish at fixed depths is used as an external Elo anchor and as the judge of move quality.

## Layout

```
engine/engine.cpp      C++ core: rules, perft, alpha-beta, MCTS, PUCT search, network forward pass, self-play
minichess/             Python package: ctypes wrapper (_lib), State (core), agents, specs,
                       tdleaf.py (RL), ga.py (evolution), azero.py (RL, numpy backprop + Adam)
harness/               tournament (parallel), openings, elo (Bradley-Terry + bootstrap), fairy (UCI link),
                       run_all.py (all experiments), plots.py (all figures)
web/                   Flask app + single-page UI (play, watch, search policy overlay, RL explorer, results)
tests/                 pytest: rules, perft against Fairy-Stockfish numbers, mate/stalemate/repetition
results/               logs, learned weights/networks, data/*.json (raw numbers), figures/
report/                IEEE-format report (LaTeX source + PDF)
```

## Quick start

```bash
pip install -r requirements.txt
g++ -O3 -march=native -std=c++17 -shared -fPIC engine/engine.cpp -o engine/libminichess.so   # also built automatically on first import
python -m pytest -q tests                      # rules are checked against known perft numbers
python -m web.app                              # http://localhost:5000   play / watch / explore
```

Fairy-Stockfish (only needed for experiments E2, E4 and the perft cross-check):

```bash
git clone --depth 1 https://github.com/fairy-stockfish/Fairy-Stockfish && cd Fairy-Stockfish/src
make -j2 build ARCH=x86-64   # see the Fairy-Stockfish README for your platform
cp stockfish ../../tools/fairy-stockfish
```

## Reproducing the results

```bash
python -m minichess.tdleaf --games 8000 --eval-every 400 --alpha 150000 --depth 3 --out results/tdleaf
python -m minichess.ga --generations 80 --pop 32 --budget 400 --kr 8 --out results/ga
python -m minichess.ga --constrain --generations 80 --pop 32 --budget 400 --kr 8 --out results/gac
python -m minichess.azero --iterations 300 --games 64 --sims 150 --steps 120 --H 192 --lr 1.5e-3 --out results/azero2
python -m harness.run_all          # E1 round robin, scaling, E2 vs Fairy-Stockfish, E3 learning curves, E4 move quality
python -m harness.plots            # figures into results/figures
```

Every stage caches to `results/data/*.json`; `--force` recomputes, `--quick` is a tiny smoke test.
On 2 CPU cores the full pipeline takes roughly an hour of wall-clock time (training dominates).

## Where reinforcement learning is used (for the viva)

* **TDLeaf(lambda)** (`minichess/tdleaf.py`). State = board; action = move chosen by a depth-limited alpha-beta search
  (epsilon-greedy for exploration); reward = game result, given only at the end. The value function is
  `V(s) = tanh(w . phi(s) / 400)` with 14 features `phi`. After each game the temporal-difference errors between
  successive *principal-variation leaf* values are propagated back with decay lambda and used to move `w`
  (gradient of `V` with respect to `w` evaluated at the leaves, which is what makes it "TD-Leaf").
* **AlphaZero-style** (`minichess/azero.py`, search in `engine/engine.cpp`). State = board planes; action = move;
  reward = +1/0/-1 at the end. Self-play with PUCT search produces (position, visit-count distribution, outcome)
  triples; the network is trained so that its policy imitates the visit counts (search = policy improvement) and its value
  predicts the outcome (policy evaluation). Repeating the two is policy iteration.
* The genetic algorithm and the hand-crafted/alpha-beta/MCTS agents are **not** reinforcement learning:
  the GA uses a population and selection on match results, not per-step credit assignment.

## Conventions

Squares: `a1`..`e5`, board index `rank*5+file`. Moves are UCI strings (`b2b3`, promotion `b4b5q`).
Scores are from the point of view of the first-named agent: win 1, draw 0.5, loss 0.
Every pairing is played from 30-60 fixed random 2-4-ply openings, both colours, so that deterministic agents
do not repeat one game.
