# Viva guide: Mini Chess AI project

Read this once the night before. Every number marked **[R]** is in `results/data/*.json` and in the report; the
table at the bottom of this file repeats the headline ones. If a number here ever disagrees with the report,
trust the report/JSON and tell the examiner you re-checked.

---------------------------------------------------------------------------------------------------

## 0. Thirty-second summary (say this first)

We took Gardner 5x5 minichess, which is a proven draw with perfect play. We built one C++ rules engine, validated it
against Fairy-Stockfish by counting legal move sequences (perft), and put **five techniques** on top of it:
alpha-beta search with a hand-written evaluation, UCT Monte-Carlo tree search, alpha-beta with an evaluation learned by
**TDLeaf(lambda) reinforcement learning**, alpha-beta with an evaluation found by a **genetic algorithm**, and an
**AlphaZero-style** agent trained by **self-play reinforcement learning**. We compared them at equal compute
(400 nodes or simulations per move) in a round robin, against Fairy-Stockfish at fixed depths, and by how much
evaluation each move throws away. Headline results (all at equal budget, B = 400):
the four alpha-beta agents with a sensible evaluation (hand-crafted, TDLeaf, GA, constrained GA) are within 47 Elo of each other
and statistically tied (CI about +-32 Elo). Learning matters a lot relative to its start: TDLeaf starts from equal piece values and
gains **+272 Elo**; the GA starts from random weights and ends at the hand-crafted level (+7 / +23). The AlphaZero-style network
is **-90 Elo** behind the hand-crafted agent and costs about 12x more time per simulation (2.3 ms vs 0.19 ms per move); at equal
wall-clock it scores 0.25. UCT MCTS is far behind (-598). Fairy-Stockfish and the move-quality metric give the same ranking.
We also found and fixed a bug in our own harness (section 6b): you should be ready to tell that story, it shows the checks work.

---------------------------------------------------------------------------------------------------

## 1. The game

* 5x5 board, White `RNBQK` on rank 1, `PPPPP` on rank 2; Black mirrored. No castling, no two-square pawn move, no
  en passant. Pawns promote on the last rank (to Q, R, B or N; we generate all four).
* Draws: stalemate, 50 moves without capture/pawn move (100 plies), insufficient material, threefold repetition.
  We also stop at 150 plies and call it a draw (only used in experiments, reason code 6).
* Mhalla and Prost (2013) solved it: **draw** with best play. So "winning" means the opponent erred.

**How do you know your move generator is right?** Perft: count all move sequences to depth d from the start position
and compare with Fairy-Stockfish's `go perft d` on variant `gardner`. They agree (tests/test_core.py). Perft also catches
subtle bugs (illegal moves leaving the king in check, promotions).

---------------------------------------------------------------------------------------------------

## 2. Where reinforcement learning is applied (the question you WILL be asked)

RL = an agent learns from *rewards* obtained by interacting with an environment. Here the environment is the game.
Our only reward is at the end of a game: +1 win, 0 draw, -1 loss. Two of the five agents use it.

| | TDLeaf(lambda) | AlphaZero-style |
|---|---|---|
| State | board position | board position as 12 piece planes from the mover's side |
| Action | the move picked by a depth-3 alpha-beta search (random move with prob. 0.15 for exploration) | a move; the network scores all 625 (from,to) pairs, illegal ones masked |
| Reward | game result at the end only | game result at the end only |
| What is learned | 14 numbers `w` of a linear evaluation | ~216k network weights (policy and value) |
| How | temporal-difference update on the **value of the principal-variation leaf** | self-play with PUCT search; policy imitates search visit counts; value regresses onto game result |
| RL view | TD(lambda) with a search in the loop | approximate **policy iteration**: search = policy improvement, regression on outcome = policy evaluation |

Not RL: the GA (selection on match results, no per-move credit assignment, no value function), alpha-beta and MCTS
(no learning at all; MCTS "learns" within one move but forgets afterwards).

**Why TD instead of just using the final result for every position (Monte-Carlo)?** Lower variance: a TD target is the
next position's value, which already contains information, and lambda interpolates (lambda=0 one-step, lambda=1 equals
using the final result).

**Why "Leaf"?** The program does not play `argmax V(next board)`; it plays the move that search says is best. The value
that explains the move is the evaluation at the *end of the best line* (the leaf), so the derivative and the TD error
are taken at that leaf (Baxter et al., 2000).

**Exploration**: epsilon = 0.15 random moves in TDLeaf; Dirichlet noise (alpha 0.3, weight 0.25) at the AlphaZero root and
sampling the first 12 moves in proportion to visit counts.

---------------------------------------------------------------------------------------------------

## 3. The five agents, one paragraph each

**Alpha-beta (hand-crafted).** Minimax with alpha-beta pruning: do not search lines that cannot change the result.
Iterative deepening (depth 1, 2, 3, ...) until the node budget is used; Zobrist transposition table (cache of positions
already searched); killer moves, history heuristic and MVV-LVA capture ordering (good ordering is what makes pruning
work); quiescence search (keep searching captures at depth 0 so we do not evaluate in the middle of an exchange);
check extension. Leaf evaluation = linear combination of 14 features (material, advanced pawns, piece centrality, mobility,
king shield). Weak spot: only as good as its evaluation and the depth it can afford.

**UCT MCTS.** Build a tree by repeating: select child by `Q + c*sqrt(ln N / n)` (c = 0.7), expand a leaf, play a *random*
rollout, back the result up. No evaluation function. Weak spot: random rollouts are very noisy in chess, where one
blunder decides the game, so it needs many simulations; we also adjudicate unfinished rollouts by material.

**Alpha-beta + TDLeaf.** Same search, weights learned from 8000 self-play games starting from "all pieces worth 100,
everything else 0" (no chess knowledge).

**Alpha-beta + GA.** Same search, weights found by evolution: population 32, fitness = score against random members of
the population (co-evolution), tournament selection, blend crossover, Gaussian mutation, 2 elites, 80 generations.
Variant "constrained": weights limited to a sensible box (pieces at least 1.5 pawns, mobility not negative).

**AlphaZero-style.** MLP 300-192-192 with policy (625) and value heads, PUCT search with 150 simulations during
self-play, 64 games per iteration, replay buffer of 60,000 positions, Adam. Loss = (z - v)^2 - sum pi log p + L2.
We trained 200 iterations = 12,800 self-play games on 2 CPU cores in about 35 minutes.
Weak spot: small data and tiny network compared with the real AlphaZero (which used thousands of TPUs).

---------------------------------------------------------------------------------------------------

## 4. Evaluation methodology (why the numbers can be trusted)

* **Equal compute**: one budget number per move (nodes for alpha-beta, simulations for MCTS and AlphaZero). We also
  report milliseconds per move because a simulation is not the same cost as a node.
* **Fixed openings**: agents are deterministic, so playing one game repeatedly proves nothing. We use 60 fixed random
  2-4-ply openings, each played with both colours (120 games per pairing). Same openings for every pairing.
* **Elo**: Bradley-Terry maximum likelihood on all games (draw = half a point), 95% bootstrap intervals by resampling
  games. Anchor: hand-crafted alpha-beta = 0.
* **External anchor (E2)**: Fairy-Stockfish at fixed depth is independent of everything we wrote.
* **Move quality (E4)**: for 300 positions, centipawn loss of the agent's move against Fairy-Stockfish depth 9. Blunder =
  loss of at least 200 cp.
* **Limitations to state honestly**: one training run per method (no seeds), 2 CPU cores, small networks, 120 games per pairing
  (CI about +-32 Elo on the four strongest, so no ranking among them), 20 openings and 40 games per point against Fairy-Stockfish,
  Fairy-Stockfish depth is not a calibrated Elo, learned weights were tuned at one budget only.

---------------------------------------------------------------------------------------------------

## 5. Likely questions and good answers

**Q: Why minichess?** Small enough to train and test complete pipelines on CPUs; game value known (draw), so there is a
ground truth; chess-like structure (tactics, king safety).

**Q: Why not just use Stockfish?** It is our *referee*, not a competitor. The assignment is to build and compare
AI techniques; Stockfish would answer nothing about learning.

**Q: What is the difference between the policy and the value head?** Policy: probability for each move (where to look).
Value: expected outcome of the position (how good it is). PUCT uses the policy to decide which branches to explore
and the value instead of a rollout.

**Q: What is PUCT?** `Q + c * P * sqrt(N) / (1 + n)`: prefer moves with high value (Q), but explore moves the network
likes (P) that have few visits (n).

**Q: Why mask illegal moves?** The network outputs 625 numbers; softmax over only legal moves removes the need to learn
the rules.

**Q: Why does MCTS lose so badly to alpha-beta here?** Random rollouts are too noisy for a tactical game, and 400
simulations is little. Check the scaling plot: it improves with budget but stays far below at equal budget.

**Q: Did the GA/TDLeaf actually learn chess knowledge, or just overfit?** Be exact. Learned piece values (in pawns): TDLeaf knight 2.4,
bishop 2.2, rook 3.8, queen 4.9 (hand-crafted 3, 3.2, 5, 9); unconstrained GA 1.4, 1.3, 1.9, 2.6. The ordering is right, the scale is
compressed. A possible reason is that they were tuned at a shallow budget (about depth 1.8 at 400 nodes, mean over our E4 positions),
so part of the weights may correct for shallow search; we did not test that. Evidence it is real and not an artefact: E2 (Fairy-Stockfish)
and E4 (move quality) rank them the same as the round robin, and the GA at 2000 nodes is at 0.48 against hand-crafted, i.e. no hidden edge.

**Q: Is TDLeaf better than the hand-crafted evaluation?** No. Starting from equal weights it gains +272 Elo, ends at -24 Elo
[-54, +7] relative to hand-crafted, i.e. close but slightly lower (score 0.44), and it is a bit lower against Fairy-Stockfish at shallow depth.
Do not say "TDLeaf matches it exactly" and do not say it has lower move loss: E4 loss is 83 cp vs 73 cp for hand-crafted.

**Q: Why is the GA RL or not?** Not RL: selection on match results, no value function and no per-move credit assignment. It is
evolutionary optimisation, using game results as a black-box fitness. We include it because it is the natural non-RL way to learn
the same 14 weights, which makes the comparison with TDLeaf clean.

**Q: Why is the AlphaZero-style agent behind?** Small data (12,800 games), a 216k-parameter MLP, 150 simulations per move in
training. A tuned 14-feature evaluation plus exact tactics from quiescence search is a stronger evaluator than this network. Its losses
decrease early and then plateau (policy loss 1.37); its scaling curve is the steepest (-199 to +70 Elo from 25 to 1600 simulations), so
it is not hopeless, but each simulation costs about 12 alpha-beta nodes. At equal wall-clock (6300 nodes vs 400 simulations) the network
scores 0.25 (W12 D36 L72). We do not claim anything about AlphaZero at scale.

**Q: Why does the E3 curve not reach 0.5 for everything?** TDLeaf plateaus near 0.45, GA near 0.55, AlphaZero-style near 0.41. None is
shown to be better than hand-crafted.

**Q: What would you do with more time?** Several seeds and confidence bands per learner; fitness against a hall of fame
or against Fairy-Stockfish for the GA; value targets from search values (reduce noise) and a convolutional network for
AlphaZero; larger variants (Los Alamos 6x6, Silverman 4x5) and transfer between them.

**Q: Is your result new?** Not as an algorithm. The contribution is the controlled, compute-matched comparison with an
external anchor and a move-quality metric on a solved game. Do not claim more.

---------------------------------------------------------------------------------------------------

## 6b. The harness bug we found (tell this story)

In an earlier version both sides in a game shared one transposition table (cache of searched positions), so one agent could reuse scores
computed with the other agent's weights. That made the GA look strong (0.74 against hand-crafted at 2000 nodes) and the AlphaZero agent look best.
The external check with Fairy-Stockfish disagreed (an agent that beat hand-crafted head to head scored zero against the engine from depth 4),
which exposed it. We gave each side its own table, cleared tables between independent queries, retrained all learners and reran every experiment.
Now the GA is at 0.48 at 2000 nodes. Lesson: an external reference is essential. The old data is kept in `results/_superseded_shared_tt/`.

---------------------------------------------------------------------------------------------------

## 6. How to run the demo in the viva

```
python -m web.app          # open http://localhost:5000
```
* **Play** tab: choose Human vs AlphaZero-style, budget 400. The green arrows are the AlphaZero search's visit
  distribution (what it "wants" to play). Switch both sides to agents to watch AI vs AI.
* **RL training explorer** tab: shows the same position evaluated by checkpoints from early to late training.
* **Experiments** tab: all figures and the Elo table.


---------------------------------------------------------------------------------------------------

## 7. Headline numbers [R] (E1, B = 400; Elo relative to hand-crafted; E4 loss against Fairy-Stockfish depth 9)

| Agent | Elo [95% CI] | score vs hand | ms/move | E4 loss (cp) | blunders |
|---|---|---|---|---|---|
| Alpha-beta + GA (constrained) | +23 [-10, +53] | 0.53 | 0.2 | 61 | 9% |
| Alpha-beta + GA | +7 [-24, +37] | 0.53 | 0.19 | 66 | 10% |
| Alpha-beta, hand-crafted | 0 | - | 0.19 | 73 | 11% |
| Alpha-beta + TDLeaf (RL) | -24 [-54, +7] | 0.44 | 0.2 | 83 | 13% |
| AlphaZero-style (RL) | -90 [-123, -63] | 0.43 | 2.3 | 143 | 21% |
| Alpha-beta, equal weights | -296 [-333, -265] | 0.13 | 0.2 | 125 | 19% |
| UCT MCTS | -598 [-643, -558] | 0.02 | 5.7 | 386 | 50% |
| Greedy 1-ply | -744 [-797, -700] | 0.01 | 0.013 | 260 | 34% |
| Random | -973 [-1029, -921] | 0.00 | 0.009 | 612 | 71% |
