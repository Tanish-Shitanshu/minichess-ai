"""Interactive Mini-Chess platform.   python -m web.app   ->  http://localhost:5000

 * play against any of the agents (or watch two agents play each other)
 * see the engine's evaluation, principal variation, nodes/depth
 * for MCTS / AlphaZero: the root visit distribution drawn on the board (what the search "wants")
 * Experiments tab: every figure produced by harness.run_all / harness.plots and the raw tables
 * Training tab: pick an AlphaZero checkpoint (iteration N) and see how its policy on the same position changes
"""
import ctypes
import glob
import json
import os
import sys

import numpy as np
from flask import Flask, jsonify, request, send_from_directory

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from minichess._lib import CABOut, CAgentCfg, NF, lib          # noqa: E402
from minichess.agents import HAND_WEIGHTS, MCTSAgent, AZeroAgent  # noqa: E402
from minichess.core import State, move_to_uci                  # noqa: E402
from minichess.specs import build_agent, load_net              # noqa: E402

RES = os.path.join(ROOT, "results")
app = Flask(__name__, static_folder=os.path.join(ROOT, "web", "static"), static_url_path="/static")

REASONS = {1: "checkmate", 2: "stalemate", 3: "50-move rule", 4: "insufficient material",
           5: "threefold repetition", 6: "move limit"}


def agent_table():
    """key -> (label, spec). Built every call so new checkpoints appear without a restart."""
    T = {"random": ("Random", {"kind": "random"}),
         "greedy": ("Greedy 1-ply", {"kind": "greedy"}),
         "ab": ("Alpha-beta (hand-crafted eval)", {"kind": "ab"}),
         "mcts": ("MCTS (random rollouts)", {"kind": "mcts"})}
    for key, f, label in (("tdleaf", "tdleaf_weights.npy", "Alpha-beta + TDLeaf-learned eval (RL)"),
                          ("ga", "ga_weights.npy", "Alpha-beta + GA-evolved eval"),
                          ("gac", "gac_weights.npy", "Alpha-beta + GA (constrained) eval")):
        p = os.path.join(RES, f)
        if os.path.exists(p):
            T[key] = (label, {"kind": "ab", "weights": list(map(float, np.load(p)))})
    final = [p for p in ("azero2_final.npz", "azero_final.npz") if os.path.exists(os.path.join(RES, p))]
    if final:
        T["az"] = ("AlphaZero-style (self-play RL)", {"kind": "az", "net": os.path.join(RES, final[0])})
    for p in sorted(glob.glob(os.path.join(RES, "azero*_iter*.npz"))):
        k = os.path.basename(p)[:-4]
        T["ck:" + k] = ("AlphaZero checkpoint " + k.replace("_iter", " iter "), {"kind": "az", "net": p})
    return T


def state_from(moves):
    s = State.start()
    for u in moves:
        s = s.play(s.from_uci(u))
    return s


def describe(s):
    code, reason = s.status()
    legal = s.legal_moves(underpromo=True)
    lm = []
    for m in legal:
        u = move_to_uci(m)
        lm.append(u)
    return {"fen": s.fen(), "board": s.board(), "legal": lm, "turn": "w" if s.player() > 0 else "b",
            "check": s.in_check(), "result": code, "reason": REASONS.get(reason, "") if code != 0 else "",
            "ply": len(s.hist)}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/agents")
def api_agents():
    T = agent_table()
    return jsonify([{"key": k, "label": v[0]} for k, v in T.items()])


@app.route("/api/state", methods=["POST"])
def api_state():
    return jsonify(describe(state_from(request.json.get("moves", []))))


@app.route("/api/ai", methods=["POST"])
def api_ai():
    j = request.json
    s = state_from(j.get("moves", []))
    if s.is_terminal():
        return jsonify({"error": "game over"}), 400
    T = agent_table()
    label, spec = T[j["agent"]]
    spec = dict(spec)
    spec["budget"] = int(j.get("budget", 400))
    spec["seed"] = int(j.get("seed", 1)) + len(s.hist)
    ag = build_agent(spec)
    info = {"agent": label, "budget": spec["budget"]}
    policy = []
    h = (ctypes.c_uint64 * max(1, len(s.hist)))(*s.hist)
    if spec["kind"] == "ab":
        out = CABOut()
        lib.mc_clear_tt()
        w = (ctypes.c_double * NF)(*(spec.get("weights") or HAND_WEIGHTS))
        lib.mc_alphabeta(ctypes.byref(s.c), w, spec["budget"], 40, 1, h, len(s.hist), ctypes.byref(out))
        mv = out.move
        pv, t = [], s
        for i in range(out.pvlen):
            pv.append(move_to_uci(out.pv[i]))
        info.update(score_cp=out.score, depth=out.depth, nodes=int(out.nodes), pv=pv)
    elif spec["kind"] in ("mcts", "az"):
        moves, visits = (ctypes.c_int * 256)(), (ctypes.c_int * 256)()
        cfg = ag.cfg()
        if spec["kind"] == "az":
            val = ctypes.c_double()
            n = lib.mc_az_policy(ctypes.byref(cfg), ctypes.byref(s.c), moves, visits, ctypes.byref(val))
            info["value"] = val.value          # expected outcome for the side to move, in [-1, 1]
        else:
            n = lib.mc_mcts_policy(ctypes.byref(cfg), ctypes.byref(s.c), moves, visits)
        tot = max(1, sum(visits[i] for i in range(n)))
        policy = sorted(({"uci": move_to_uci(moves[i]), "visits": visits[i], "p": visits[i] / tot} for i in range(n)),
                        key=lambda d: -d["visits"])
        mv = moves[max(range(n), key=lambda i: visits[i])] if n else ag.choose_move(s)
        info["simulations"] = tot
    else:
        mv = ag.choose_move(s)
    return jsonify({"move": move_to_uci(mv), "info": info, "policy": policy[:12]})


@app.route("/api/eval", methods=["POST"])
def api_eval():
    """Static features and hand-crafted evaluation of the current position (for the side to move)."""
    s = state_from(request.json.get("moves", []))
    f = s.features()
    return jsonify({"features": [float(x) for x in f], "eval_cp": int(lib.mc_eval(ctypes.byref(s.c), (ctypes.c_double * NF)(*HAND_WEIGHTS)))})


@app.route("/api/results")
def api_results():
    d = {}
    for p in glob.glob(os.path.join(RES, "data", "*.json")):
        d[os.path.basename(p)[:-5]] = json.load(open(p))
    d["figures"] = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RES, "figures", "*.png")))
    return jsonify(d)


@app.route("/figures/<path:name>")
def figures(name):
    return send_from_directory(os.path.join(RES, "figures"), name)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=False)
