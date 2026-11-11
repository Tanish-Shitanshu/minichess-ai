"""Loads (building it first if necessary) the C++ engine and declares its C API.

The engine is compiled from engine/engine.cpp with g++ the first time it is needed, so the
repository contains no binaries and works on any machine that has g++ (or clang++).
"""
import ctypes
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "engine", "engine.cpp")
LIB = os.path.join(ROOT, "engine", "libminichess.so")
NF = 14  # number of evaluation features


def build(force=False):
    if not force and os.path.exists(LIB) and os.path.getmtime(LIB) >= os.path.getmtime(SRC):
        return LIB
    cxx = os.environ.get("CXX", "g++")
    cmd = [cxx, "-O3", "-march=native", "-std=c++17", "-shared", "-fPIC", "-o", LIB, SRC]
    print("[minichess] building engine:", " ".join(cmd), file=sys.stderr)
    subprocess.check_call(cmd)
    return LIB


class CState(ctypes.Structure):
    _fields_ = [
        ("b", ctypes.c_int8 * 25),
        ("stm", ctypes.c_int8),
        ("halfmove", ctypes.c_int16),
        ("ply", ctypes.c_int16),
        ("hash", ctypes.c_uint64),
    ]


class CABOut(ctypes.Structure):
    _fields_ = [
        ("nodes", ctypes.c_int64),
        ("depth", ctypes.c_int),
        ("score", ctypes.c_int),
        ("move", ctypes.c_int),
        ("pvlen", ctypes.c_int),
        ("pv", ctypes.c_int * 32),
        ("leaf", ctypes.c_double * NF),
    ]


class CAgentCfg(ctypes.Structure):
    _fields_ = [
        ("budget", ctypes.c_int64),
        ("seed", ctypes.c_uint64),
        ("w", ctypes.c_double * NF),
        ("cpuct", ctypes.c_double),
        ("dir_alpha", ctypes.c_double),
        ("dir_eps", ctypes.c_double),
        ("net", ctypes.c_void_p),
        ("type", ctypes.c_int),
        ("maxdepth", ctypes.c_int),
        ("use_tt", ctypes.c_int),
        ("rollout_cap", ctypes.c_int),
        ("rollout_mode", ctypes.c_int),
        ("H", ctypes.c_int),
        ("temp_moves", ctypes.c_int),
        ("rand_open", ctypes.c_int),
    ]


def _load():
    path = build()
    lib = ctypes.CDLL(path)
    P = ctypes.POINTER
    ci, cd, c64, cu64 = ctypes.c_int, ctypes.c_double, ctypes.c_int64, ctypes.c_uint64
    sig = {
        "mc_start": (None, [P(CState)]),
        "mc_from_fen": (ci, [P(CState), ctypes.c_char_p]),
        "mc_to_fen": (ci, [P(CState), ctypes.c_char_p]),
        "mc_legal_moves": (ci, [P(CState), P(ci), ci]),
        "mc_make": (None, [P(CState), ci]),
        "mc_in_check": (ci, [P(CState)]),
        "mc_perft": (cu64, [P(CState), ci, ci]),
        "mc_status": (ci, [P(CState), P(cu64), ci, P(ci)]),
        "mc_features": (None, [P(CState), P(cd)]),
        "mc_eval": (ci, [P(CState), P(cd)]),
        "mc_num_features": (ci, []),
        "mc_hand_weights": (None, [P(cd)]),
        "mc_canonical_board": (None, [P(CState), P(ctypes.c_int8)]),
        "mc_canon_index": (ci, [P(CState), ci]),
        "mc_clear_tt": (None, []),
        "mc_move_to_str": (ci, [ci, ctypes.c_char_p]),
        "mc_str_to_move": (ci, [P(CState), ctypes.c_char_p]),
        "mc_alphabeta": (ci, [P(CState), P(cd), c64, ci, ci, P(cu64), ci, P(CABOut)]),
        "mc_choose": (ci, [P(CAgentCfg), P(CState), P(cu64), ci, P(cd), P(c64), P(ci)]),
        "mc_az_policy": (ci, [P(CAgentCfg), P(CState), P(ci), P(ci), P(cd)]),
        "mc_mcts_policy": (ci, [P(CAgentCfg), P(CState), P(ci), P(ci)]),
        "mc_play_game": (ci, [P(CAgentCfg), P(CAgentCfg), P(ci), ci, ci, P(ci), P(ci), P(ci)]),
        "mc_selfplay": (ci, [P(CAgentCfg), ci, ci, P(ctypes.c_int8), P(ctypes.c_float), P(ctypes.c_int8), P(ctypes.c_float), ci, P(ci)]),
    }
    for name, (res, args) in sig.items():
        f = getattr(lib, name)
        f.restype = res
        f.argtypes = args
    assert lib.mc_num_features() == NF
    return lib


lib = _load()
