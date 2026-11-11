// engine.cpp -- Gardner 5x5 minichess engine used by the Python package.
//
// Contents (in order):
//   1. Board, Zobrist hashing, attack tables, move generation, make-move
//   2. Game status (mate, stalemate, 50-move rule, insufficient material, repetition)
//   3. Evaluation features  phi(s)  (14 numbers, white minus black)
//   4. Alpha-beta search (iterative deepening, transposition table, move ordering, quiescence)
//   5. UCT Monte-Carlo tree search with random rollouts
//   6. AlphaZero-style network forward pass, PUCT search and self-play data generation
//   7. Agent dispatch, whole-game playing, and the C API used by Python (ctypes)
//
// Conventions
//   * Square index sq = rank*5 + file; rank 0 is White's back rank, file 0 is the a-file.
//   * Pieces are signed: +1..+6 for White, -1..-6 for Black (P,N,B,R,Q,K).
//   * A move is the integer ((from*25 + to)*5 + promo); promo 0=none,1=N,2=B,3=R,4=Q.
//   * Our agents only generate queen promotions ("queen-only"); the rules code can
//     generate all four so that perft can be compared against Fairy-Stockfish.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

typedef uint64_t u64;
typedef int64_t i64;

static const int NSQ = 25;
enum Piece { EMPTY = 0, PAWN = 1, KNIGHT = 2, BISHOP = 3, ROOK = 4, QUEEN = 5, KING = 6 };
static const int MATE = 30000;
static const int INF = 32000;

struct State {
    int8_t b[25];
    int8_t stm;       // +1 White to move, -1 Black to move
    int16_t halfmove; // plies since last capture or pawn move
    int16_t ply;      // plies since the start of the game
    u64 hash;         // Zobrist hash (board + side to move)
};

// ---------------------------------------------------------------------------
// 1. Tables, hashing, attacks, move generation
// ---------------------------------------------------------------------------
static u64 ZP[13][25];
static u64 ZSIDE;
static int8_t KNT[25][8];
static int KNN[25];
static int8_t KGT[25][8];
static int KGN[25];
static int8_t RAYT[25][8][4];
static int RAYN[25][8];
static const int DR[8] = {1, -1, 0, 0, 1, 1, -1, -1};
static const int DF[8] = {0, 0, 1, -1, 1, -1, 1, -1};
static const int PVAL[7] = {0, 100, 300, 320, 500, 900, 20000};

static u64 splitmix(u64& x) {
    u64 z = (x += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

struct Rng {
    u64 s;
    explicit Rng(u64 seed = 1) : s(seed * 0x9E3779B97F4A7C15ULL + 0x1234567ULL) { next(); next(); }
    u64 next() { return splitmix(s); }
    double uni() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    int below(int n) { return (int)(next() % (u64)n); }
};

struct Init {
    Init() {
        u64 seed = 20261112ULL;
        for (int p = 0; p < 13; p++)
            for (int s = 0; s < 25; s++) ZP[p][s] = splitmix(seed);
        ZSIDE = splitmix(seed);
        static const int kn[8][2] = {{1, 2}, {2, 1}, {-1, 2}, {-2, 1}, {1, -2}, {2, -1}, {-1, -2}, {-2, -1}};
        for (int sq = 0; sq < 25; sq++) {
            int r = sq / 5, f = sq % 5;
            KNN[sq] = 0;
            KGN[sq] = 0;
            for (int i = 0; i < 8; i++) {
                int nr = r + kn[i][0], nf = f + kn[i][1];
                if (nr >= 0 && nr < 5 && nf >= 0 && nf < 5) KNT[sq][KNN[sq]++] = nr * 5 + nf;
            }
            for (int d = 0; d < 8; d++) {
                int nr = r + DR[d], nf = f + DF[d];
                if (nr >= 0 && nr < 5 && nf >= 0 && nf < 5) KGT[sq][KGN[sq]++] = nr * 5 + nf;
                RAYN[sq][d] = 0;
                nr = r + DR[d];
                nf = f + DF[d];
                while (nr >= 0 && nr < 5 && nf >= 0 && nf < 5) {
                    RAYT[sq][d][RAYN[sq][d]++] = nr * 5 + nf;
                    nr += DR[d];
                    nf += DF[d];
                }
            }
        }
    }
} g_init;

static u64 compute_hash(const State& s) {
    u64 h = 0;
    for (int i = 0; i < 25; i++)
        if (s.b[i]) h ^= ZP[s.b[i] + 6][i];
    if (s.stm < 0) h ^= ZSIDE;
    return h;
}

static void set_start(State& s) {
    static const int back[5] = {ROOK, KNIGHT, BISHOP, QUEEN, KING};  // a1..e1 = R N B Q K
    memset(&s, 0, sizeof(s));
    for (int f = 0; f < 5; f++) {
        s.b[f] = (int8_t)back[f];
        s.b[5 + f] = PAWN;
        s.b[15 + f] = -PAWN;
        s.b[20 + f] = (int8_t)-back[f];
    }
    s.stm = 1;
    s.halfmove = 0;
    s.ply = 0;
    s.hash = compute_hash(s);
}

// Is square `sq` attacked by a piece of colour `by` (+1 White, -1 Black)?
static inline bool attacked(const int8_t* b, int sq, int by) {
    int r = sq / 5, f = sq % 5;
    int pr = r - by;  // a White pawn on rank r-1 attacks rank r; a Black pawn on rank r+1 does
    if (pr >= 0 && pr < 5) {
        if (f > 0 && b[pr * 5 + f - 1] == by * PAWN) return true;
        if (f < 4 && b[pr * 5 + f + 1] == by * PAWN) return true;
    }
    for (int i = 0; i < KNN[sq]; i++)
        if (b[KNT[sq][i]] == by * KNIGHT) return true;
    for (int i = 0; i < KGN[sq]; i++)
        if (b[KGT[sq][i]] == by * KING) return true;
    for (int d = 0; d < 8; d++) {
        int want = d < 4 ? ROOK : BISHOP;
        for (int k = 0; k < RAYN[sq][d]; k++) {
            int p = b[RAYT[sq][d][k]];
            if (p) {
                if (p == by * want || p == by * QUEEN) return true;
                break;
            }
        }
    }
    return false;
}

static inline int king_square(const State& s, int color) {
    for (int i = 0; i < 25; i++)
        if (s.b[i] == color * KING) return i;
    return -1;
}

static inline bool in_check(const State& s) {
    return attacked(s.b, king_square(s, s.stm), -s.stm);
}

static inline int mk(int from, int to, int promo) { return (from * 25 + to) * 5 + promo; }

// Pseudo-legal moves (may leave own king in check). caps_only = captures + promotions only.
static int gen(const State& s, int* mv, bool caps_only, bool underpromo) {
    int n = 0;
    const int us = s.stm;
    const int last_rank = us > 0 ? 4 : 0;
    for (int sq = 0; sq < 25; sq++) {
        int p = s.b[sq];
        if (p * us <= 0) continue;
        int pt = p * us;
        int r = sq / 5, f = sq % 5;
        if (pt == PAWN) {
            int fwd = sq + 5 * us;
            bool promo = (fwd / 5 == last_rank);
            if (s.b[fwd] == 0) {
                if (promo) {
                    if (underpromo) for (int pr = 1; pr <= 4; pr++) mv[n++] = mk(sq, fwd, pr);
                    else mv[n++] = mk(sq, fwd, 4);
                } else if (!caps_only) {
                    mv[n++] = mk(sq, fwd, 0);
                }
            }
            for (int df = -1; df <= 1; df += 2) {
                int nf = f + df;
                if (nf < 0 || nf > 4) continue;
                int to = fwd + df;
                if (s.b[to] * us < 0) {
                    if (promo) {
                        if (underpromo) for (int pr = 1; pr <= 4; pr++) mv[n++] = mk(sq, to, pr);
                        else mv[n++] = mk(sq, to, 4);
                    } else {
                        mv[n++] = mk(sq, to, 0);
                    }
                }
            }
        } else if (pt == KNIGHT || pt == KING) {
            const int8_t* tg = pt == KNIGHT ? KNT[sq] : KGT[sq];
            int cnt = pt == KNIGHT ? KNN[sq] : KGN[sq];
            for (int i = 0; i < cnt; i++) {
                int to = tg[i];
                int t = s.b[to];
                if (t * us > 0) continue;
                if (caps_only && t == 0) continue;
                mv[n++] = mk(sq, to, 0);
            }
        } else {
            int d0 = (pt == BISHOP) ? 4 : 0;
            int d1 = (pt == ROOK) ? 4 : 8;
            for (int d = d0; d < d1; d++) {
                for (int k = 0; k < RAYN[sq][d]; k++) {
                    int to = RAYT[sq][d][k];
                    int t = s.b[to];
                    if (t == 0) {
                        if (!caps_only) mv[n++] = mk(sq, to, 0);
                        continue;
                    }
                    if (t * us < 0) mv[n++] = mk(sq, to, 0);
                    break;
                }
            }
        }
        (void)r;
    }
    return n;
}

static inline void make(State& s, int m) {
    int pr = m % 5, ft = m / 5, to = ft % 25, fr = ft / 25;
    int8_t pc = s.b[fr], cap = s.b[to];
    s.hash ^= ZP[pc + 6][fr];
    if (cap) s.hash ^= ZP[cap + 6][to];
    int8_t np = pc;
    if (pr) {
        static const int pm[5] = {0, KNIGHT, BISHOP, ROOK, QUEEN};
        np = (int8_t)(s.stm * pm[pr]);
    }
    s.b[fr] = 0;
    s.b[to] = np;
    s.hash ^= ZP[np + 6][to];
    s.halfmove = (cap || abs(pc) == PAWN) ? 0 : (int16_t)(s.halfmove + 1);
    s.stm = (int8_t)-s.stm;
    s.hash ^= ZSIDE;
    s.ply++;
}

static inline bool legal_after(const State& t, int mover) {
    // After `make`, is the mover's king safe?
    return !attacked(t.b, king_square(t, mover), -mover);
}

static int legal_moves(const State& s, int* out, bool underpromo) {
    int mv[256];
    int n = gen(s, mv, false, underpromo), k = 0;
    for (int i = 0; i < n; i++) {
        State t = s;
        make(t, mv[i]);
        if (legal_after(t, s.stm)) out[k++] = mv[i];
    }
    return k;
}

static u64 perft(const State& s, int depth, bool underpromo) {
    int mv[256];
    int n = legal_moves(s, mv, underpromo);
    if (depth <= 1) return (u64)n;
    u64 tot = 0;
    for (int i = 0; i < n; i++) {
        State t = s;
        make(t, mv[i]);
        tot += perft(t, depth - 1, underpromo);
    }
    return tot;
}

// ---------------------------------------------------------------------------
// 2. Game status
// ---------------------------------------------------------------------------
static bool insufficient_material(const State& s) {
    int minor = 0;
    for (int i = 0; i < 25; i++) {
        int p = abs(s.b[i]);
        if (p == 0 || p == KING) continue;
        if (p == KNIGHT || p == BISHOP) minor++;
        else return false;
    }
    return minor <= 1;
}

// Returns 0 ongoing, +1 White wins, -1 Black wins, 2 draw. reason: 1 mate, 2 stalemate,
// 3 fifty-move, 4 insufficient material, 5 threefold repetition.
// hist = hashes of the positions BEFORE s (s itself is not included).
static int game_status(const State& s, const u64* hist, int nhist, int* reason) {
    int mv[256];
    int n = legal_moves(s, mv, false);
    if (n == 0) {
        if (in_check(s)) { if (reason) *reason = 1; return -s.stm; }
        if (reason) *reason = 2;
        return 2;
    }
    if (s.halfmove >= 100) { if (reason) *reason = 3; return 2; }
    if (insufficient_material(s)) { if (reason) *reason = 4; return 2; }
    int cnt = 1;
    for (int i = nhist - 1; i >= 0 && i >= nhist - s.halfmove; i--)
        if (hist[i] == s.hash) cnt++;
    if (cnt >= 3) { if (reason) *reason = 5; return 2; }
    return 0;
}

// ---------------------------------------------------------------------------
// 3. Evaluation features: phi(s), 14 numbers, White minus Black
//    0-4  piece counts P,N,B,R,Q
//    5,6  pawns on their 3rd / 4th relative rank (advancement)
//    7-11 centrality of N,B,R,Q,K  (2 = centre square, 1 = ring, 0 = edge)
//    12   mobility of N,B,R,Q (squares reachable, pseudo-legal)
//    13   king shield (own pieces next to own king)
// ---------------------------------------------------------------------------
static const int NF = 14;

static inline int mobility_of(const State& s, int sq, int pt, int c) {
    int m = 0;
    if (pt == KNIGHT) {
        for (int i = 0; i < KNN[sq]; i++)
            if (s.b[KNT[sq][i]] * c <= 0) m++;
    } else {
        int d0 = (pt == BISHOP) ? 4 : 0;
        int d1 = (pt == ROOK) ? 4 : 8;
        for (int d = d0; d < d1; d++)
            for (int k = 0; k < RAYN[sq][d]; k++) {
                int t = s.b[RAYT[sq][d][k]];
                if (t == 0) { m++; continue; }
                if (t * c < 0) m++;
                break;
            }
    }
    return m;
}

static void features(const State& s, double* f) {
    for (int i = 0; i < NF; i++) f[i] = 0;
    int ksq[2] = {-1, -1};
    for (int sq = 0; sq < 25; sq++) {
        int p = s.b[sq];
        if (!p) continue;
        int c = p > 0 ? 1 : -1, pt = abs(p);
        int r = sq / 5, fl = sq % 5;
        if (pt <= QUEEN) f[pt - 1] += c;
        if (pt == PAWN) {
            int rel = c > 0 ? r : 4 - r;
            if (rel == 2) f[5] += c;
            else if (rel == 3) f[6] += c;
        }
        if (pt >= KNIGHT) {
            int cen = 2 - std::max(abs(r - 2), abs(fl - 2));
            f[7 + (pt - 2)] += c * cen;
        }
        if (pt >= KNIGHT && pt <= QUEEN) f[12] += c * mobility_of(s, sq, pt, c);
        if (pt == KING) ksq[c > 0 ? 0 : 1] = sq;
    }
    for (int ci = 0; ci < 2; ci++) {
        int c = ci == 0 ? 1 : -1;
        if (ksq[ci] < 0) continue;
        int cnt = 0;
        for (int i = 0; i < KGN[ksq[ci]]; i++) {
            int t = s.b[KGT[ksq[ci]][i]];
            if (t * c > 0) cnt++;
        }
        f[13] += c * cnt;
    }
}

// Default hand-crafted weights (centipawn-like units) for the baseline agents.
static const double W_HAND[NF] = {100, 300, 320, 500, 900, 15, 40, 20, 10, 5, 5, -5, 4, 10};

// Evaluation from the point of view of the side to move, as an int.
static inline int eval_stm(const State& s, const double* w) {
    if (insufficient_material(s)) return 0;
    double f[NF];
    features(s, f);
    double e = 0;
    for (int i = 0; i < NF; i++) e += w[i] * f[i];
    int v = (int)std::lrint(e);
    if (v > 20000) v = 20000;
    if (v < -20000) v = -20000;
    return v * s.stm;
}

// ---------------------------------------------------------------------------
// 4. Alpha-beta search
// ---------------------------------------------------------------------------
struct TTE { u64 key; int32_t score; int16_t depth; int8_t flag; int16_t mv; };
static const int TT_BITS = 20;
static thread_local std::vector<TTE> g_tt;

static void tt_clear() {
    if (g_tt.empty()) g_tt.assign((size_t)1 << TT_BITS, TTE{0, 0, -1, 0, -1});
    else std::fill(g_tt.begin(), g_tt.end(), TTE{0, 0, -1, 0, -1});
}

struct Ctx {
    const double* w;
    i64 budget, nodes;
    bool abort, hard, use_tt;
    u64 path[512];
    int pathn;
    int killers[64][2];
    int hist[2][625];
    int pv[64][64];
    int pvlen[64];
};

static inline int score_to_tt(int sc, int ply) {
    if (sc >= MATE - 200) return sc + ply;
    if (sc <= -MATE + 200) return sc - ply;
    return sc;
}
static inline int score_from_tt(int sc, int ply) {
    if (sc >= MATE - 200) return sc - ply;
    if (sc <= -MATE + 200) return sc + ply;
    return sc;
}

struct PathGuard {
    Ctx& c;
    explicit PathGuard(Ctx& cc, u64 h) : c(cc) { c.path[c.pathn++] = h; }
    ~PathGuard() { c.pathn--; }
};

static inline int move_order_score(const Ctx& c, const State& s, int m, int ttmove, int ply) {
    if (m == ttmove) return 10000000;
    int pr = m % 5, ft = m / 5, to = ft % 25, fr = ft / 25;
    int victim = abs(s.b[to]), att = abs(s.b[fr]);
    int sc = 0;
    if (victim) sc = 1000000 + 10 * PVAL[victim] - att;
    else if (pr) sc = 900000;
    else if (ply < 64 && (m == c.killers[ply][0] || m == c.killers[ply][1])) sc = 800000;
    else sc = c.hist[s.stm > 0 ? 0 : 1][fr * 25 + to] % 700000;
    if (victim && pr) sc += 50000;
    return sc;
}

static int qsearch(Ctx& c, const State& s, int alpha, int beta, int ply) {
    if (c.abort) return 0;
    c.nodes++;
    if (c.hard && c.nodes > c.budget) { c.abort = true; return 0; }
    c.pvlen[ply] = 0;
    int stand = eval_stm(s, c.w);
    if (ply >= 60) return stand;
    if (stand >= beta) return stand;
    if (stand > alpha) alpha = stand;
    int mv[256], sc[256];
    int n = gen(s, mv, true, false);
    for (int i = 0; i < n; i++) sc[i] = move_order_score(c, s, mv[i], -1, ply);
    int best = stand;
    for (int i = 0; i < n; i++) {
        int j = i;
        for (int k = i + 1; k < n; k++) if (sc[k] > sc[j]) j = k;
        std::swap(mv[i], mv[j]);
        std::swap(sc[i], sc[j]);
        State t = s;
        make(t, mv[i]);
        if (!legal_after(t, s.stm)) continue;
        int v = -qsearch(c, t, -beta, -alpha, ply + 1);
        if (c.abort) return 0;
        if (v > best) {
            best = v;
            if (v > alpha) {
                alpha = v;
                c.pv[ply][0] = mv[i];
                for (int k = 0; k < c.pvlen[ply + 1]; k++) c.pv[ply][k + 1] = c.pv[ply + 1][k];
                c.pvlen[ply] = c.pvlen[ply + 1] + 1;
                if (alpha >= beta) break;
            }
        }
    }
    return best;
}

static int negamax(Ctx& c, const State& s, int depth, int alpha, int beta, int ply) {
    if (c.abort) return 0;
    c.pvlen[ply] = 0;
    PathGuard pg(c, s.hash);
    if (ply > 0) {
        if (s.halfmove >= 100) return 0;
        int lim = std::max(0, c.pathn - 1 - (int)s.halfmove);
        for (int i = c.pathn - 3; i >= lim; i -= 2)
            if (c.path[i] == s.hash) return 0;
    }
    bool inchk = in_check(s);
    if (inchk && ply < 50) depth++;
    if (depth <= 0) return qsearch(c, s, alpha, beta, ply);
    c.nodes++;
    if (c.hard && c.nodes > c.budget) { c.abort = true; return 0; }

    int ttmove = -1;
    TTE* te = nullptr;
    if (c.use_tt) {
        te = &g_tt[s.hash & (((u64)1 << TT_BITS) - 1)];
        if (te->key == s.hash) {
            ttmove = te->mv;
            if (te->depth >= depth && ply > 0) {
                int sc = score_from_tt(te->score, ply);
                if (te->flag == 0) return sc;
                if (te->flag == 1 && sc >= beta) return sc;
                if (te->flag == 2 && sc <= alpha) return sc;
            }
        }
    }
    if (ply >= 60) return eval_stm(s, c.w);

    int mv[256], sc[256];
    int n = gen(s, mv, false, false);
    for (int i = 0; i < n; i++) sc[i] = move_order_score(c, s, mv[i], ttmove, ply);
    int best = -INF, bestmove = -1, legal = 0, orig_alpha = alpha;
    for (int i = 0; i < n; i++) {
        int j = i;
        for (int k = i + 1; k < n; k++) if (sc[k] > sc[j]) j = k;
        std::swap(mv[i], mv[j]);
        std::swap(sc[i], sc[j]);
        int m = mv[i];
        State t = s;
        make(t, m);
        if (!legal_after(t, s.stm)) continue;
        legal++;
        int v = -negamax(c, t, depth - 1, -beta, -alpha, ply + 1);
        if (c.abort) return 0;
        if (v > best) {
            best = v;
            bestmove = m;
            if (v > alpha) {
                alpha = v;
                c.pv[ply][0] = m;
                for (int k = 0; k < c.pvlen[ply + 1]; k++) c.pv[ply][k + 1] = c.pv[ply + 1][k];
                c.pvlen[ply] = c.pvlen[ply + 1] + 1;
                if (alpha >= beta) {
                    int to = (m / 5) % 25, fr = (m / 5) / 25;
                    if (s.b[to] == 0 && m % 5 == 0 && ply < 64) {
                        c.killers[ply][1] = c.killers[ply][0];
                        c.killers[ply][0] = m;
                        int& h = c.hist[s.stm > 0 ? 0 : 1][fr * 25 + to];
                        h = std::min(h + depth * depth, 600000);
                    }
                    break;
                }
            }
        }
    }
    if (legal == 0) return inchk ? -(MATE - ply) : 0;
    if (te) {
        int flag = best <= orig_alpha ? 2 : (best >= beta ? 1 : 0);
        if (te->key != s.hash || te->depth <= depth) {
            te->key = s.hash;
            te->score = score_to_tt(best, ply);
            te->depth = (int16_t)depth;
            te->flag = (int8_t)flag;
            te->mv = (int16_t)bestmove;
        }
    }
    return best;
}

struct ABOut {
    i64 nodes;
    int depth;
    int score;      // centipawn-like, from the point of view of the side to move
    int move;
    int pvlen;
    int pv[32];
    double leaf[NF];  // features (White minus Black) of the principal-variation leaf
};

static int alphabeta_root(const State& root, const double* w, i64 budget, int maxdepth, bool use_tt,
                          const u64* hist, int nhist, ABOut* out) {
    static thread_local Ctx* cp = nullptr;
    if (!cp) cp = new Ctx();
    Ctx& c = *cp;
    c.w = w;
    c.budget = budget;
    c.nodes = 0;
    c.abort = false;
    c.use_tt = use_tt;
    c.pathn = 0;
    memset(c.killers, 0, sizeof(c.killers));
    memset(c.hist, 0, sizeof(c.hist));
    for (int i = 0; i < nhist && i < 400; i++) c.path[c.pathn++] = hist[i];
    if (use_tt && g_tt.empty()) tt_clear();

    int mv[256];
    int nl = legal_moves(root, mv, false);
    int qn = 0;
    int qm[256];
    {
        int tmp[256];
        int n2 = legal_moves(root, tmp, false);
        for (int i = 0; i < n2; i++) {
            int pr = tmp[i] % 5;
            if (pr == 0 || pr == 4) qm[qn++] = tmp[i];
        }
    }
    (void)nl;
    out->nodes = 0; out->depth = 0; out->score = 0; out->move = -1; out->pvlen = 0;
    for (int i = 0; i < NF; i++) out->leaf[i] = 0;
    if (qn == 0) return -1;
    int bestmove = qm[0], bestscore = 0, bestdepth = 0, bestpvlen = 0;
    int bestpv[64];
    bestpv[0] = bestmove;
    bestpvlen = 1;
    if (qn == 1) {
        out->move = qm[0];
        out->pvlen = 1;
        out->pv[0] = qm[0];
        State t = root;
        make(t, qm[0]);
        features(t, out->leaf);
        out->score = 0;
        return qm[0];
    }
    for (int d = 1; d <= maxdepth; d++) {
        c.hard = d > 1;
        c.pathn = std::min(nhist, 400);
        int sc = negamax(c, root, d, -INF, INF, 0);
        if (c.abort) break;
        if (c.pvlen[0] > 0) {
            bestmove = c.pv[0][0];
            bestscore = sc;
            bestdepth = d;
            bestpvlen = std::min(c.pvlen[0], 32);
            for (int k = 0; k < c.pvlen[0] && k < 64; k++) bestpv[k] = c.pv[0][k];
        }
        if (std::abs(sc) >= MATE - 200) break;
        if (c.nodes >= budget) break;
    }
    out->nodes = c.nodes;
    out->depth = bestdepth;
    out->score = bestscore;
    out->move = bestmove;
    out->pvlen = bestpvlen;
    State t = root;
    for (int k = 0; k < bestpvlen; k++) {
        out->pv[k] = bestpv[k];
        make(t, bestpv[k]);
    }
    features(t, out->leaf);
    return bestmove;
}

// ---------------------------------------------------------------------------
// 5. UCT Monte-Carlo tree search with random rollouts
// ---------------------------------------------------------------------------
static int random_legal_move(const State& s, Rng& rng) {
    int mv[256];
    int n = gen(s, mv, false, false);
    while (n > 0) {
        int i = rng.below(n);
        State t = s;
        make(t, mv[i]);
        if (legal_after(t, s.stm)) return mv[i];
        mv[i] = mv[--n];
    }
    return -1;
}

static double material_diff(const State& s) {  // White minus Black, pawn = 1
    static const double v[7] = {0, 1, 3, 3, 5, 9, 0};
    double d = 0;
    for (int i = 0; i < 25; i++) {
        int p = s.b[i];
        if (p) d += (p > 0 ? 1 : -1) * v[abs(p)];
    }
    return d;
}

// Random playout. Returns the result for White in [0,1] (1 = White wins, 0.5 = draw).
// rollout_mode 0: random to the end or to the cap (draw at the cap);
// rollout_mode 1: random to the cap, then a soft material adjudication.
static double rollout(State s, int cap, int mode, Rng& rng) {
    for (int k = 0; k < cap; k++) {
        if (s.halfmove >= 100 || insufficient_material(s)) return 0.5;
        int m = random_legal_move(s, rng);
        if (m < 0) {
            if (in_check(s)) return s.stm > 0 ? 0.0 : 1.0;
            return 0.5;
        }
        make(s, m);
    }
    if (mode == 1) return 0.5 + 0.5 * std::tanh(material_diff(s) / 6.0);
    return 0.5;
}

struct MNode {
    int parent, move;
    int visits;
    double wins;  // reward for the player who made `move` (the player to move at the parent)
    std::vector<int> children;
    std::vector<int> untried;
    bool inited, terminal;
    double term_value;  // reward for the player who made `move`
};

static int mcts_root(const State& root, int sims, double cexp, int cap, int mode, Rng& rng,
                     int* out_moves, int* out_visits, int* out_n) {
    std::vector<MNode> nodes;
    nodes.reserve((size_t)sims + 2);
    nodes.push_back(MNode{-1, -1, 0, 0.0, {}, {}, false, false, 0.0});
    for (int it = 0; it < sims; it++) {
        State s = root;
        int node = 0;
        // selection
        while (true) {
            MNode* nd = &nodes[node];
            if (!nd->inited) {
                nd->inited = true;
                int mv[256];
                int n = legal_moves(s, mv, false);
                int k = 0;
                for (int i = 0; i < n; i++) {
                    int pr = mv[i] % 5;
                    if (pr == 0 || pr == 4) mv[k++] = mv[i];
                }
                for (int i = 0; i < k; i++) nd->untried.push_back(mv[i]);
                if (n == 0) {
                    nd->terminal = true;
                    // reward for the player who made the move into this node
                    nd->term_value = in_check(s) ? 1.0 : 0.5;
                } else if (s.halfmove >= 100 || insufficient_material(s)) {
                    nd->terminal = true;
                    nd->term_value = 0.5;
                }
            }
            if (nd->terminal) break;
            if (!nd->untried.empty()) break;
            // all children tried: pick best UCB1 child
            double bestu = -1e18;
            int bc = -1;
            double lnN = std::log((double)std::max(1, nd->visits));
            for (int ch : nd->children) {
                const MNode& cn = nodes[ch];
                double u = cn.wins / cn.visits + cexp * std::sqrt(lnN / cn.visits);
                if (u > bestu) { bestu = u; bc = ch; }
            }
            // wins are stored for the mover into the child: parent chooses to maximise them
            node = bc;
            make(s, nodes[node].move);
        }
        MNode* nd = &nodes[node];
        double reward_for_mover_into_node;  // from the view of the player who made the move into `node`
        if (nd->terminal) {
            reward_for_mover_into_node = nd->term_value;
        } else {
            // expansion: take a random untried move
            int idx = rng.below((int)nd->untried.size());
            int m = nd->untried[idx];
            nd->untried[idx] = nd->untried.back();
            nd->untried.pop_back();
            make(s, m);
            int mover = -s.stm;  // the player who just moved
            int child = (int)nodes.size();
            nodes.push_back(MNode{node, m, 0, 0.0, {}, {}, false, false, 0.0});
            nodes[node].children.push_back(child);
            node = child;
            double rw = rollout(s, cap, mode, rng);  // for White
            reward_for_mover_into_node = mover > 0 ? rw : 1.0 - rw;
        }
        // backpropagation: alternate perspective
        double r = reward_for_mover_into_node;
        for (int x = node; x >= 0; x = nodes[x].parent) {
            nodes[x].visits++;
            nodes[x].wins += r;
            r = 1.0 - r;
        }
    }
    int best = -1, bv = -1, n = 0;
    for (int ch : nodes[0].children) {
        out_moves[n] = nodes[ch].move;
        out_visits[n] = nodes[ch].visits;
        n++;
        if (nodes[ch].visits > bv) { bv = nodes[ch].visits; best = nodes[ch].move; }
    }
    *out_n = n;
    return best;
}

// ---------------------------------------------------------------------------
// 6. AlphaZero-style policy-value network, PUCT search, self-play
//
// Network (all float32), input = 12 one-hot planes x 25 squares from the point of view
// of the side to move (own pieces first). Layout of the flat weight array:
//   W1[300][H]  b1[H]  W2[H][H]  b2[H]  Wp[625][H]  bp[625]  Wv[H]  bv[1]
// h1 = relu(b1 + sum_active W1[idx]); h2 = relu(b2 + h1 . W2); value = tanh(bv + Wv . h2)
// logit(move) = bp[o] + Wp[o] . h2, o = from_c*25 + to_c (canonical orientation)
// ---------------------------------------------------------------------------
struct Net {
    int H;
    const float *W1, *b1, *W2, *b2, *Wp, *bp, *Wv, *bv;
};

static Net net_from_flat(const float* p, int H) {
    Net n;
    n.H = H;
    n.W1 = p;            p += 300 * H;
    n.b1 = p;            p += H;
    n.W2 = p;            p += H * H;
    n.b2 = p;            p += H;
    n.Wp = p;            p += 625 * H;
    n.bp = p;            p += 625;
    n.Wv = p;            p += H;
    n.bv = p;
    return n;
}

static inline int flip_sq(int sq) { return (4 - sq / 5) * 5 + sq % 5; }

// canonical board: side to move is White ("positive")
static void canonical_board(const State& s, int8_t* cb) {
    if (s.stm > 0) {
        for (int i = 0; i < 25; i++) cb[i] = s.b[i];
    } else {
        for (int i = 0; i < 25; i++) cb[flip_sq(i)] = (int8_t)-s.b[i];
    }
}

static inline int canon_move_index(const State& s, int m) {
    int ft = m / 5, to = ft % 25, fr = ft / 25;
    if (s.stm < 0) { fr = flip_sq(fr); to = flip_sq(to); }
    return fr * 25 + to;
}

// Evaluate: fills priors (softmax over the given legal moves) and returns value in [-1,1]
// from the point of view of the side to move.
static float net_eval(const Net& net, const State& s, const int* moves, int nm, float* priors) {
    const int H = net.H;
    int8_t cb[25];
    canonical_board(s, cb);
    std::vector<float> h1(H), h2(H);
    for (int j = 0; j < H; j++) h1[j] = net.b1[j];
    for (int sq = 0; sq < 25; sq++) {
        int p = cb[sq];
        if (!p) continue;
        int plane = p > 0 ? p - 1 : 6 + (-p - 1);
        const float* row = net.W1 + (size_t)(plane * 25 + sq) * H;
        for (int j = 0; j < H; j++) h1[j] += row[j];
    }
    for (int j = 0; j < H; j++) h1[j] = h1[j] > 0 ? h1[j] : 0;
    for (int j = 0; j < H; j++) h2[j] = net.b2[j];
    for (int i = 0; i < H; i++) {
        float a = h1[i];
        if (a == 0) continue;
        const float* row = net.W2 + (size_t)i * H;
        for (int j = 0; j < H; j++) h2[j] += a * row[j];
    }
    for (int j = 0; j < H; j++) h2[j] = h2[j] > 0 ? h2[j] : 0;
    float v = net.bv[0];
    for (int j = 0; j < H; j++) v += net.Wv[j] * h2[j];
    v = std::tanh(v);
    float mx = -1e30f;
    float logit[256];
    for (int i = 0; i < nm; i++) {
        int o = canon_move_index(s, moves[i]);
        const float* row = net.Wp + (size_t)o * H;
        float a = net.bp[o];
        for (int j = 0; j < H; j++) a += row[j] * h2[j];
        logit[i] = a;
        if (a > mx) mx = a;
    }
    float sum = 0;
    for (int i = 0; i < nm; i++) { logit[i] = std::exp(logit[i] - mx); sum += logit[i]; }
    for (int i = 0; i < nm; i++) priors[i] = logit[i] / sum;
    return v;
}

struct ANode {
    int parent, move;
    int first, nch;
    float prior;
    int N;
    float W;  // total value for the player to move AT this node
    bool expanded, terminal;
    float tv;
};

struct AZParams {
    int sims;
    double cpuct;
    bool noise;
    double dir_alpha, dir_eps;
};

// Runs PUCT. Fills moves/visits for the root children. Returns root value estimate
// (mean value for the side to move).
static float az_search(const Net& net, const State& root, const AZParams& P, Rng& rng,
                       int* out_moves, int* out_visits, int* out_n) {
    std::vector<ANode> nodes;
    nodes.reserve((size_t)P.sims * 12 + 64);
    nodes.push_back(ANode{-1, -1, -1, 0, 1.0f, 0, 0.0f, false, false, 0.0f});
    auto expand = [&](int idx, const State& s) -> float {
        int mv[256];
        int n = legal_moves(s, mv, false);
        int k = 0;
        for (int i = 0; i < n; i++) {
            int pr = mv[i] % 5;
            if (pr == 0 || pr == 4) mv[k++] = mv[i];
        }
        n = k;
        nodes[idx].expanded = true;
        if (n == 0 || s.halfmove >= 100 || insufficient_material(s)) {
            nodes[idx].terminal = true;
            nodes[idx].tv = (n == 0 && in_check(s)) ? -1.0f : 0.0f;
            return nodes[idx].tv;
        }
        float pri[256];
        float v = net_eval(net, s, mv, n, pri);
        int first = (int)nodes.size();
        for (int i = 0; i < n; i++) nodes.push_back(ANode{idx, mv[i], -1, 0, pri[i], 0, 0.0f, false, false, 0.0f});
        nodes[idx].first = first;
        nodes[idx].nch = n;
        return v;
    };
    {
        float v = expand(0, root);
        nodes[0].N = 1;
        nodes[0].W = v;
    }
    if (nodes[0].nch == 0) { *out_n = 0; return 0; }
    if (P.noise) {
        std::mt19937_64 gen64(rng.next());
        std::gamma_distribution<double> gd(P.dir_alpha, 1.0);
        std::vector<double> d(nodes[0].nch);
        double sum = 0;
        for (auto& x : d) { x = gd(gen64); sum += x; }
        for (int i = 0; i < nodes[0].nch; i++) {
            ANode& ch = nodes[nodes[0].first + i];
            ch.prior = (float)((1 - P.dir_eps) * ch.prior + P.dir_eps * d[i] / sum);
        }
    }
    int path[256];
    for (int it = 0; it < P.sims; it++) {
        State s = root;
        int node = 0, pl = 0;
        path[pl++] = 0;
        while (nodes[node].expanded && !nodes[node].terminal && nodes[node].nch > 0) {
            const ANode& nd = nodes[node];
            double sq = std::sqrt((double)nd.N);
            double qparent = nd.N > 0 ? nd.W / nd.N : 0.0;
            double best = -1e18;
            int bc = -1;
            for (int i = 0; i < nd.nch; i++) {
                const ANode& ch = nodes[nd.first + i];
                double q = ch.N > 0 ? -(double)ch.W / ch.N : qparent - 0.15;
                double u = q + P.cpuct * ch.prior * sq / (1.0 + ch.N);
                if (u > best) { best = u; bc = nd.first + i; }
            }
            node = bc;
            make(s, nodes[node].move);
            if (pl < 255) path[pl++] = node;
        }
        float v;
        if (nodes[node].terminal) v = nodes[node].tv;
        else if (!nodes[node].expanded) v = expand(node, s);
        else v = 0;
        for (int i = pl - 1; i >= 0; i--) {
            nodes[path[i]].N++;
            nodes[path[i]].W += v;
            v = -v;
        }
    }
    int n = 0;
    for (int i = 0; i < nodes[0].nch; i++) {
        const ANode& ch = nodes[nodes[0].first + i];
        out_moves[n] = ch.move;
        out_visits[n] = ch.N;
        n++;
    }
    *out_n = n;
    return nodes[0].N > 0 ? nodes[0].W / nodes[0].N : 0.0f;
}

// ---------------------------------------------------------------------------
// 7. Agents, whole-game playing and the C API
// ---------------------------------------------------------------------------
enum AgentType { A_RANDOM = 0, A_ALPHABETA = 1, A_MCTS = 2, A_AZERO = 3, A_GREEDY = 4 };

struct AgentCfg {
    i64 budget;       // nodes (alpha-beta) or simulations (MCTS / AlphaZero)
    u64 seed;
    double w[NF];     // evaluation weights (alpha-beta, greedy)
    double cpuct;     // UCT exploration constant (MCTS) or PUCT constant (AlphaZero)
    double dir_alpha, dir_eps;
    const float* net; // flat AlphaZero weights
    int type, maxdepth, use_tt, rollout_cap, rollout_mode, H, temp_moves, rand_open;
};

struct ChoiceInfo {
    double score;  // alpha-beta: centipawns (side to move); AlphaZero: value in [-1,1]
    i64 nodes;
    int depth;
};

static int choose_move(const AgentCfg& cfg, const State& s, const u64* hist, int nhist, ChoiceInfo* info) {
    Rng rng(cfg.seed ^ (s.hash * 0x9E3779B97F4A7C15ULL) ^ (u64)nhist);
    info->score = 0; info->nodes = 0; info->depth = 0;
    int mv[256];
    int n = legal_moves(s, mv, false);
    int k = 0;
    for (int i = 0; i < n; i++) {
        int pr = mv[i] % 5;
        if (pr == 0 || pr == 4) mv[k++] = mv[i];
    }
    n = k;
    if (n == 0) return -1;
    switch (cfg.type) {
        case A_RANDOM:
            return mv[rng.below(n)];
        case A_GREEDY: {
            int best = mv[0], bs = -1000000;
            for (int i = 0; i < n; i++) {
                State t = s;
                make(t, mv[i]);
                int sc = -eval_stm(t, cfg.w) + rng.below(3);
                if (sc > bs) { bs = sc; best = mv[i]; }
            }
            info->score = bs;
            return best;
        }
        case A_ALPHABETA: {
            ABOut o;
            alphabeta_root(s, cfg.w, cfg.budget, cfg.maxdepth > 0 ? cfg.maxdepth : 40, cfg.use_tt != 0, hist, nhist, &o);
            info->score = o.score; info->nodes = o.nodes; info->depth = o.depth;
            return o.move;
        }
        case A_MCTS: {
            int om[256], ov[256], on = 0;
            int best = mcts_root(s, (int)cfg.budget, cfg.cpuct > 0 ? cfg.cpuct : 0.7, cfg.rollout_cap > 0 ? cfg.rollout_cap : 40,
                                 cfg.rollout_mode, rng, om, ov, &on);
            info->nodes = cfg.budget;
            return best;
        }
        case A_AZERO: {
            Net net = net_from_flat(cfg.net, cfg.H);
            AZParams P{(int)cfg.budget, cfg.cpuct > 0 ? cfg.cpuct : 1.5, false, cfg.dir_alpha, cfg.dir_eps};
            int om[256], ov[256], on = 0;
            float q = az_search(net, s, P, rng, om, ov, &on);
            int best = om[0], bv = -1;
            for (int i = 0; i < on; i++)
                if (ov[i] > bv) { bv = ov[i]; best = om[i]; }
            info->score = q;
            info->nodes = cfg.budget;
            return best;
        }
    }
    return mv[0];
}

// Plays one game. opening = list of moves to play first (not chosen by the agents).
// Returns the result from White's point of view: +1, 0, -1. moves_out receives the moves.
static int play_game_impl(const AgentCfg& white, const AgentCfg& black, const int* opening, int nopen,
                          int max_plies, int* moves_out, int* nmoves, int* reason_out) {
    State s;
    set_start(s);
    std::vector<u64> hist;
    int nm = 0;
    // Each side keeps its OWN transposition table for the whole game. (A shared table would let one agent reuse
    // scores that the other computed with a different evaluation, which contaminates comparisons.)
    std::vector<TTE> tt_side[2];
    for (int i = 0; i < nopen; i++) {
        hist.push_back(s.hash);
        make(s, opening[i]);
        moves_out[nm++] = opening[i];
    }
    int result = 0;
    int reason = 0;
    while (true) {
        int rs = 0;
        int st = game_status(s, hist.data(), (int)hist.size(), &rs);
        if (st != 0) {
            result = st == 2 ? 0 : st;
            reason = rs;
            break;
        }
        if (nm >= max_plies) { result = 0; reason = 6; break; }
        const AgentCfg& cfg = s.stm > 0 ? white : black;
        ChoiceInfo ci;
        const int side = s.stm > 0 ? 0 : 1;
        std::swap(g_tt, tt_side[side]);
        int m = choose_move(cfg, s, hist.data(), (int)hist.size(), &ci);
        std::swap(g_tt, tt_side[side]);
        if (m < 0) { result = 0; break; }
        hist.push_back(s.hash);
        make(s, m);
        moves_out[nm++] = m;
    }
    *nmoves = nm;
    if (reason_out) *reason_out = reason;
    return result;
}

extern "C" {

void mc_start(State* s) { set_start(*s); }

int mc_from_fen(State* s, const char* fen) {
    memset(s, 0, sizeof(*s));
    int r = 4, f = 0;
    const char* p = fen;
    for (; *p && *p != ' '; p++) {
        char ch = *p;
        if (ch == '/') { r--; f = 0; }
        else if (ch >= '1' && ch <= '5') f += ch - '0';
        else {
            int pt = 0;
            switch (ch | 0x20) {
                case 'p': pt = PAWN; break;
                case 'n': pt = KNIGHT; break;
                case 'b': pt = BISHOP; break;
                case 'r': pt = ROOK; break;
                case 'q': pt = QUEEN; break;
                case 'k': pt = KING; break;
                default: return -1;
            }
            if (r < 0 || f > 4) return -1;
            s->b[r * 5 + f] = (int8_t)((ch >= 'a') ? -pt : pt);
            f++;
        }
    }
    if (*p == ' ') p++;
    s->stm = (*p == 'b') ? -1 : 1;
    while (*p && *p != ' ') p++;
    while (*p == ' ') p++;
    while (*p && *p != ' ') p++;  // castling field
    while (*p == ' ') p++;
    while (*p && *p != ' ') p++;  // en passant field
    while (*p == ' ') p++;
    s->halfmove = (int16_t)atoi(p);
    s->ply = 0;
    s->hash = compute_hash(*s);
    return 0;
}

int mc_to_fen(const State* s, char* buf) {
    static const char* sym = " pnbrqk";
    int n = 0;
    for (int r = 4; r >= 0; r--) {
        int empty = 0;
        for (int f = 0; f < 5; f++) {
            int p = s->b[r * 5 + f];
            if (!p) { empty++; continue; }
            if (empty) { buf[n++] = (char)('0' + empty); empty = 0; }
            char ch = sym[abs(p)];
            buf[n++] = p > 0 ? (char)(ch - 32) : ch;
        }
        if (empty) buf[n++] = (char)('0' + empty);
        if (r) buf[n++] = '/';
    }
    n += snprintf(buf + n, 40, " %c - - %d %d", s->stm > 0 ? 'w' : 'b', (int)s->halfmove, (int)(s->ply / 2 + 1));
    return n;
}

int mc_legal_moves(const State* s, int* out, int underpromo) { return legal_moves(*s, out, underpromo != 0); }
void mc_make(State* s, int m) { make(*s, m); }
int mc_in_check(const State* s) { return in_check(*s) ? 1 : 0; }
u64 mc_perft(const State* s, int depth, int underpromo) { return perft(*s, depth, underpromo != 0); }
int mc_status(const State* s, const u64* hist, int nhist, int* reason) { return game_status(*s, hist, nhist, reason); }
void mc_features(const State* s, double* out) { features(*s, out); }
int mc_eval(const State* s, const double* w) { return eval_stm(*s, w); }
int mc_num_features() { return NF; }
void mc_hand_weights(double* out) { for (int i = 0; i < NF; i++) out[i] = W_HAND[i]; }
void mc_canonical_board(const State* s, int8_t* out) { canonical_board(*s, out); }
int mc_canon_index(const State* s, int m) { return canon_move_index(*s, m); }
void mc_clear_tt() { tt_clear(); }

int mc_move_to_str(int m, char* buf) {
    int pr = m % 5, ft = m / 5, to = ft % 25, fr = ft / 25;
    int n = snprintf(buf, 8, "%c%c%c%c", 'a' + fr % 5, '1' + fr / 5, 'a' + to % 5, '1' + to / 5);
    if (pr) buf[n++] = "?nbrq"[pr];
    buf[n] = 0;
    return n;
}

// Parses e.g. "e2e3" or "b4b5q"; returns the move code or -1 if it is not legal.
int mc_str_to_move(const State* s, const char* str) {
    if (!str || strlen(str) < 4) return -1;
    int fr = (str[0] - 'a') + 5 * (str[1] - '1');
    int to = (str[2] - 'a') + 5 * (str[3] - '1');
    if (fr < 0 || fr > 24 || to < 0 || to > 24) return -1;
    int pr = 0;
    if (strlen(str) >= 5) {
        switch (str[4]) {
            case 'n': pr = 1; break;
            case 'b': pr = 2; break;
            case 'r': pr = 3; break;
            case 'q': pr = 4; break;
        }
    }
    int mv[256];
    int n = legal_moves(*s, mv, true);
    for (int i = 0; i < n; i++) {
        int m = mv[i], mp = m % 5, ft = m / 5;
        if (ft % 25 == to && ft / 25 == fr) {
            if (mp == pr || (pr == 0 && mp == 4 && abs(s->b[fr]) == PAWN)) return m;
        }
    }
    return -1;
}

int mc_alphabeta(const State* s, const double* w, i64 budget, int maxdepth, int use_tt, const u64* hist, int nhist, ABOut* out) {
    return alphabeta_root(*s, w, budget, maxdepth, use_tt != 0, hist, nhist, out);
}

int mc_choose(const AgentCfg* cfg, const State* s, const u64* hist, int nhist, double* score, i64* nodes, int* depth) {
    ChoiceInfo ci;
    int m = choose_move(*cfg, *s, hist, nhist, &ci);
    if (score) *score = ci.score;
    if (nodes) *nodes = ci.nodes;
    if (depth) *depth = ci.depth;
    return m;
}

// Visit counts of the AlphaZero search at the root (for the web UI / analysis).
int mc_az_policy(const AgentCfg* cfg, const State* s, int* moves, int* visits, double* value) {
    Rng rng(cfg->seed ^ s->hash);
    Net net = net_from_flat(cfg->net, cfg->H);
    AZParams P{(int)cfg->budget, cfg->cpuct > 0 ? cfg->cpuct : 1.5, false, cfg->dir_alpha, cfg->dir_eps};
    int n = 0;
    float q = az_search(net, *s, P, rng, moves, visits, &n);
    *value = q;
    return n;
}

int mc_mcts_policy(const AgentCfg* cfg, const State* s, int* moves, int* visits) {
    Rng rng(cfg->seed ^ s->hash);
    int n = 0;
    mcts_root(*s, (int)cfg->budget, cfg->cpuct > 0 ? cfg->cpuct : 0.7, cfg->rollout_cap > 0 ? cfg->rollout_cap : 40,
              cfg->rollout_mode, rng, moves, visits, &n);
    return n;
}

int mc_play_game(const AgentCfg* white, const AgentCfg* black, const int* opening, int nopen, int max_plies,
                 int* moves_out, int* nmoves, int* reason) {
    return play_game_impl(*white, *black, opening, nopen, max_plies, moves_out, nmoves, reason);
}

// Self-play for AlphaZero training. Writes positions (canonical board, visit distribution,
// outcome for the side to move, plus a mask of the legal moves) into the caller's buffers; returns the number of positions.
// stats = {white wins, draws, black wins, total plies}.
int mc_selfplay(const AgentCfg* cfg, int ngames, int max_plies, int8_t* boards, float* pols, int8_t* masks,
                float* zs, int max_pos, int* stats) {
    Net net = net_from_flat(cfg->net, cfg->H);
    AZParams P{(int)cfg->budget, cfg->cpuct > 0 ? cfg->cpuct : 1.5, true, cfg->dir_alpha > 0 ? cfg->dir_alpha : 0.3,
               cfg->dir_eps > 0 ? cfg->dir_eps : 0.25};
    Rng rng(cfg->seed);
    int total = 0;
    stats[0] = stats[1] = stats[2] = stats[3] = 0;
    for (int g = 0; g < ngames; g++) {
        State s;
        set_start(s);
        std::vector<u64> hist;
        int start_pos = total;
        std::vector<int8_t> stm_of;
        int k_open = cfg->rand_open > 0 ? rng.below(cfg->rand_open + 1) : 0;
        for (int i = 0; i < k_open; i++) {
            int m = random_legal_move(s, rng);
            if (m < 0) break;
            hist.push_back(s.hash);
            make(s, m);
        }
        int result = 0, plies = 0;
        while (true) {
            int rs = 0;
            int st = game_status(s, hist.data(), (int)hist.size(), &rs);
            if (st != 0) { result = st == 2 ? 0 : st; break; }
            if (plies >= max_plies) { result = 0; break; }
            int om[256], ov[256], on = 0;
            az_search(net, s, P, rng, om, ov, &on);
            double tot = 0;
            for (int i = 0; i < on; i++) tot += ov[i];
            // record training sample
            if (total < max_pos) {
                canonical_board(s, boards + (size_t)total * 25);
                float* pol = pols + (size_t)total * 625;
                memset(pol, 0, sizeof(float) * 625);
                int8_t* mk_ = masks + (size_t)total * 625;
                memset(mk_, 0, 625);
                for (int i = 0; i < on; i++) {
                    int ci_ = canon_move_index(s, om[i]);
                    pol[ci_] += (float)(ov[i] / tot);
                    mk_[ci_] = 1;
                }
                zs[total] = (float)s.stm;  // temporarily the side to move; converted below
                total++;
            }
            int choice = 0;
            if (plies < cfg->temp_moves) {
                double r = rng.uni() * tot, acc = 0;
                for (int i = 0; i < on; i++) { acc += ov[i]; if (r <= acc) { choice = i; break; } }
            } else {
                int bv = -1;
                for (int i = 0; i < on; i++) if (ov[i] > bv) { bv = ov[i]; choice = i; }
            }
            hist.push_back(s.hash);
            make(s, om[choice]);
            plies++;
        }
        for (int i = start_pos; i < total; i++) zs[i] = (float)(zs[i] * result);
        stats[result > 0 ? 0 : (result == 0 ? 1 : 2)]++;
        stats[3] += plies;
    }
    return total;
}

}  // extern "C"
