"""Builds report/main.pdf. Every number in the tables comes from results/data/*.json.
    python report/build_report.py
"""
import json
import numpy as np
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "results", "data")
FIG = os.path.join(ROOT, "results", "figures")
sys.path.insert(0, HERE)
from refs import REFS  # noqa: E402


def load(n):
    p = os.path.join(DATA, n + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


SHORT = {"Random": "Random", "Greedy-1ply": "Greedy 1-ply", "AB hand-crafted": "Alpha-beta, hand-crafted",
         "AB equal weights (TDLeaf start)": "Alpha-beta, equal weights",
         "MCTS (random rollouts)": "UCT MCTS", "AB + TDLeaf (RL)": "Alpha-beta + TDLeaf (RL)",
         "AB + GA (evolution)": "Alpha-beta + GA", "AB + GA (constrained)": "Alpha-beta + GA (constr.)",
         "AlphaZero-style (RL)": "AlphaZero-style (RL)"}


def table_e1():
    e1, tm, e4 = load("e1"), load("timing"), load("e4")
    names = e1["names"]
    order = sorted(range(len(names)), key=lambda i: -e1["elo"][i])
    ref = "AB hand-crafted"
    rows = []
    for i in order:
        n = names[i]
        key = "%s|%s" % (ref, n)
        key2 = "%s|%s" % (n, ref)
        if n == ref:
            sc = "--"
        elif key in e1["pairs"]:
            sc = "%.2f" % (1 - e1["pairs"][key][0])
        else:
            sc = "%.2f" % e1["pairs"][key2][0]
        ms = tm["ms_per_move"].get(n, float("nan")) if tm else float("nan")
        rows.append("%s & %+.0f [%+.0f, %+.0f] & %s & %.1f \\\\" % (SHORT[n], e1["elo"][i], e1["lo"][i], e1["hi"][i], sc, ms))
    return "\n".join(rows) + "\n"


def table_e4():
    e4 = load("e4")
    r = e4["results"]
    order = sorted(r, key=lambda n: r[n]["mean_loss"])
    return "\n".join("%s & %.0f & %.0f & %.0f \\\\" % (SHORT[n], r[n]["mean_loss"], 100 * r[n]["best_move_rate"],
                                                      100 * r[n]["blunder_rate_200"]) for n in order)


H, Q, T, G, C, M, Z, R, GR = ("AB hand-crafted", "AB equal weights (TDLeaf start)", "AB + TDLeaf (RL)", "AB + GA (evolution)",
                              "AB + GA (constrained)", "MCTS (random rollouts)", "AlphaZero-style (RL)", "Random", "Greedy-1ply")


def numbers():
    e1, e2, e3, e4, tm, sc, wt, eq = (load(n) for n in ("e1", "e2", "e3", "e4", "timing", "scaling", "weights", "equaltime"))
    names = e1["names"]
    idx = {n: i for i, n in enumerate(names)}
    v = {}

    def S(a, b):                      # score of a against b
        k = "%s|%s" % (a, b)
        return e1["pairs"][k][0] if k in e1["pairs"] else 1 - e1["pairs"]["%s|%s" % (b, a)][0]

    def WDL(a, b):
        k = "%s|%s" % (a, b)
        if k in e1["pairs"]:
            _, w, d, l = e1["pairs"][k]
        else:
            _, l, d, w = e1["pairs"]["%s|%s" % (b, a)]
        return "W%d D%d L%d" % (w, d, l)

    f2 = lambda x: ("%.2f" % x)
    sg = lambda x: (("%+.0f" % x).replace("-", "$-$"))
    for tag, n in (("Z", Z), ("M", M), ("G", G), ("C", C), ("T", T), ("Q", Q), ("R", R), ("GR", GR)):
        i = idx[n]
        v["elo_" + tag], v["lo_" + tag], v["hi_" + tag] = sg(e1["elo"][i]), sg(e1["lo"][i]), sg(e1["hi"][i])
    for a, b in ((Z, H), (G, H), (G, T), (G, Q), (G, Z), (G, M), (M, H), (Q, H)):
        t = {Z: "Z", H: "H", G: "G", T: "T", Q: "Q", M: "M"}
        v["sc_%s%s" % (t[a], t[b])] = f2(S(a, b))
    v["wdl_ZH"] = WDL(Z, H)
    close = [Q, T, C]
    v["band"] = "%.0f" % max(abs(e1["elo"][idx[n]]) for n in close)
    v["ci"] = "%.0f" % np.mean([(e1["hi"][idx[n]] - e1["lo"][idx[n]]) / 2 for n in close + [H] if n != H])
    v["mcts_max"] = f2(max(S(M, n) for n in (H, Q, T, C, Z)))
    # timing / equal time
    v["ms_Z"], v["ms_H"], v["ms_M"] = "%.1f" % tm["ms_per_move"][Z], "%.2f" % tm["ms_per_move"][H], "%.1f" % tm["ms_per_move"][M]
    v["cost_ratio"] = "%.0f" % (tm["ms_per_move"][Z] / tm["ms_per_move"][H])
    r = {x["agent"]: x for x in eq["rows"]}
    v["eqt_budget"] = "{:,}".format(int(round(r[Z]["ab_budget"], -2))).replace(",", "\\,")
    v["eqt_score"], v["eqt_wdl"] = f2(r[Z]["score"]), "W%d D%d L%d" % (r[Z]["w"], r[Z]["d"], r[Z]["l"])
    v["eqt_m_score"] = f2(r[M]["score"])
    v["depth_400"], v["depth_eq"] = "%.1f" % eq["depth_400"], "%.1f" % eq["depth_eq"]
    # external
    ds = e2["depths"]
    top = [H, G, C]
    sc2 = lambda n, d: e2["results"][n][str(d)][0]
    v["ext_top_low_min"] = f2(min(sc2(n, d) for n in top for d in (1, 2)))
    v["ext_top_low_max"] = f2(max(sc2(n, d) for n in top for d in (1, 2)))
    v["ext_top_high_min"] = f2(min(sc2(n, d) for n in top for d in (10, 12)))
    v["ext_top_high_max"] = f2(max(sc2(n, d) for n in top for d in (10, 12)))
    v["ext_bad_max"] = f2(max(sc2(n, 1) for n in (M, GR)))
    v["ext_bad_max4"] = f2(max(sc2(n, d) for n in (G, M, GR) for d in (4, 6, 8, 10, 12)))
    avg = lambda n: np.mean([sc2(n, d) for d in (1, 2, 3, 4)])
    v["ext_eq"], v["ext_td"], v["ext_hand"] = f2(avg(Q)), f2(avg(T)), f2(avg(H))
    v["ext_az"], v["ext_c"], v["ext_g"] = f2(avg(Z)), f2(avg(C)), f2(avg(G))
    # learning curves
    td = [s for _, s in e3[T]]
    v["td_min"], v["td_max"] = f2(min(td[1:])), f2(max(td[1:]))
    az = [s for _, s in e3[Z]]
    v["az_first"], v["az_plateau"] = f2(az[0]), f2(np.mean(az[-6:]))
    ga = [s for _, s in e3[G]]
    v["ga_plateau"] = f2(np.mean(ga[-10:]))
    # E4
    for tag, n in (("H", H), ("Q", Q), ("T", T), ("Z", Z), ("G", G), ("C", C), ("M", M), ("R", R), ("GR", GR)):
        v["loss_" + tag] = "%.0f" % e4["results"][n]["mean_loss"]
        v["bl_" + tag] = "%.0f" % (100 * e4["results"][n]["blunder_rate_200"])
    for tag, k in (("N", 1), ("B", 2), ("R", 3), ("Q", 4)):
        v["tv_" + tag] = "%.1f" % (wt["TDLeaf"][k] / wt["TDLeaf"][0])
        v["gv_" + tag] = "%.1f" % (wt["GA"][k] / wt["GA"][0])
        v["cv_" + tag] = "%.1f" % (wt["GA-constrained"][k] / wt["GA-constrained"][0])
    v["gv_R"] = "%.1f" % (wt["GA"][3] / wt["GA"][0])
    v["gv_mob"] = "%+.0f" % wt["GA"][12]
    z = sc["curves"][Z]
    v["scal_Z0"], v["scal_Z1"] = sg(z[0]["elo"]), sg(z[-1]["elo"])
    v["scal_Mmax"] = f2(max(x["score"] for x in sc["curves"][M]))
    v["scal_G1600"] = f2(sc["curves"][G][-1]["score"])
    # additional keys used by the template
    four = [H, T, G, C]
    el = [e1["elo"][idx[n]] for n in four]
    v["spread"] = "%.0f" % (max(el) - min(el))
    v["td_gain"] = sg(e1["elo"][idx[T]] - e1["elo"][idx[Q]])
    v["sc_ZH"] = f2(S(Z, H))
    v["wdl_ZH"] = WDL(Z, H)
    v["wdl_HZ"] = WDL(H, Z)
    td_pts = [(g, x) for g, x in e3[T]]
    v["td_first"] = f2(td_pts[0][1])
    v["td_plateau"] = f2(np.mean([x for g, x in td_pts if g >= 2400]))
    gp = [(g, x) for g, x in e3[G]]
    v["ga_first"] = f2(gp[0][1])
    v["ga_plateau"] = f2(np.mean([x for g, x in gp if g >= 4000]))
    rng = np.random.default_rng(0)
    Hl = np.array(e4["results"][H]["losses"])
    d = np.array(e4["results"][Z]["losses"]) - Hl
    bs = [rng.choice(d, len(d)).mean() for _ in range(2000)]
    v["pd_Z"] = "%.0f" % d.mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    v["pd_Z_ci"] = "%.0f to %.0f" % (lo, hi)
    old = os.path.join(ROOT, "results", "_superseded_shared_tt", "ga_log.csv")
    import csv
    rows = list(csv.DictReader(open(os.path.join(ROOT, "results", "ga_log.csv"))))
    v["ga_hand2000"] = "%.2f" % float(rows[-1]["score"])
    return v


def render(text, v):
    def sub(m):
        k = m.group(1)
        if k not in v:
            raise SystemExit("template key missing: " + k)
        return v[k]
    return re.sub(r"<<(\w+)>>", sub, text)


def table_main():
    e1, tm, e4 = load("e1"), load("timing"), load("e4")
    names = e1["names"]
    order = sorted(range(len(names)), key=lambda i: -e1["elo"][i])
    rows = []
    for i in order:
        n = names[i]
        if n == H:
            scr = "--"
        else:
            k = "%s|%s" % (H, n)
            scr = "%.2f" % ((1 - e1["pairs"][k][0]) if k in e1["pairs"] else e1["pairs"]["%s|%s" % (n, H)][0])
        r = e4["results"][n]
        rows.append(SHORT[n] + " & %+.0f [%+.0f, %+.0f] & " % (e1["elo"][i], e1["lo"][i], e1["hi"][i]) + scr
                    + " & %.2g & %.0f & %.0f\\%% \\\\" % (tm["ms_per_move"][n], r["mean_loss"], 100 * r["blunder_rate_200"]))
    return "\n".join(rows).rstrip()[:-2].rstrip() + "\n"


def bibliography(tex):
    keys = []
    for m in re.finditer(r"\\cite\{([^}]*)\}", tex):
        for k in m.group(1).split(","):
            k = k.strip()
            if k and k not in keys:
                keys.append(k)
    missing = [k for k in keys if k not in REFS]
    if missing:
        raise SystemExit("missing references: %s" % missing)
    out = ["\\begin{thebibliography}{%d}" % len(keys)]
    for k in keys:
        out.append("\\bibitem{%s}\n%s" % (k, REFS[k]))
    out.append("\\end{thebibliography}")
    unused = [k for k in REFS if k not in keys]
    if unused:
        print("note: unused references:", unused)
    return "\n\n".join(out)


def main():
    os.chdir(HERE)
    for f in os.listdir(FIG):
        if f.endswith(".pdf"):
            shutil.copy(os.path.join(FIG, f), os.path.join(HERE, f))
    v = numbers()
    for name in ("abstract", "sec_results"):
        tpl = os.path.join(HERE, name + ".tex.in")
        open(os.path.join(HERE, name + ".tex"), "w").write(render(open(tpl).read(), v))
    open("table_main.tex", "w").write(table_main())
    tex = "".join(open(os.path.join(HERE, p)).read() for p in ("main.tex", "abstract.tex", "sec_intro_methods.tex", "sec_results.tex"))
    open("refs.tex", "w").write(bibliography(tex))
    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main.tex"], capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-3000:])
            raise SystemExit("pdflatex failed")
    out = subprocess.run(["pdfinfo", "main.pdf"], capture_output=True, text=True).stdout
    print([l for l in out.splitlines() if l.startswith("Pages")])


if __name__ == "__main__":
    main()
