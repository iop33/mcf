import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence
collections.Sequence = collections.abc.Sequence

from sklearn.covariance import LedoitWolf

ROOT = Path(__file__).resolve().parent.parent
KOD = Path(__file__).resolve().parent
DATA_CLEAN = ROOT / "rezultati_izlaz" / "ocisceni_podaci"
PYRMT_DIR = KOD / "pyRMT"
OUT_FIG = ROOT / "rezultati_izlaz" / "slike" / "markowitz_oos.png"
OUT_TXT = ROOT / "rezultati_izlaz" / "markowitz_izvestaj.txt"
L = 60
ANN = np.sqrt(252)

ETF_UNIVERSE = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
                "XLU", "XLV", "XLY", "EWA", "EWC", "EWG", "EWH", "EWJ", "EWT",
                "EWW", "EWY", "IEF", "SHY", "TLT"]

sys.path.insert(0, str(PYRMT_DIR.resolve()))
import pyRMT


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def build_returns(universe=None):
    d = pd.read_parquet(DATA_CLEAN / "market_daily_clean.parquet")
    piv = d.assign(_i=d.Datetime.dt.date).pivot_table(index="_i", columns="Ticker", values="Close")
    if universe is not None:
        piv = piv[[c for c in universe if c in piv.columns]]
    return np.log(piv).diff().dropna(how="any")


def corr_sample(Zc):
    return np.corrcoef(Zc, rowvar=False)

def corr_clipped(Zc):
    return pyRMT.clipped(Zc)

def corr_rie(Zc):
    return pyRMT.optimalShrinkage(Zc, method="rie")

def corr_ledoit(Zc):
    return LedoitWolf(assume_centered=True).fit(Zc).covariance_

METHODS = {
    "Empirijska":  corr_sample,
    "clipped":     corr_clipped,
    "RIE":         corr_rie,
    "Ledoit-Wolf": corr_ledoit,
}


def gmv_weights(C):
    ones = np.ones(C.shape[0])
    inv1 = np.linalg.solve(C, ones)
    return inv1 / (ones @ inv1)


def run(universe_name, universe):
    ret = build_returns(universe)
    R = ret.values
    T, N = R.shape
    print(f"\n{'='*64}\n{universe_name}: T={T}, N={N}, q=N/L={N/L:.2f} | L={L} | OOS dana={T-L}\n{'='*64}")

    oos = {m: [] for m in METHODS}
    pred = {m: [] for m in METHODS}
    lev = {m: [] for m in METHODS}
    cond = {m: [] for m in METHODS}
    oos["1/N"] = []

    for t in range(L, T):
        win = R[t - L:t]
        mu, sd = win.mean(0), win.std(0)
        Zc = (win - mu) / sd
        z_next = (R[t] - mu) / sd
        for m, fn in METHODS.items():
            C = fn(Zc)
            w = gmv_weights(C)
            oos[m].append(float(w @ z_next))
            pred[m].append(float(np.sqrt(max(w @ C @ w, 0.0))))
            lev[m].append(float(np.abs(w).sum()))
            ev = np.linalg.eigvalsh(C)
            cond[m].append(ev[-1] / max(ev[0], 1e-12))
        oos["1/N"].append(float(np.ones(N) / N @ z_next))

    print("{:<13} {:>11} {:>11} {:>10} {:>8} {:>9}".format(
        "metoda", "OOS vol", "predvidj.", "real/pred", "leverage", "cond"))
    print("-" * 64)
    rows = []
    for m in list(METHODS) + ["1/N"]:
        rv = np.std(oos[m], ddof=1)
        if m in pred:
            pv = np.mean(pred[m]); ratio = rv / pv
            print(f"{m:<13} {rv:>11.4f} {pv:>11.4f} {ratio:>10.2f} "
                  f"{np.mean(lev[m]):>8.1f} {np.mean(cond[m]):>9.0f}")
        else:
            print(f"{m:<13} {rv:>11.4f} {'-':>11} {'-':>10} {'-':>8} {'-':>9}")
        rows.append((m, rv))
    emp = dict(rows)["Empirijska"]
    print("-" * 64)
    print("OOS vol = realizovana vol normalizovanog GMV portfolija (nizi = bolji).")
    for m, rv in rows:
        if m != "Empirijska":
            chg = (rv / emp - 1) * 100
            smer = "nizi" if chg < 0 else "VISI"
            print(f"  {m:<12}: {chg:+5.1f}% rizika ({smer}) vs empirijska")
    if np.mean(cond["RIE"]) > 1000:
        print("  NAPOMENA: RIE je nestabilan na kratkom uzorku (cond>>1000): jedva")
        print("  regularizuje sumni deo spektra -> ogroman leverage u inverziji.")
    return rows


def main():
    res = {}
    res["Svih 40 (FX+futures+ETF)"] = run("Svih 40 (FX+futures+ETF)", None)
    res["ETF/akcije (23)"] = run("ETF/akcije (23)", ETF_UNIVERSE)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, (title, rows) in zip(axes, res.items()):
        names = [r[0] for r in rows]; vols = [r[1] for r in rows]
        colors = ["#E45756" if n == "Empirijska" else "#4C78A8" if n in METHODS else "#999999"
                  for n in names]
        bars = ax.bar(names, vols, color=colors, edgecolor="white")
        ax.set_title(title); ax.set_ylabel("Realizovana OOS vol (normalizovano)")
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vols):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("Out-of-sample rizik GMV portfolija: empirijska vs RMT-ocisceno", y=1.02)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"\nSlika sacuvana: {OUT_FIG.resolve()}")


if __name__ == "__main__":
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w") as _f:
        _old = sys.stdout
        sys.stdout = _Tee(_old, _f)
        try:
            main()
        finally:
            sys.stdout = _old
    print(f"Izvestaj sacuvan: {OUT_TXT.resolve()}")
