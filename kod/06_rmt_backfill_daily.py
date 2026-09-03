"""
06_rmt_backfill_daily.py
------------------------
ISTA analiza kao 02 (spektar+MP+ciscenje) i 03 (Markowitz OOS), ali nad
BACKFILL dnevnim podacima iz CSV-a (puna istorija sa yfinance, vidi
00_backfill_daily.py). Ne dira postojece skripte ni rezultate.

Cita:  podaci_ulaz/dnevni_backfill/_spojeno_dnevni.csv
Pise (sve sa sufiksom _backfill):
  rezultati_izlaz/rmt_izvestaj_backfill.txt
  rezultati_izlaz/slike/rmt_spektar_backfill.png
  rezultati_izlaz/markowitz_izvestaj_backfill.txt
  rezultati_izlaz/slike/markowitz_oos_backfill.png

Pokretanje:  .venv/bin/python kod/06_rmt_backfill_daily.py
"""
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
CSV = ROOT / "podaci_ulaz" / "dnevni_backfill" / "_spojeno_dnevni.csv"
PYRMT_DIR = KOD / "pyRMT"
FIG_SPEC = ROOT / "rezultati_izlaz" / "slike" / "rmt_spektar_backfill.png"
TXT_SPEC = ROOT / "rezultati_izlaz" / "rmt_izvestaj_backfill.txt"
FIG_MKW = ROOT / "rezultati_izlaz" / "slike" / "markowitz_oos_backfill.png"
TXT_MKW = ROOT / "rezultati_izlaz" / "markowitz_izvestaj_backfill.txt"
L = 60
ETF_UNIVERSE = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
                "XLU", "XLV", "XLY", "EWA", "EWC", "EWG", "EWH", "EWJ", "EWT",
                "EWW", "EWY", "IEF", "SHY", "TLT"]

sys.path.insert(0, str(PYRMT_DIR.resolve()))
import pyRMT  # noqa: E402


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def load_panel(universe=None):
    d = pd.read_csv(CSV, parse_dates=["Datetime"])
    piv = d.assign(_i=d.Datetime.dt.date).pivot_table(index="_i", columns="Ticker", values="Close")
    if universe is not None:
        piv = piv[[c for c in universe if c in piv.columns]]
    ret = np.log(piv).diff().dropna(how="any")
    return ret


# ---------- DEO 1: spektar (kao 02) ----------
def spectrum():
    ret = load_panel(None)
    X = ret.values
    T, N = X.shape
    q = N / T
    print(f"BACKFILL dnevni | period {ret.index.min()} -> {ret.index.max()}")
    print(f"Matrica prinosa: T={T}, N={N}, q=N/T={q:.4f}")

    Xs = (X - X.mean(0)) / X.std(0)
    E = np.corrcoef(Xs, rowvar=False)
    ev = np.sort(np.linalg.eigvalsh(E))[::-1]
    (lmin, lmax), rho = pyRMT.marcenkoPastur(X)
    n_sig = int((ev > lmax).sum())
    print(f"Marcenko-Pastur: [{lmin:.3f}, {lmax:.3f}]")
    print(f"Top svojstvene vrednosti: {np.round(ev[:6], 3)}")
    print(f"Signal iznad MP ivice: {n_sig}/{N} | trzisni mod = {ev[0]/N*100:.1f}% varijanse")

    E_clip = pyRMT.clipped(X)
    E_rie = pyRMT.optimalShrinkage(X, method="rie")
    for name, M in [("Empirijska", E), ("clipped", E_clip), ("RIE", E_rie)]:
        w = np.linalg.eigvalsh(M)
        print(f"  {name:11}: cond={w[-1]/max(w[0],1e-12):9.1f}  min_eig={w.min():+.4f}  "
              f"PSD={w.min() > -1e-8}  diag~1={np.allclose(np.diag(M), 1, atol=1e-6)}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(ev, bins=50, density=True, color="#4C78A8", alpha=0.75, edgecolor="white",
               label="empirijske")
    xs = np.linspace(lmin, lmax, 400)
    ax[0].plot(xs, [rho(x) for x in xs], "r-", lw=2, label="Marčenko–Pastur (šum)")
    ax[0].axvline(lmax, color="k", ls="--", lw=1)
    ax[0].set_xlim(0, 4)
    ax[0].set_title(f"Spektar BACKFILL (T={T}, N={N}, q={q:.3f})")
    ax[0].set_xlabel("λ"); ax[0].legend()
    ax[1].semilogy(range(1, N + 1), ev, "o-", color="#4C78A8", ms=5)
    ax[1].axhline(lmax, color="r", ls="--", lw=1, label=f"λ+={lmax:.2f}")
    ax[1].axhline(lmin, color="orange", ls="--", lw=1, label=f"λ-={lmin:.2f}")
    ax[1].set_title("Sve svojstvene vrednosti (log)")
    ax[1].set_xlabel("rang"); ax[1].set_ylabel("λ"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_SPEC, dpi=130, bbox_inches="tight")
    print(f"Slika sacuvana: {FIG_SPEC.resolve()}")


# ---------- DEO 2: Markowitz OOS (kao 03) ----------
def corr_sample(Zc): return np.corrcoef(Zc, rowvar=False)
def corr_clipped(Zc): return pyRMT.clipped(Zc)
def corr_rie(Zc): return pyRMT.optimalShrinkage(Zc, method="rie")
def corr_ledoit(Zc): return LedoitWolf(assume_centered=True).fit(Zc).covariance_

METHODS = {"Empirijska": corr_sample, "clipped": corr_clipped,
           "RIE": corr_rie, "Ledoit-Wolf": corr_ledoit}


def gmv_weights(C):
    ones = np.ones(C.shape[0])
    inv1 = np.linalg.solve(C, ones)
    return inv1 / (ones @ inv1)


def run_mkw(universe_name, universe):
    ret = load_panel(universe)
    R = ret.values
    T, N = R.shape
    print(f"\n{'='*64}\n{universe_name}: T={T}, N={N}, q=N/L={N/L:.2f} | L={L} | OOS dana={T-L}\n{'='*64}")
    oos = {m: [] for m in METHODS}; pred = {m: [] for m in METHODS}
    lev = {m: [] for m in METHODS}; cond = {m: [] for m in METHODS}
    oos["1/N"] = []
    skip = 0
    for t in range(L, T):
        win = R[t - L:t]
        mu, sd = win.mean(0), win.std(0)
        # vezani FX (USDAED/USDHKD/USDSAR) na nekom prozoru imaju sd=0 -> deljenje
        # nulom -> degenerisana matrica. Preskoci ceo prozor (za SVE metode jednako).
        if (sd < 1e-10).any():
            skip += 1
            continue
        Zc = (win - mu) / sd
        z_next = (R[t] - mu) / sd
        try:
            tmp = {}
            for m, fn in METHODS.items():
                C = fn(Zc)
                if not np.all(np.isfinite(C)):
                    raise np.linalg.LinAlgError("nije konacna matrica")
                w = gmv_weights(C)
                ev = np.linalg.eigvalsh(C)
                tmp[m] = (float(w @ z_next), float(np.sqrt(max(w @ C @ w, 0.0))),
                          float(np.abs(w).sum()), ev[-1] / max(ev[0], 1e-12))
        except np.linalg.LinAlgError:
            skip += 1
            continue
        for m, (o, p, lv, cd) in tmp.items():
            oos[m].append(o); pred[m].append(p); lev[m].append(lv); cond[m].append(cd)
        oos["1/N"].append(float(np.ones(N) / N @ z_next))
    if skip:
        print(f"(preskoceno {skip} prozora zbog degenerisanih/vezanih serija)")

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
    return rows


def markowitz():
    res = {}
    res["Svih 40 (FX+futures+ETF)"] = run_mkw("Svih 40 (FX+futures+ETF)", None)
    res["ETF/akcije (23)"] = run_mkw("ETF/akcije (23)", ETF_UNIVERSE)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, rows) in zip(axes, res.items()):
        names = [r[0] for r in rows]; vols = [r[1] for r in rows]
        colors = ["#E45756" if n == "Empirijska" else "#4C78A8" if n in METHODS else "#999999"
                  for n in names]
        bars = ax.bar(names, vols, color=colors, edgecolor="white")
        ax.set_title(title); ax.set_ylabel("Realizovana OOS vol (normalizovano)")
        ax.tick_params(axis="x", rotation=20)
        for b, v in zip(bars, vols):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    plt.suptitle("BACKFILL OOS rizik GMV portfolija: empirijska vs RMT-ocisceno", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_MKW, dpi=130, bbox_inches="tight")
    print(f"\nSlika sacuvana: {FIG_MKW.resolve()}")


if __name__ == "__main__":
    TXT_SPEC.parent.mkdir(parents=True, exist_ok=True)
    FIG_SPEC.parent.mkdir(parents=True, exist_ok=True)
    with open(TXT_SPEC, "w") as _f:
        _old = sys.stdout; sys.stdout = _Tee(_old, _f)
        try: spectrum()
        finally: sys.stdout = _old
    with open(TXT_MKW, "w") as _f:
        _old = sys.stdout; sys.stdout = _Tee(_old, _f)
        try: markowitz()
        finally: sys.stdout = _old
    print(f"Izvestaji: {TXT_SPEC.resolve()}\n           {TXT_MKW.resolve()}")
