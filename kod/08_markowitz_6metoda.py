"""
08_markowitz_6metoda.py
-----------------------
Prosireno Markowitz OOS poredjenje na BACKFILL dnevnim podacima, sa SVIH PET
metoda ciscenja iz rada Bun-Bouchaud-Potters (Risk, april 2016) + empirijska
+ 1/N. RIE je implementiran ispravno po Box 1 rada (rmt_metode.rie);
pyRMT-ova verzija je zadrzana u poredjenju da se kvantifikuje efekat
njegovog buga u stieltjes() funkciji.

Novine u odnosu na 03/06:
  - metode: LW-basic, LW-advanced, clipped, substitution, RIE(Box1), RIE(pyRMT)
  - dva prozora procene: L=60 (kao ranije) i L=1000 (protokol iz rada)
  - robusnost: varijante bez SI=F i bez perioda kraha srebra (15.01-15.02.2026)

Cita:  podaci_ulaz/dnevni_backfill/_spojeno_dnevni.csv
Pise:  rezultati_izlaz/markowitz_6metoda.txt
       rezultati_izlaz/slike/markowitz_6metoda.png
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

ROOT = Path(__file__).resolve().parent.parent
KOD = Path(__file__).resolve().parent
CSV = ROOT / "podaci_ulaz" / "dnevni_backfill" / "_spojeno_dnevni.csv"
OUT_TXT = ROOT / "rezultati_izlaz" / "markowitz_6metoda.txt"
OUT_FIG = ROOT / "rezultati_izlaz" / "slike" / "markowitz_6metoda.png"

sys.path.insert(0, str(KOD.resolve()))
sys.path.insert(0, str((KOD / "pyRMT").resolve()))
import rmt_metode as rm  # noqa: E402
import pyRMT             # noqa: E402

ETF_UNIVERSE = ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
                "XLU", "XLV", "XLY", "EWA", "EWC", "EWG", "EWH", "EWJ", "EWT",
                "EWW", "EWY", "IEF", "SHY", "TLT"]

METHODS = {
    "Empirijska":   rm.empirical,
    "LW-basic":     rm.linear_basic,
    "LW-advanced":  rm.linear_advanced,
    "clipped":      rm.clipped,
    "substitution": rm.substitution,
    "RIE(Box1)":    rm.rie,
    "RIE(pyRMT)":   lambda X: pyRMT.optimalShrinkage(X, method="rie"),
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def load_panel(universe=None, drop_tickers=None, drop_period=None):
    d = pd.read_csv(CSV, parse_dates=["Datetime"])
    piv = d.assign(_i=d.Datetime.dt.date).pivot_table(
        index="_i", columns="Ticker", values="Close")
    if universe is not None:
        piv = piv[[c for c in universe if c in piv.columns]]
    if drop_tickers:
        piv = piv.drop(columns=[t for t in drop_tickers if t in piv.columns])
    if drop_period:
        a, b = pd.Timestamp(drop_period[0]).date(), pd.Timestamp(drop_period[1]).date()
        piv = piv[(piv.index < a) | (piv.index > b)]
    return np.log(piv).diff().dropna(how="any")


def gmv_weights(C):
    ones = np.ones(C.shape[0])
    inv1 = np.linalg.solve(C, ones)
    return inv1 / (ones @ inv1)


def spectrum_sanity():
    ret = load_panel(None)
    X = ret.values
    T, N = X.shape
    q = N / T
    lmin, lmax = rm.mp_edges(q)
    print(f"SPEKTAR (pun panel): T={T}, N={N}, q={q:.4f} | MP=[{lmin:.3f}, {lmax:.3f}]")
    print(f"{'metoda':13} {'cond':>10} {'min_eig':>9}  {'PSD':>5}  {'diag~1':>6}")
    print("-" * 50)
    for name, fn in METHODS.items():
        M = fn(X)
        w = np.linalg.eigvalsh(M)
        print(f"{name:13} {w[-1]/max(w[0],1e-12):10.1f} {w[0]:+9.4f}  "
              f"{str(w[0] > -1e-8):>5}  {str(np.allclose(np.diag(M), 1, atol=1e-6)):>6}")
    print()


def run(naslov, universe, L, drop_tickers=None, drop_period=None):
    ret = load_panel(universe, drop_tickers, drop_period)
    R = ret.values
    T, N = R.shape
    print(f"\n{'='*72}\n{naslov}: T={T}, N={N}, q=N/L={N/L:.2f} | L={L} | OOS dana={T-L}\n{'='*72}")
    oos = {m: [] for m in METHODS}
    pred = {m: [] for m in METHODS}
    lev = {m: [] for m in METHODS}
    cond = {m: [] for m in METHODS}
    oos["1/N"] = []
    skip = 0
    for i, t in enumerate(range(L, T)):
        if i % 250 == 0:
            print(f"    ... prozor {i}/{T-L}", file=sys.__stderr__, flush=True)
        win = R[t - L:t]
        mu, sd = win.mean(0), win.std(0)
        if (sd < 1e-10).any():
            skip += 1
            continue
        z_next = (R[t] - mu) / sd
        try:
            tmp = {}
            for m, fn in METHODS.items():
                C = fn(win)
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
    print("-" * 72)
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
    print("-" * 72)
    print("OOS vol = realizovana vol normalizovanog GMV portfolija (nizi = bolji).")
    for m, rv in rows:
        if m != "Empirijska":
            chg = (rv / emp - 1) * 100
            smer = "nizi" if chg < 0 else "VISI"
            print(f"  {m:<12}: {chg:+6.1f}% rizika ({smer}) vs empirijska")
    return rows


def main():
    spectrum_sanity()

    grid = {}
    grid[("Svih 40", 60)] = run("Svih 40 (FX+futures+ETF), L=60", None, 60)
    grid[("Svih 40", 1000)] = run("Svih 40 (FX+futures+ETF), L=1000 (protokol rada)", None, 1000)
    grid[("ETF (23)", 60)] = run("ETF/akcije (23), L=60", ETF_UNIVERSE, 60)
    grid[("ETF (23)", 1000)] = run("ETF/akcije (23), L=1000 (protokol rada)", ETF_UNIVERSE, 1000)

    print(f"\n\n{'#'*72}\nROBUSNOST NA KRAH SREBRA (svih 40, L=60)\n{'#'*72}")
    run("Svih 40 BEZ SI=F (srebro iskljuceno), L=60", None, 60, drop_tickers=["SI=F"])
    run("Svih 40 BEZ perioda 15.01-15.02.2026, L=60", None, 60,
        drop_period=("2026-01-15", "2026-02-15"))

    # ---- slika: 2x2 grid ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ax, ((uni, L), rows) in zip(axes.ravel(), grid.items()):
        names = [r[0] for r in rows]; vols = [r[1] for r in rows]
        colors = ["#E45756" if n == "Empirijska"
                  else "#999999" if n == "1/N"
                  else "#B5A642" if n == "RIE(pyRMT)"
                  else "#4C78A8" for n in names]
        bars = ax.bar(names, vols, color=colors, edgecolor="white")
        for b, n in zip(bars, names):
            if n == "RIE(pyRMT)":
                b.set_hatch("//")
        ax.set_title(f"{uni}, L={L}")
        ax.set_ylabel("Realizovana OOS vol (normalizovano)")
        ax.tick_params(axis="x", rotation=30)
        for b, v in zip(bars, vols):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)
    plt.suptitle("OOS rizik GMV portfolija: svih 5 metoda iz rada + empirijska + 1/N\n"
                 "(RIE(pyRMT) srafirano: biblioteka sa greskom u Stieltjes transformaciji)",
                 y=1.00)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"\nSlika sacuvana: {OUT_FIG.resolve()}")


if __name__ == "__main__":
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w") as _f:
        _old = sys.stdout
        sys.stdout = _Tee(_old, _f)
        try:
            main()
        finally:
            sys.stdout = _old
    print(f"Izvestaj sacuvan: {OUT_TXT.resolve()}")
