"""
02_rmt_analysis_minute.py
-------------------------
ISTA analiza kao 02_rmt_analysis.py, ali nad MINUTNIM podacima (ceo niz).
Ne dira postojece skripte ni dnevne rezultate: cita ocisceni minutni panel i
upisuje u NOVE fajlove (sufiks _minute):
  - rezultati_izlaz/rmt_izvestaj_minute.txt
  - rezultati_izlaz/slike/rmt_spektar_minute.png

Napomena (za rad): na celom minutnom nizu T je ogromno pa je q=N/T ~ 0.
MP pojas se suzi oko 1 -> empirijska matrica je vec skoro savrseno procenjena
i 'nema sta da se cisti'. To je ocekivano i samo po sebi rezultat (kontrast
dnevnoj analizi gde je q netrivijalno).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ---- ZAKRPA za pyRMT na Python 3.10+ ----
import collections
import collections.abc
collections.MutableSequence = collections.abc.MutableSequence
collections.Sequence = collections.abc.Sequence

# ---- PODESAVANJA ----
ROOT = Path(__file__).resolve().parent.parent
KOD = Path(__file__).resolve().parent
DATA_CLEAN = ROOT / "rezultati_izlaz" / "ocisceni_podaci"
PYRMT_DIR = KOD / "pyRMT"
FREQ = "minute"
OUT_FIG = ROOT / "rezultati_izlaz" / "slike" / "rmt_spektar_minute.png"
OUT_TXT = ROOT / "rezultati_izlaz" / "rmt_izvestaj_minute.txt"

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


def build_returns(freq):
    fn = "market_daily_clean.parquet" if freq == "daily" else "market_minute_clean.parquet"
    d = pd.read_parquet(DATA_CLEAN / fn)
    idx = d.Datetime.dt.date if freq == "daily" else d.Datetime
    piv = d.assign(_i=idx).pivot_table(index="_i", columns="Ticker", values="Close")
    # MINUTNI: klase aktiva ne trguju u isto vreme (FX 24h vs ETF RTH), pa
    # presek 'svih 40 u istom minutu' bude prazan. Standardno: poravnaj na
    # zajednicku minutnu mrezu + forward-fill (poslednja poznata cena), pa
    # izbaci pocetne NaN. (Epps/stale-price artefakt -> vidi napomenu u izvestaju.)
    if freq != "daily":
        piv = piv.sort_index().ffill().dropna(how="any")
    ret = np.log(piv).diff().dropna(how="any")
    return ret


def main():
    ret = build_returns(FREQ)
    X = ret.values
    T, N = X.shape
    q = N / T
    print(f"Matrica prinosa (MINUTNI): T={T}, N={N}, q=N/T={q:.5f}")

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

    # ---- slika ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(ev, bins=60, density=True, color="#4C78A8", alpha=0.75, edgecolor="white",
               label="empirijske")
    xs = np.linspace(lmin, lmax, 400)
    ax[0].plot(xs, [rho(x) for x in xs], "r-", lw=2, label="Marčenko–Pastur (šum)")
    ax[0].axvline(lmax, color="k", ls="--", lw=1)
    ax[0].set_xlim(0, max(2.0, lmax * 1.2))
    ax[0].set_title(f"Spektar korelacije MINUTNI (T={T}, N={N}, q={q:.4f})")
    ax[0].set_xlabel("λ"); ax[0].legend()
    ax[1].semilogy(range(1, N + 1), ev, "o-", color="#4C78A8", ms=5)
    ax[1].axhline(lmax, color="r", ls="--", lw=1, label=f"λ+={lmax:.2f}")
    ax[1].axhline(lmin, color="orange", ls="--", lw=1, label=f"λ-={lmin:.2f}")
    ax[1].set_title("Sve svojstvene vrednosti (log)")
    ax[1].set_xlabel("rang"); ax[1].set_ylabel("λ"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=130, bbox_inches="tight")
    print(f"Slika sacuvana: {OUT_FIG.resolve()}")


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
