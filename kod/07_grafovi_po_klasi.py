"""
07_grafovi_po_klasi.py
----------------------
Tri grafa, po jedan za svaku klasu aktive (FX / robni fjucersi / ETF):
normalizovane vremenske serije cena zatvaranja (100 = pocetak zajednickog
prozora), sa oznacenim dogadjajima (COVID slom 2020, tarifni sok 2025,
krah srebra 30.01.2026). Svrha: golim okom uociti odstupanja u podacima
po instrumentu i klasi (motiv: komentari mentora, jul 2026).

Cita:  podaci_ulaz/dnevni_backfill/_spojeno_dnevni.csv
Pise:  rezultati_izlaz/slike/klasa_fx.png
       rezultati_izlaz/slike/klasa_futures.png
       rezultati_izlaz/slike/klasa_etf.png
       rezultati_izlaz/grafovi_po_klasi_izvestaj.txt
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "podaci_ulaz" / "dnevni_backfill" / "_spojeno_dnevni.csv"
OUT_DIR = ROOT / "rezultati_izlaz" / "slike"
OUT_TXT = ROOT / "rezultati_izlaz" / "grafovi_po_klasi_izvestaj.txt"

KLASE = {
    "fx": ("Valutni parovi (9)",
           ["AUDUSD=X", "EURUSD=X", "GBPUSD=X", "USDAED=X", "USDCAD=X",
            "USDCHF=X", "USDHKD=X", "USDJPY=X", "USDSAR=X"]),
    "futures": ("Robni fjucersi (8)",
                ["CL=F", "GC=F", "HG=F", "NG=F", "SI=F", "ZC=F", "ZS=F", "ZW=F"]),
    "etf": ("ETF / akcije / obveznice (23)",
            ["SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
             "XLU", "XLV", "XLY", "EWA", "EWC", "EWG", "EWH", "EWJ", "EWT",
             "EWW", "EWY", "IEF", "SHY", "TLT"]),
}

DOGADJAJI = [
    ("2020-02-19", "COVID slom"),
    ("2025-04-02", "tarifni sok"),
    ("2026-01-30", "krah srebra"),
]


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    d = pd.read_csv(CSV, parse_dates=["Datetime"])
    piv = d.assign(_i=d.Datetime.dt.date).pivot_table(
        index="_i", columns="Ticker", values="Close")
    piv.index = pd.to_datetime(piv.index)
    # zajednicki prozor svih analiza (ogranicava XLC)
    start = piv.dropna(how="any").index.min()
    piv = piv.loc[piv.index >= start]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print(f"Prozor: {piv.index.min().date()} -> {piv.index.max().date()} "
          f"({len(piv)} dana)")
    print("\nNajvece dnevno pomeranje po instrumentu (|log-prinos|):")
    print(f"{'tiker':10} {'datum':12} {'prinos':>8}   napomena")
    print("-" * 52)

    ret = np.log(piv).diff()
    for kljuc, (naslov, tikeri) in KLASE.items():
        fig, ax = plt.subplots(figsize=(13, 6))
        norm = piv[tikeri].ffill()
        norm = norm / norm.iloc[0] * 100.0
        for t in tikeri:
            lw = 2.0 if t == "SI=F" else 1.1
            ax.plot(norm.index, norm[t], lw=lw, label=t.replace("=X", "").replace("=F", ""))
        for dt, ime in DOGADJAJI:
            ax.axvline(pd.Timestamp(dt), color="k", ls="--", lw=0.8, alpha=0.6)
            ax.text(pd.Timestamp(dt), ax.get_ylim()[1], " " + ime,
                    rotation=90, va="top", ha="right", fontsize=8, alpha=0.8)
        ax.set_yscale("log")
        ax.set_title(f"{naslov}: cena zatvaranja, normalizovano (100 = {piv.index[0].date()})")
        ax.set_ylabel("indeks (log skala)")
        ax.legend(ncol=6 if len(tikeri) > 10 else 3, fontsize=7, loc="upper left")
        ax.grid(alpha=0.25)
        plt.tight_layout()
        out = OUT_DIR / f"klasa_{kljuc}.png"
        plt.savefig(out, dpi=130, bbox_inches="tight")
        plt.close(fig)

        # tabela najvecih pomeranja za txt izvestaj
        for t in tikeri:
            r = ret[t].dropna()
            if len(r) == 0:
                continue
            k = r.abs().idxmax()
            nap = ""
            if abs(r.loc[k]) < 1e-9:
                nap = "vezana valuta (prakticno konstantna)"
            elif abs(r.loc[k]) > 0.15:
                nap = "<-- ekstremno odstupanje"
            print(f"{t:10} {str(k.date()):12} {r.loc[k]:+8.3f}   {nap}")
        print("-" * 52)

    print(f"\nSlike: {OUT_DIR.resolve()}/klasa_[fx|futures|etf].png")


if __name__ == "__main__":
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w") as _f:
        _old = sys.stdout
        sys.stdout = _Tee(_old, _f)
        try:
            main()
        finally:
            sys.stdout = _old
    print(f"Izvestaj sacuvan: {OUT_TXT.resolve()}")
