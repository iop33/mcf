"""
00_backfill_daily.py
--------------------
Povlaci MAKSIMALNU dnevnu istoriju (period='max') za svih 40 tikera preko
yfinance i pakuje u CSV. Ne dira postojece podatke/skripte.

Izlaz (u podaci_ulaz/dnevni_backfill/):
  - <TICKER>.csv          : po jedan CSV po tikeru (puna OHLCV istorija)
  - _spojeno_dnevni.csv   : svi tikeri u 'long' formatu (Datetime,Ticker,OHLCV)
                            -> ovo cita analiza (06_rmt_backfill_daily.py)

Pokretanje:  .venv/bin/python kod/00_backfill_daily.py
"""
from pathlib import Path
import time
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "podaci_ulaz" / "dnevni_backfill"

TICKERS = [
    # FX (9)
    "AUDUSD=X", "EURUSD=X", "GBPUSD=X", "USDAED=X", "USDCAD=X",
    "USDCHF=X", "USDHKD=X", "USDJPY=X", "USDSAR=X",
    # Futures (8)
    "CL=F", "GC=F", "HG=F", "NG=F", "SI=F", "ZC=F", "ZS=F", "ZW=F",
    # ETF / akcije (23)
    "SPY", "XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE",
    "XLU", "XLV", "XLY", "EWA", "EWC", "EWG", "EWH", "EWJ", "EWT",
    "EWW", "EWY", "IEF", "SHY", "TLT",
]


def safe_name(t):
    return t.replace("=", "_").replace("/", "_")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    long_rows = []
    summary = []
    for t in TICKERS:
        try:
            df = yf.download(t, period="max", interval="1d",
                             progress=False, auto_adjust=False, threads=False)
        except Exception as e:
            print(f"  {t:10}: GRESKA {e}")
            summary.append((t, 0, None, None))
            continue
        if df is None or len(df) == 0:
            print(f"  {t:10}: nema podataka")
            summary.append((t, 0, None, None))
            continue
        # yfinance vraca MultiIndex kolone kad je 1 tiker -> spljosti
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df = df.rename(columns={"Date": "Datetime"})
        keep = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
        df = df[[c for c in keep if c in df.columns]].copy()
        df.insert(1, "Ticker", t)
        # po-tiker CSV
        df.to_csv(OUT / f"{safe_name(t)}.csv", index=False)
        long_rows.append(df)
        d0, d1 = df.Datetime.min().date(), df.Datetime.max().date()
        print(f"  {t:10}: {len(df):5d} dana | {d0} -> {d1}")
        summary.append((t, len(df), d0, d1))
        time.sleep(0.3)  # blago da ne preopteretimo izvor

    if long_rows:
        big = pd.concat(long_rows, ignore_index=True)
        big.to_csv(OUT / "_spojeno_dnevni.csv", index=False)
        print(f"\nSpojeno: {len(big)} redova -> {(OUT / '_spojeno_dnevni.csv').resolve()}")

    ok = [s for s in summary if s[1] > 0]
    print(f"\nUspesno: {len(ok)}/{len(TICKERS)} tikera")
    if ok:
        najmladi = max(ok, key=lambda s: s[2])  # tiker sa najkasnijim pocetkom
        print(f"Najmladja istorija (ogranicava zajednicki prozor): "
              f"{najmladi[0]} pocinje {najmladi[2]}")


if __name__ == "__main__":
    main()
