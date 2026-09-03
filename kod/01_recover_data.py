"""
01_recover_data.py
-------------------
Cita sirove nedeljne parquet snapshotove, popravlja prazna vremena u
pokvarenim fajlovima (2026-04-22 i 2026-05-27) za likvidne ETF-ove i
pravi konsolidovane ciste fajlove.

Pokretanje u PyCharm-u: podesi DATA_RAW i DATA_CLEAN ispod, pa Run.

Logika: blok minutnih redova likvidnog ETF-a je tacno N*390 (390 min RTH:
13:30-19:59 UTC). Blok se poploca uzastopnim berzanskim danima i validira na
danima koji se preklapaju sa ispravnim fajlovima (razlika cene mora biti ~0).
"""
from pathlib import Path
import re
import pandas as pd

# ---- PODESAVANJA ----
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "podaci_ulaz" / "nedeljni_snapshotovi"   # folder sa market_data_YYYY-MM-DD.parquet
DATA_CLEAN = ROOT / "rezultati_izlaz" / "ocisceni_podaci"  # izlazni folder
HOLIDAYS = {pd.Timestamp("2026-05-25").date()}  # Memorial Day; dodaj druge ako prosiris period
BROKEN = {  # pokvareni fajlovi: datum_snimanja -> (poslednja_sesija, broj_sesija)
    "2026-04-22": ("2026-04-21", 7),
    "2026-05-27": ("2026-05-26", 7),
}
COLS = ["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume", "Freq", "DateSaved"]


def load(path):
    d = pd.read_parquet(path)
    d["Datetime"] = pd.to_datetime(d["Datetime"], errors="coerce")
    return d


def trading_days_ending(last, n):
    out, d = [], pd.Timestamp(last)
    while len(out) < n:
        if d.dayofweek < 5 and d.date() not in HOLIDAYS:
            out.append(d.date())
        d -= pd.Timedelta(days=1)
    return sorted(out)


def rth_grid(day):
    base = pd.Timestamp(day) + pd.Timedelta(hours=13, minutes=30)
    return [base + pd.Timedelta(minutes=m) for m in range(390)]


def main():
    DATA_CLEAN.mkdir(parents=True, exist_ok=True)
    files = sorted(DATA_RAW.glob("market_data_*.parquet"),
                   key=lambda f: re.search(r"(\d{4}-\d{2}-\d{2})", f.name).group(1))
    if not files:
        raise SystemExit(f"Nema parquet fajlova u {DATA_RAW.resolve()}")

    # ground truth: svi minutni redovi sa ispravnim vremenom
    O = pd.concat([load(f).query("Freq=='minute'").dropna(subset=["Datetime"]) for f in files],
                  ignore_index=True).drop_duplicates(["Ticker", "Datetime"])

    def recover_block(path, last, n):
        """Vrati DataFrame rekonstruisanih redova (samo gap dani) za ETF blokove koji validiraju."""
        d = load(path)
        recovered = []
        for tick, blk in d[d.Freq == "minute"].groupby("Ticker"):
            if len(blk) != n * 390:
                continue
            days = trading_days_ending(last, n)
            times = [t for day in days for t in rth_grid(day)]
            tmp = blk.copy()
            tmp["Datetime"] = times
            tmp["dd"] = pd.to_datetime(tmp["Datetime"]).dt.date
            gap_days, ok = [], True
            for day in days:
                truth = O[(O.Ticker == tick) & (O.Datetime.dt.date == day)]
                if len(truth) == 0:
                    gap_days.append(day)
                    continue
                rc = tmp[tmp.dd == day].set_index("Datetime").Close
                tr = truth.set_index("Datetime").Close
                j = rc.index.intersection(tr.index)
                if len(j) == 0 or (rc.loc[j] - tr.loc[j]).abs().max() > 1e-6:
                    ok = False
                    break
            if ok and gap_days:
                recovered.append(tmp[tmp.dd.isin(gap_days)].drop(columns="dd"))
        return pd.concat(recovered, ignore_index=True) if recovered else None

    O["source"] = "original"
    recs = []
    for f in files:
        date = re.search(r"(\d{4}-\d{2}-\d{2})", f.name).group(1)
        if date in BROKEN:
            r = recover_block(f, *BROKEN[date])
            if r is not None:
                r["source"] = "recovered_exact"
                recs.append(r)
                print(f"  {date}: vraceno {len(r)} minutnih redova")

    R = pd.concat(recs, ignore_index=True) if recs else pd.DataFrame(columns=O.columns)
    minute = (pd.concat([O, R], ignore_index=True)
              .drop_duplicates(["Ticker", "Datetime"], keep="first")
              .sort_values(["Ticker", "Datetime"]))
    out_cols = ["Datetime", "Ticker", "Open", "High", "Low", "Close", "Volume", "Freq", "source"]
    minute[out_cols].to_parquet(DATA_CLEAN / "market_minute_clean.parquet", index=False)

    daily = (pd.concat([load(f).query("Freq=='daily'").dropna(subset=["Datetime"]) for f in files],
                       ignore_index=True).drop_duplicates(["Ticker", "Datetime"]))
    daily["source"] = "original"
    daily[out_cols].sort_values(["Ticker", "Datetime"]).to_parquet(
        DATA_CLEAN / "market_daily_clean.parquet", index=False)

    print(f"\nMINUTNI: {len(minute)} redova | {minute.Datetime.min()} -> {minute.Datetime.max()}")
    print(minute.source.value_counts().to_string())
    print(f"DNEVNI:  {len(daily)} redova")
    print(f"Sacuvano u: {DATA_CLEAN.resolve()}")


if __name__ == "__main__":
    main()
