"""
99_pokreni_sve.py
-----------------
Pokrece CEO pipeline jednim Run-om (npr. u PyCharm-u: desni klik -> Run).

Redosled:
  01 recovery -> 02 spektar -> 03 Markowitz -> 06 backfill analiza ->
  02_minute -> 07 grafovi po klasi -> 08 svih 6 metoda -> 09 funkcije ciscenja

Napomene:
  - SKINI_SVEZE_PODATKE = True dodaje 00_backfill_daily.py na pocetak
    (povlaci svezu istoriju sa interneta, ~5 min); podrazumevano iskljuceno
    jer podaci vec postoje u podaci_ulaz/.
  - Skripta 08 je najteza (~10-20 min); ukupno racunaj na ~15-25 min.
  - Svaka skripta sama upisuje svoje izvestaje u rezultati_izlaz/ i slike u
    rezultati_izlaz/slike/, pa je bezbedno pokretati vise puta.
"""
import subprocess
import sys
import time
from pathlib import Path

KOD = Path(__file__).resolve().parent

SKINI_SVEZE_PODATKE = False   # True -> prvo osvezi dnevni backfill (internet)

SKRIPTE = [
    "01_recover_data.py",        # konsolidacija snapshotova
    "02_rmt_analysis.py",        # spektar, prikupljeni uzorak
    "03_markowitz_oos.py",       # OOS test, prikupljeni uzorak
    "06_rmt_backfill_daily.py",  # spektar + OOS, osmogodisnja istorija
    "02_rmt_analysis_minute.py", # minutna kontrola
    "07_grafovi_po_klasi.py",    # 3 grafa po klasi aktive
    "08_markowitz_6metoda.py",   # svih 5 metoda iz rada + robusnost (NAJDUZE)
    "09_funkcije_ciscenja.py",   # reprodukcija slika iz Risk rada
]
if SKINI_SVEZE_PODATKE:
    SKRIPTE.insert(0, "00_backfill_daily.py")


def main():
    ukupno_start = time.time()
    rezime = []
    for ime in SKRIPTE:
        print(f"\n{'='*70}\n>>> {ime}\n{'='*70}", flush=True)
        start = time.time()
        r = subprocess.run([sys.executable, str(KOD / ime)])
        traj = time.time() - start
        status = "OK" if r.returncode == 0 else f"GRESKA (kod {r.returncode})"
        rezime.append((ime, status, traj))
        if r.returncode != 0:
            print(f"\n!!! {ime} je pao — prekidam dalje korake.", flush=True)
            break

    print(f"\n{'='*70}\nREZIME ({(time.time()-ukupno_start)/60:.1f} min ukupno)\n{'='*70}")
    for ime, status, traj in rezime:
        print(f"  {ime:28} {status:12} {traj:6.1f}s")
    koren = KOD.parent / "rezultati_izlaz"
    print(f"\nSvi izvestaji: {koren.resolve()}")
    print(f"Sve slike:     {(koren / 'slike').resolve()}")


if __name__ == "__main__":
    main()
