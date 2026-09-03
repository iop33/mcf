# Čišćenje korelacionih matrica pomoću Random Matrix Theory

Kod, podaci i rezultati za master rad *Correlation Matrix Cleaning: Methods, Stability,
and Cross-Asset Evidence*. Cilj je da se na realnim, samostalno prikupljenim finansijskim
podacima izmeri koliko kratak uzorak „zašumi" korelacionu matricu i koliko pet metoda
čišćenja iz Random Matrix Theory zaista poboljša procenu rizika portfolija van uzorka
(out-of-sample), i na jednoj klasi aktive i na mešovitom, cross-asset univerzumu.

## Metode

Implementirano je svih pet šema čišćenja iz referentnog rada (Bun, Bouchaud i Potters,
*Cleaning correlation matrices*, Risk, 2016), svaka direktno iz svoje formule, u
`kod/rmt_metode.py`:

- basic linear shrinkage
- advanced linear shrinkage
- eigenvalue clipping
- eigenvalue substitution
- rotationally invariant estimator (RIE, debiased po Box 1)

Porede se sa sirovom empirijskom matricom i sa naivnim 1/N portfolijem.

## Podaci

Dva skupa, oba su u repou (`podaci_ulaz/`):

- **Prikupljeni uzorak** — 27 nedeljnih snapshotova tržišta, jedan po nedelji, od januara
  do jula 2026. U svakom su 40 instrumenata: 9 valutnih parova, 8 commodity futures-a i
  23 ETF-a/akcije (SPY, sektorski XL*, ETF-ovi po zemljama EW*, obveznice IEF/SHY/TLT).
  Svaki fajl ima i dnevne i minutne sveće (kolona `Freq`).
- **Istorijski backfill** — najduža dnevna istorija za iste instrumente, jun 2018 – jul
  2026 (~1900 dana), preuzeta sa Yahoo Finance (yfinance).

Deo snapshotova je imao prazna vremena, pa je za neke serije rekonstruisan datum; ceo
postupak i status po tikeru su u `rezultati_izlaz/ocisceni_podaci/recovery_status.csv`.
Očišćeni parquet fajlovi se namerno ne drže u repou (izvedeni su i regenerišu se
pokretanjem skripte `01`).

## Kako se pokreće

Treba samo Python 3.9+ (na Mac-u je komanda `python3`, ne `python`). Iz korena projekta:

```
./pokreni.sh
```

Skripta sama napravi virtuelno okruženje (`.venv`), instalira biblioteke iz
`kod/requirements.txt` i pokrene ceo pipeline (`kod/99_pokreni_sve.py`, redosled
01 → 02 → 03 → 06 → minutna → 07 → 08 → 09). Računaj na ~15–25 minuta; skripta `08` je
najduža.

Ručno ide ovako:

```
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r kod/requirements.txt
python kod/99_pokreni_sve.py
```

Bitno je da `01` ide prvi — on od sirovih snapshotova napravi očišćene panele
(`market_daily_clean.parquet`, `market_minute_clean.parquet`) koje ostale skripte čitaju.
Sveže preuzimanje backfill podataka sa interneta je isključeno po difoltu
(`SKINI_SVEZE_PODATKE = False` u `99_pokreni_sve.py`) jer podaci već postoje u repou.

(Sitnica: pyRMT je iz 2017. i puca na novijem Pythonu zbog `from collections import ...`;
skripte to same reše pre importa, tako da ne moraš ništa ručno da diraš.)

## Šta rade skripte

- `01_recover_data.py` — konsoliduje sirove snapshotove u čiste dnevne i minutne panele.
- `02_rmt_analysis.py` / `02_rmt_analysis_minute.py` — spektar korelacione matrice
  (dnevni / minutni), poređenje sa Marčenko–Pastur granicom (iznad nje su svojstvene
  vrednosti „signal", ispod „šum").
- `03_markowitz_oos.py` — out-of-sample minimum-variance test na prikupljenom uzorku.
- `06_rmt_backfill_daily.py` — spektar i OOS test na osmogodišnjoj istoriji; odatle idu
  glavni kvantitativni rezultati rada.
- `07_grafovi_po_klasi.py` — normalizovane cene po klasi aktive (FX / futures / ETF).
- `08_markowitz_6metoda.py` — svih pet metoda + empirijska + 1/N, kratak i dug prozor,
  oba univerzuma, plus test robusnosti (najduža skripta).
- `09_funkcije_ciscenja.py` — reprodukcija dijagnostičkih slika iz referentnog rada.
- `rmt_metode.py` — implementacija svih pet metoda čišćenja.

`kod/99_pokreni_sve.py` pokreće ceo analitički pipeline jednim pozivom.

## Šta je ispalo

Podaci pokazuju strukturu koju teorija predviđa: dominantni tržišni mod (najveća
svojstvena vrednost, oko trećine ukupne varijanse), par faktora iznad MP granice, i veliki
bulk šuma. Na osmogodišnjem uzorku, sa kratkim prozorom procene, svaka metoda čišćenja
obori stvarni out-of-sample rizik minimum-variance portfolija u odnosu na empirijsku
matricu — linearni shrinkage za oko četvrtinu, eigenvalue clipping za oko šestinu — dok
empirijska matrica realizuje skoro četiri puta veći rizik od onog koji predviđa. Sa dugim
prozorom je empirijska matrica već blizu optimalne, pa agresivno čišćenje postaje
kontraproduktivno. Na homogenom ETF univerzumu RIE je jedina metoda koja pobedi
empirijsku; na mešovitom univerzumu je jednostavniji shrinkage pouzdaniji, jer vezane
valute i izolovani commodity šokovi slabe zajedničku korelacionu strukturu. Tabele su u
`rezultati_izlaz/*.txt`, a slike u `rezultati_izlaz/slike/`.

## Struktura foldera

```
kod/                                skripte + rmt_metode.py + pyRMT
podaci_ulaz/nedeljni_snapshotovi/   sirovi nedeljni snapshotovi (2026)
podaci_ulaz/dnevni_backfill/        dnevni backfill (2018–2026)
rezultati_izlaz/                    slike, izveštaji, recovery_status.csv
```
