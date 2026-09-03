#!/usr/bin/env bash
# Pokrece ceo pipeline jednom komandom:  ./pokreni.sh
# Sam pronalazi python3, napravi virtuelno okruzenje, instalira zavisnosti
# i redom pokrene 01 -> 02 -> 03. Radi i kad ne postoji komanda `python`.
set -e
cd "$(dirname "$0")"

PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "GRESKA: nije pronadjen python3. Instaliraj Python 3 pa probaj ponovo."
  exit 1
fi
echo ">> koristim: $PY ($($PY --version 2>&1))"

if [ ! -d .venv ]; then
  echo ">> pravim virtuelno okruzenje (.venv) ..."
  "$PY" -m venv .venv
fi
source .venv/bin/activate

echo ">> instaliram zavisnosti ..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r kod/requirements.txt

echo ">> pokrecem ceo pipeline (01 -> 02 -> 03 -> 06 -> minutna -> 07 -> 08 -> 09) ..."
python kod/99_pokreni_sve.py

echo ""
echo "GOTOVO. Rezultati su u: rezultati_izlaz/  (slike u rezultati_izlaz/slike/)"
