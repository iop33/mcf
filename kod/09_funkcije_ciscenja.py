"""
09_funkcije_ciscenja.py
-----------------------
Reprodukcija dve kljucne slike iz rada Bun-Bouchaud-Potters (Risk, 2016)
na NASIM podacima (backfill, svih 40 instrumenata):

  LEVI panel  (kao Slika 2 rada): funkcije ciscenja xi(lambda) za
      clipping / linearni shrinkage (alpha=0.5) / debiased RIE,
      na prozoru od T=80 dana (q = 40/80 = 0.5, kao u radu).

  DESNI panel (kao Slika 1 rada): "oracle" test — koliko RIE pogadja
      STVARNI vanuzoracki rizik svojstvenih portfolija (jed. 11-13 rada):
      trening T=80 dana (q=0.5, isti odnos kao u radu), test T_out=60 dana,
      n uzastopnih neprekrivajucih uzoraka. Tacke: (lambda_i, xi_ora_i) po
      uzorku; simboli/linije: prosek po rangu.

Napomena: Gamma-korekcija iz Box 1 se primenjuje samo kada je najmanja
sopstvena vrednost konzistentna sa MP ivicom (vidi rmt_metode.rie_xi) —
na nasem univerzumu sa vezanim valutama lam_N je stvarna ~nulta vrednost.

Cita:  podaci_ulaz/dnevni_backfill/_spojeno_dnevni.csv
Pise:  rezultati_izlaz/funkcije_ciscenja.txt
       rezultati_izlaz/slike/funkcije_ciscenja.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
KOD = Path(__file__).resolve().parent
CSV = ROOT / "podaci_ulaz" / "dnevni_backfill" / "_spojeno_dnevni.csv"
OUT_TXT = ROOT / "rezultati_izlaz" / "funkcije_ciscenja.txt"
OUT_FIG = ROOT / "rezultati_izlaz" / "slike" / "funkcije_ciscenja.png"

sys.path.insert(0, str(KOD.resolve()))
import rmt_metode as rm  # noqa: E402

T_TREN = 80     # trening prozor za oracle: q = 40/80 = 0.5, kao u radu
T_OUT = 60      # vanuzoracki prozor (protokol rada)


class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s)
    def flush(self):
        for st in self.streams:
            st.flush()


def load_returns():
    d = pd.read_csv(CSV, parse_dates=["Datetime"])
    piv = d.assign(_i=d.Datetime.dt.date).pivot_table(
        index="_i", columns="Ticker", values="Close")
    return np.log(piv).diff().dropna(how="any")


def eig_i_xi(win):
    """Za prozor prinosa vrati (lam, xi_clip, xi_lin, xi_rie) — sve sortirano rastuce."""
    Z = (win - win.mean(0)) / win.std(0)
    T, N = Z.shape
    q = N / T
    E = np.corrcoef(Z, rowvar=False)
    lam, U = np.linalg.eigh(E)
    _, lmax = rm.mp_edges(q)

    # clipping
    keep = lam > lmax
    xi_clip = lam.copy()
    if (~keep).any():
        xi_clip[~keep] = (N - lam[keep].sum()) / (~keep).sum()

    # linearni shrinkage alpha = 0.5 (kao na Slici 2 rada)
    xi_lin = 0.5 * lam + 0.5

    # debiased RIE (Box 1, sa Gamma-guardom iz rmt_metode)
    _, _, xi_rie = rm.rie_xi(win)

    return lam, xi_clip, xi_lin, xi_rie


def main():
    ret = load_returns()
    R = ret.values
    T_tot, N = R.shape
    print(f"Panel: T={T_tot}, N={N} | {ret.index.min()} -> {ret.index.max()}")

    # ---- LEVI panel: q=0.5 prozor (poslednjih 80 dana) ----
    win = R[-80:]
    lam, xi_clip, xi_lin, xi_rie = eig_i_xi(win)
    q80 = N / 80
    print(f"\nFunkcije ciscenja: prozor T=80 (q={q80:.2f}), "
          f"lambda raspon [{lam[0]:.3f}, {lam[-1]:.2f}]")

    # ---- DESNI panel: oracle (jed. 11-13) ----
    n = (T_tot - T_TREN - 1) // T_OUT
    print(f"Oracle: T_tren={T_TREN}, T_out={T_OUT}, n={n} uzoraka")
    tacke_lam, tacke_ora, tacke_rie = [], [], []
    for j in range(n):
        a = j * T_OUT
        tren = R[a:a + T_TREN]
        out = R[a + T_TREN:a + T_TREN + T_OUT]
        mu, sd = tren.mean(0), tren.std(0)
        if (sd < 1e-10).any():
            continue
        Z = (tren - mu) / sd
        E = np.corrcoef(Z, rowvar=False)
        lamj, U = np.linalg.eigh(E)
        Xout = (out - mu) / sd
        # R^2(t_j, u_i) = (1/T_out) sum_tau (u_i . X~_tau)^2   (jed. 12-13)
        proj = Xout @ U                    # T_out x N
        ora = (proj ** 2).mean(0)          # xi_ora po svojstvenom vektoru
        _, _, _, rie_j = eig_i_xi(tren)
        tacke_lam.append(lamj); tacke_ora.append(ora); tacke_rie.append(rie_j)
    L = np.array(tacke_lam); O = np.array(tacke_ora); Rr = np.array(tacke_rie)
    # prosek po rangu (rang = pozicija u sortiranom spektru)
    lam_bar, ora_bar, rie_bar = L.mean(0), O.mean(0), Rr.mean(0)

    maska = lam_bar < 4.0   # bulk (trzisni mod van prikaza, kao u radu)
    c_rie = np.corrcoef(rie_bar[maska], ora_bar[maska])[0, 1]
    c_raw = np.corrcoef(lam_bar[maska], ora_bar[maska])[0, 1]
    mse_rie = np.mean((rie_bar[maska] - ora_bar[maska]) ** 2)
    mse_raw = np.mean((lam_bar[maska] - ora_bar[maska]) ** 2)
    print(f"\nSlaganje sa oracle procenom (bulk, prosek po rangu):")
    print(f"  sirova lambda : corr={c_raw:.3f}  MSE={mse_raw:.4f}")
    print(f"  RIE (Box1)    : corr={c_rie:.3f}  MSE={mse_rie:.4f}")
    print("  (RIE treba da ima manju gresku od sirove lambde — to je poenta Slike 1 rada)")

    # ---- crtanje ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))

    ax[0].plot(lam, xi_clip, "r--", lw=2, label="clipping (MP ivica)")
    ax[0].plot(lam, xi_lin, "g:", lw=2, label="linearni shrinkage (α=0.5)")
    ax[0].plot(lam, xi_rie, "b-", lw=2, label="RIE (Box 1)")
    ax[0].plot([0, 4], [0, 4], color="gray", lw=0.8, alpha=0.6, label="bez ciscenja (ξ=λ)")
    ax[0].set_xlim(0, 4); ax[0].set_ylim(0, 4)
    ax[0].set_xlabel("empirijska λ"); ax[0].set_ylabel("očišćena ξ(λ)")
    ax[0].set_title(f"Funkcije čišćenja (T=80, q={q80:.2f}) — kao Slika 2 rada")
    ax[0].legend(); ax[0].grid(alpha=0.25)

    ax[1].scatter(L.ravel(), O.ravel(), s=8, alpha=0.25, color="#888888",
                  label="pojedinačni uzorci")
    ax[1].plot(lam_bar[maska], ora_bar[maska], "^", color="#E8B10C", ms=7,
               label="oracle (prosek po rangu)")
    ax[1].plot(lam_bar[maska], rie_bar[maska], "r--", lw=2, label="RIE (prosek po rangu)")
    ax[1].plot([0, 4], [0, 4], color="gray", lw=0.8, alpha=0.6, label="bez čišćenja (ξ=λ)")
    ax[1].set_xlim(0, 4); ax[1].set_ylim(0, 4)
    ax[1].set_xlabel("empirijska λ (trening)"); ax[1].set_ylabel("stvarni OOS rizik ξ_ora")
    ax[1].set_title(f"Oracle test (T={T_TREN}, T_out={T_OUT}, n={len(L)}) — kao Slika 1 rada")
    ax[1].legend(); ax[1].grid(alpha=0.25)

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
