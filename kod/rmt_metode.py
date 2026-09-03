"""
rmt_metode.py
-------------
Svih pet shema ciscenja korelacione matrice iz rada:
  Bun, Bouchaud & Potters, "Cleaning correlation matrices",
  Risk (Cutting edge: Portfolio management), april 2016.

  1. linear_basic      - basic linear shrinkage ka jedinicnoj matrici (jed. 3)
  2. linear_advanced   - advanced linear shrinkage ka matrici konstantne
                         korelacije (trzisni mod), jed. (4); intenzitet po
                         Ledoit-Wolf "Honey, I shrunk the sample covariance"
  3. clipped           - eigenvalue clipping na MP ivici (jed. 5)
  4. substitution      - eigenvalue substitution: sopstvene vrednosti iznad MP
                         ivice zamenjene inverzijom Marcenko-Pastur relacije
                         za spike (jed. 6), bulk -> 1
  5. rie               - debiased RIE, doslovno po Box 1 rada (jed. 15-20)

VAZNO: pyRMT (v0.1.0) ima gresku u funkciji stieltjes(): racuna
trace(zI - E)/N BEZ inverzije matrice (tacno z-1 za korelacionu matricu),
a Stieltjesova transformacija je trace INVERZA. Zbog toga je pyRMT-ov
optimalShrinkage numericki pogresan. Ovde je RIE implementiran ispravno,
po Box 1; s_k(z_k) = (1/N) * sum_{j != k} 1/(z_k - lambda_j)  (jed. 16).

Sve funkcije primaju matricu prinosa X (T x N; T opservacija, N instrumenata),
interno je standardizuju po kolonama i vracaju korelacionu matricu
(simetricna, dijagonala tacno 1).
"""
import numpy as np


# ---------- pomocne ----------

def _standardize(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(0)
    sd = X.std(0)
    return (X - mu) / sd


def _corr(Z):
    T = Z.shape[0]
    E = (Z.T @ Z) / T
    d = 1.0 / np.sqrt(np.diag(E))
    E = E * d[:, None] * d[None, :]
    return (E + E.T) / 2


def _rebuild(xi, U, renorm=True):
    """Sastavi matricu iz korigovanih sopstvenih vrednosti xi i vektora U
    (kolone U su sopstveni vektori, kao kod np.linalg.eigh)."""
    M = (U * xi) @ U.T
    M = (M + M.T) / 2
    if renorm:
        d = 1.0 / np.sqrt(np.diag(M))
        M = M * d[:, None] * d[None, :]
        np.fill_diagonal(M, 1.0)
    return M


def mp_edges(q):
    """Marcenko-Pastur granice za varijansu 1."""
    return (1 - np.sqrt(q)) ** 2, (1 + np.sqrt(q)) ** 2


# ---------- 0. empirijska ----------

def empirical(X):
    return _corr(_standardize(X))


# ---------- 1. basic linear shrinkage (jed. 3) ----------

def linear_basic(X, alpha=None):
    """Xi = alpha*E + (1-alpha)*I. Ako alpha nije zadat, koristi se
    Ledoit-Wolf optimalni intenzitet (sklearn), sto na standardizovanim
    podacima odgovara istoj meti (skalirana jedinicna matrica)."""
    Z = _standardize(X)
    E = _corr(Z)
    if alpha is None:
        from sklearn.covariance import LedoitWolf
        M = LedoitWolf(assume_centered=True).fit(Z).covariance_
        d = 1.0 / np.sqrt(np.diag(M))
        M = M * d[:, None] * d[None, :]
        np.fill_diagonal(M, 1.0)
        return (M + M.T) / 2
    N = E.shape[0]
    return alpha * E + (1 - alpha) * np.eye(N)


# ---------- 2. advanced linear shrinkage (jed. 4) ----------

def linear_advanced(X):
    """Xi = (1-delta)*E + delta*F, F = (1-rbar)*I + rbar*J (konstantna
    korelacija = jedinicna + trzisni mod). Intenzitet delta po Ledoit-Wolf
    formuli za constant-correlation metu, na standardizovanim prinosima."""
    Z = _standardize(X)
    T, N = Z.shape
    S = _corr(Z)
    rbar = (S.sum() - N) / (N * (N - 1))
    F = np.full((N, N), rbar)
    np.fill_diagonal(F, 1.0)

    # pi-hat: asimptotska varijansa elemenata S
    Y = Z[:, :, None] * Z[:, None, :]          # T x N x N
    pi_mat = ((Y - S) ** 2).mean(0)
    pihat = pi_mat.sum()

    # rho-hat: dijagonalni deo + kovarijacioni clan za konst.-korel. metu
    # theta_{ii,ij} = (1/T) sum_t (z_it^2 - 1)(z_it z_jt - s_ij); s_ii = 1
    theta = ((Z ** 2 - 1.0)[:, :, None] * (Y - S)).mean(0)   # N x N
    offdiag = ~np.eye(N, dtype=bool)
    rhohat = np.trace(pi_mat) + (rbar / 2.0) * (theta + theta.T)[offdiag].sum()

    # gamma-hat: rastojanje mete od S
    gammahat = ((F - S) ** 2).sum()

    kappa = (pihat - rhohat) / max(gammahat, 1e-12)
    delta = float(np.clip(kappa / T, 0.0, 1.0))
    M = (1 - delta) * S + delta * F
    np.fill_diagonal(M, 1.0)
    return (M + M.T) / 2


# ---------- 3. eigenvalue clipping (jed. 5) ----------

def clipped(X):
    """Sopstvene vrednosti iznad MP ivice ostaju; bulk se zamenjuje
    zajednickom konstantom gama koja cuva trag; dijagonala renormalizovana."""
    Z = _standardize(X)
    T, N = Z.shape
    q = N / T
    E = _corr(Z)
    lam, U = np.linalg.eigh(E)
    _, lmax = mp_edges(q)
    keep = lam > lmax
    xi = lam.copy()
    if (~keep).any():
        gamma = (N - lam[keep].sum()) / (~keep).sum()
        xi[~keep] = gamma
    return _rebuild(xi, U)


# ---------- 4. eigenvalue substitution (jed. 6) ----------

def substitution(X):
    """Sopstvene vrednosti iznad MP ivice zamenjene procenom pravih
    inverzijom MP relacije za spike: lam = mu + q*mu/(mu-1)  =>
    mu = ((lam+1-q) + sqrt((lam+1-q)^2 - 4*lam)) / 2. Bulk -> 1."""
    Z = _standardize(X)
    T, N = Z.shape
    q = N / T
    E = _corr(Z)
    lam, U = np.linalg.eigh(E)
    _, lmax = mp_edges(q)
    xi = np.ones(N)
    for k in range(N):
        if lam[k] > lmax:
            b = lam[k] + 1.0 - q
            disc = b * b - 4.0 * lam[k]
            xi[k] = (b + np.sqrt(disc)) / 2.0 if disc > 0 else lam[k]
    return _rebuild(xi, U)


# ---------- 5. debiased RIE (Box 1, jed. 15-20) ----------

def rie_xi(X, gamma="auto"):
    """Vrati (lam, U, xi) za debiased RIE po Box 1 rada.

    gamma: "auto" (podrazumevano) — Γ-korekcija (jed. 17-19) primenjuje se
           SAMO ako je najmanja empirijska sopstvena vrednost konzistentna sa
           donjom MP ivicom (lam_N >= 0.5*(1-sqrt(q))^2). Kalibracija sigma^2
           u Box 1 pretpostavlja da je lam_N bas ta ivica; kod univerzuma sa
           vezanim valutama lam_N je STVARNA skoro-nulta sopstvena vrednost,
           pa Gamma eksplodira (izmereno: Γ~900 na nasem panelu) i unisti
           najmanju, informativnu komponentu. True/False = uvek/nikad.
    """
    Z = _standardize(X)
    T, N = Z.shape
    q = N / float(T)
    E = _corr(Z)
    lam, U = np.linalg.eigh(E)          # rastuce; lam[0] = lambda_N (najmanja)

    z = lam - 1j / np.sqrt(N)

    # jed. (16): s_k(z_k) = (1/N) sum_{j != k} 1/(z_k - lambda_j)
    diff = z[:, None] - lam[None, :]
    np.fill_diagonal(diff, np.inf)
    s = (1.0 / diff).sum(1) / N

    # jed. (15)
    xi = lam / np.abs(1.0 - q + q * z * s) ** 2

    # jed. (17)-(19): korekcija naduvavanja za male sopstvene vrednosti
    lam_N = max(lam[0], 1e-12)
    if gamma is True or (gamma == "auto" and lam_N >= 0.5 * (1 - np.sqrt(q)) ** 2):
        sigma2 = lam_N / (1.0 - np.sqrt(q)) ** 2
        lam_plus = lam_N * (1.0 + np.sqrt(q)) ** 2 / (1.0 - np.sqrt(q)) ** 2
        gmp = (z + sigma2 * (q - 1.0) - np.sqrt((z - lam_N) * (z - lam_plus))) \
            / (2.0 * q * z * sigma2)
        Gamma = sigma2 * np.abs(1.0 - q + q * z * gmp) ** 2 / lam
        # jed. (20)
        xi = np.where(Gamma > 1.0, xi * Gamma, xi)

    return lam, U, xi


def rie(X, gamma="auto"):
    lam, U, xi = rie_xi(X, gamma=gamma)
    return _rebuild(xi, U)
