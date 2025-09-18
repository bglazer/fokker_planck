import numpy as np
import torch
from weak_flow_util import generate_mixture_from_field

def make_linear_translation_data(
    N0: int = 20000,
    N: int = 20000,
    d: int = 2,
    T: int = 100,
    sigma: float = 1.0,
    discrete: bool = True,
    shift_per_unit: float = 1.0,
    seed: int = 0,
):
    rng = np.random.default_rng(seed)
    v = np.zeros(d, dtype=np.float32)
    v[0] = float(shift_per_unit)
    X0 = rng.normal(0.0, sigma, size=(N0, d)).astype(np.float32)
    if discrete:
        ts = rng.integers(low=0, high=T + 1, size=N).astype(np.float32)
    else:
        ts = rng.random(N).astype(np.float32) * float(T)
    eps = rng.normal(0.0, sigma, size=(N, d)).astype(np.float32)
    X = ts[:, None] * v[None, :] + eps
    EX0, EX = X0.mean(0), X.mean(0)
    print(
        f"[sanity] mean(X0)[0]={EX0[0]:+.3f}, mean(X)[0]={EX[0]:+.3f}, expected ≈ {(T/2.0)*shift_per_unit:+.3f}"
    )
    return X0, X, v


# ---- Test 2: Linear drift u(x)=B x ----


def make_linear_B_data(
    B: np.ndarray,
    N0: int = 20000,
    N: int = 40000,
    T: float = 4.0,
    K: int = 64,
    sigma0: float = 1.0,
    seed: int = 0,
):
    d = B.shape[0]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B_t = torch.from_numpy(B.astype(np.float32)).to(dev)

    def u_fn(x: torch.Tensor) -> torch.Tensor:
        return x @ B_t.T

    X0, X, ts = generate_mixture_from_field(
        u_fn, d=d, N0=N0, N=N, T=T, K=K, sigma=sigma0, seed=seed
    )
    return X0, X, ts, u_fn


# helper constructors for B (2D blocks)


def B_rotation_2d(omega: float) -> np.ndarray:
    return np.array([[0.0, -omega], [omega, 0.0]], dtype=np.float32)


def B_sink_source_2d(lmbda: float) -> np.ndarray:
    return np.array([[lmbda, 0.0], [0.0, lmbda]], dtype=np.float32)


def B_saddle_2d(lpos: float, lneg: float) -> np.ndarray:
    return np.array([[lpos, 0.0], [0.0, -abs(lneg)]], dtype=np.float32)


def B_spiral_sink_2d(omega: float, rho: float) -> np.ndarray:
    return np.array([[rho, -omega], [omega, rho]], dtype=np.float32)


# ---- Test 3: Nonlinear gradient flow (double-well) ----


def make_double_well_data(
    d: int = 2,
    alpha: float = 0.5,
    N0: int = 20000,
    N: int = 40000,
    T: float = 4.0,
    K: int = 64,
    sigma0: float = 0.5,
    seed: int = 0,
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def u_fn(x: torch.Tensor) -> torch.Tensor:
        x1 = x[:, :1]
        rest = x[:, 1:]
        g1 = x1 * (x1 * x1 - 1.0)
        if rest.shape[1] > 0:
            grest = alpha * rest
            return torch.cat([g1, grest], dim=1)
        else:
            return g1

    X0, X, ts = generate_mixture_from_field(
        u_fn, d=d, N0=N0, N=N, T=T, K=K, sigma=sigma0, seed=seed
    )
    return X0, X, ts, u_fn


# ---- Test 4: Branching drift (Y-shaped) ----


def make_branching_data(
    d: int = 2,
    v_root: float = 1.0,
    v_branch: float = 1.0,
    xsplit: float = 1.0,
    beta: float = 3.0,
    N0: int = 20000,
    N: int = 40000,
    T: float = 4.0,
    K: int = 64,
    sigma0: float = 0.5,
    seed: int = 0,
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    e1 = torch.zeros(d, device=dev)
    e1[0] = 1.0
    e1 = e1.view(1, -1)
    vA = torch.zeros(d, device=dev)
    vA[:2] = torch.tensor([1.0, 1.0], device=dev)
    vA = (v_branch * vA).view(1, -1)
    vB = torch.zeros(d, device=dev)
    vB[:2] = torch.tensor([1.0, -1.0], device=dev)
    vB = (v_branch * vB).view(1, -1)

    def u_fn(x):
        s = torch.sigmoid(beta * (x[:, :1] - xsplit))
        base = (v_root * x[:, :1]) * e1
        side = torch.where(x[:, 1:2] >= 0, s * vA, s * vB)
        return base + side

    X0, X, ts = generate_mixture_from_field(
        u_fn, d=d, N0=N0, N=N, T=T, K=K, sigma=sigma0, seed=seed
    )
    return X0, X, ts, u_fn


# ---- Test 5: Spiral dynamics embedded in R^d with nuisance dims ----


def make_spiral_manifold_noise_data(
    d: int = 10,
    omega: float = 1.0,
    rho: float = -0.2,
    N0: int = 20000,
    N: int = 40000,
    T: float = 6.0,
    K: int = 96,
    sigma0: float = 0.5,
    seed: int = 0,
):
    assert d >= 2
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    M = torch.tensor([[rho, -omega], [omega, rho]], device=dev)

    def u_fn(x: torch.Tensor) -> torch.Tensor:
        x01 = x[:, :2]
        u01 = x01 @ M.T
        if d > 2:
            zeros = torch.zeros(x.size(0), d - 2, device=dev)
            return torch.cat([u01, zeros], dim=1)
        else:
            return u01

    X0, X, ts = generate_mixture_from_field(
        u_fn, d=d, N0=N0, N=N, T=T, K=K, sigma=sigma0, seed=seed
    )
    return X0, X, ts, u_fn
