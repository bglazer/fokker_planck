import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad


# =============================================================
# Fixed whitening (affine) layer
# =============================================================
class FixedWhiten(nn.Module):
    """
    z = x @ W_T + b, where W_T = L^{-T} and b = -mu @ W_T.
    Buffers are non-trainable; gradients flow through the matmul.
    """

    def __init__(self, W_T: torch.Tensor, b: torch.Tensor):
        super().__init__()
        self.register_buffer("W_T", W_T)  # (D,D)
        self.register_buffer("b", b)  # (D,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W_T + self.b


# -----------------------------
# p(x,t) network
# -----------------------------
class Pxt(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim + 1, hidden_dim, bias=True))
        layers.append(nn.SiLU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=True))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_dim, 1, bias=True))
        self.layers = nn.ModuleList(layers)
        self.model = nn.Sequential(*layers)

        # Optional time scaling: last entry corresponds to t-channel
        self.tscale = nn.Parameter(torch.ones(input_dim + 1), requires_grad=False)

    def xts(self, x, ts):
        # x: (B, D), ts: (T,)
        xs = x.repeat((ts.shape[0], 1, 1))  # (T, B, D)
        ts_ = ts.repeat((x.shape[0], 1)).T.unsqueeze(2)  # (T, B, 1)
        return torch.cat((xs, ts_), dim=2)  # (T, B, D+1)

    def forward(self, xts):
        # xts: (T, B, D+1)
        s = xts.shape
        x = xts.reshape(-1, xts.shape[-1])  # (T*B, D+1)
        # first layer with optional scaling on inputs
        layer0 = self.layers[0]
        x = (x * self.tscale) @ layer0.weight.T + layer0.bias
        for layer in self.layers[1:]:
            x = layer(x)
        x = x.reshape(s[:-1] + (-1,))  # (T, B, 1)
        return x

    def log_pxt(self, x, ts):
        return self.forward(self.xts(x, ts))

    def pxt(self, x, ts):
        return torch.exp(self.log_pxt(x, ts))

    def set_tscale(self, tscale):
        self.tscale.data[-1] = tscale

    def log_px(self, x, ts):
        return torch.logsumexp(self.log_pxt(x, ts), dim=0)

    def dx_dt(self, x, ts):
        """
        Returns:
          dq_dx: (T, B, D)
          dq_dt: (T, B, 1)  # derivative w.r.t. *physical* t if tscale used
        """
        xts = self.xts(x, ts)
        xts.requires_grad_(True)
        q = self.forward(xts)  # (T, B, 1)
        g = grad(
            q,
            xts,
            grad_outputs=torch.ones_like(q),
            create_graph=True,
            retain_graph=True,
        )[
            0
        ]  # (T, B, D+1)
        dq_dx = g[..., :-1]
        dq_dt_internal = g[..., -1:]  # ∂q/∂(raw t input)
        # If forward used t_scaled = s_t * t_phys, autograd gives ∂q/∂t_scaled * s_t.
        # Divide by s_t to recover ∂q/∂t_phys if you scale inside forward.
        s_t = self.tscale[-1]
        dq_dt = dq_dt_internal / s_t
        return dq_dx, dq_dt


# -----------------------------
# Potential flow: u = ∇_x φ(x), div u = Δ_x φ
# -----------------------------
class Phi(nn.Module):
    """Scalar potential φ(x). Drift u = ∇_x φ, divergence = Δ_x φ."""

    def __init__(self, input_dim, hidden_dim, n_layers):
        super().__init__()
        # Support zero-hidden-layer case: φ(x) = w^T x + b
        if n_layers is None or n_layers <= 0:
            self.net = nn.Linear(input_dim, 1, bias=True)
        else:
            layers = [nn.Linear(input_dim, hidden_dim, bias=True), nn.SiLU()]
            for _ in range(n_layers - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.SiLU()]
            layers += [nn.Linear(hidden_dim, 1, bias=True)]  # scalar φ
            self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    def u(self, x):
        """Compute the gradient ∇_x φ(x)"""
        x = x.clone().requires_grad_(True)
        y = self.forward(x)  # (B,1)
        g = grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]
        return g  # IMPORTANT: no +1 bias

    def grad_and_laplacian(self, x, estimator="exact", n_probe=8):
        """
        Compute u = ∇_x φ(x) and Δ_x φ.
        Args:
          x: (B, D)
          estimator: "hutchinson" or "exact"
        Returns:
          u:   (B, D)
          lap: (B, 1)
        """
        x = x.clone().requires_grad_(True)
        y = self.forward(x)
        u = grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]

        if estimator == "exact":
            lap = 0.0
            for i in range(u.shape[-1]):
                gi = u[..., i]
                dgi_dx = grad(
                    gi,
                    x,
                    grad_outputs=torch.ones_like(gi),
                    create_graph=True,
                    retain_graph=True,
                )[0]
                lap = lap + dgi_dx[..., i : i + 1]
        elif estimator == "hutchinson":
            lap = 0.0
            for _ in range(n_probe):
                z = torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)  # Rademacher
                Hz = grad((u * z).sum(), x, create_graph=True, retain_graph=True)[0]
                lap = lap + (z * Hz).sum(dim=-1, keepdim=True)
            lap = lap / float(n_probe)
        else:
            raise ValueError("estimator must be 'exact' or 'hutchinson'")
        return u, lap


# =============================================================
# Main model with seamless whitening in x-space wrappers
# =============================================================
class CellDelta(nn.Module):
    """
    CellDelta jointly learns time-varying density p(x,t) and a scalar potential φ(x)
    (with drift u=∇φ). Whitening is fit once and applied as a fixed affine transform
    so runtime cost is one GEMM.
    """

    def __init__(
        self,
        input_dim,
        ux_hidden_dim,
        ux_layers,
        pxt_hidden_dim,
        pxt_layers,
        use_whitening=True,
        divide_time_by_tscale=True,
        device="cpu",
    ) -> None:
        super().__init__()
        self.phi = Phi(input_dim, ux_hidden_dim, ux_layers).to(device)
        self.pxt = Pxt(input_dim, pxt_hidden_dim, pxt_layers).to(device)
        self.device = device
        self.models = torch.nn.ModuleDict({"phi": self.phi, "pxt": self.pxt})

        # Whitening config/state
        self.use_whitening = use_whitening
        self.divide_time_by_tscale = divide_time_by_tscale
        self.whiten_ready = False
        self.whiten_layer: FixedWhiten | None = None

        # Stored params (buffers)
        self.register_buffer("whiten_mu", torch.zeros(input_dim, dtype=torch.float32))
        self.register_buffer(
            "whiten_L", torch.eye(input_dim, dtype=torch.float32)
        )  # chol(cov)
        self.register_buffer(
            "log_abs_det_W", torch.tensor(0.0, dtype=torch.float32)
        )  # log|det W|

    # ---------------- Whitening helpers ----------------
    @torch.no_grad()
    def _fit_whitener(self, X: torch.Tensor, eps: float = 1e-6):
        """Fit μ and L (Cholesky of covariance) and precompute W_T, b."""
        if not self.use_whitening:
            self.whiten_ready = False
            self.whiten_layer = None
            return
        assert X.ndim == 2, "X must be (n, d) to fit whitener"
        n, d = X.shape
        Xd = X.detach().to(self.device).to(torch.float64)
        mu = Xd.mean(dim=0)
        Xc = Xd - mu
        cov = (Xc.T @ Xc) / max(n - 1, 1)
        # stable jitter
        diag_mean = torch.mean(torch.diag(cov))
        jitter = eps * (diag_mean + 1.0)
        cov = cov + jitter * torch.eye(d, device=Xd.device, dtype=Xd.dtype)
        L = torch.linalg.cholesky(cov)  # lower-triangular
        # Compute L^{-1} by triangular solve (avoids explicit inverse)
        I = torch.eye(d, device=Xd.device, dtype=Xd.dtype)
        L_inv = torch.linalg.solve_triangular(L, I, upper=False)
        W_T = L_inv.T  # (L^{-1})^T = L^{-T}
        b = (-mu) @ W_T  # (D,)
        logdetL = torch.log(torch.diag(L)).sum()
        log_abs_det_W = (-logdetL).to(torch.float32)

        # store
        self.whiten_mu.copy_(mu.to(torch.float32))
        self.whiten_L.copy_(L.to(torch.float32))
        self.log_abs_det_W.copy_(log_abs_det_W)

        self.whiten_layer = FixedWhiten(W_T.to(torch.float32), b.to(torch.float32)).to(
            self.device
        )
        self.whiten_ready = True

    def _ensure_whitener(self, X: torch.Tensor):
        if self.use_whitening and (not self.whiten_ready):
            X_flat = X.detach().reshape(-1, X.shape[-1])
            self._fit_whitener(X_flat)

    def _whiten(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_whitening or not self.whiten_ready or self.whiten_layer is None:
            return x
        return self.whiten_layer(x)

    def _zts(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """Build (T,B,D+1) from whitened x and physical times ts."""
        z = self._whiten(x)  # (B,D)
        T = ts.shape[0]
        B, D = z.shape
        zs = z.repeat((T, 1, 1))  # (T,B,D)
        tss = ts.view(T, 1, 1).expand(T, B, 1)  # (T,B,1)
        return torch.cat([zs, tss], dim=-1)  # (T,B,D+1)

    # ---------------- x-space wrappers for Pxt ----------------
    def log_pxt_x(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        """log p_x(x,t) = log p_z(z,t) + log|det W|, shape (T,B,1)."""
        zts = self._zts(x, ts)
        log_pzt = self.pxt.forward(zts)
        if self.use_whitening and self.whiten_ready:
            return log_pzt + self.log_abs_det_W
        else:
            return log_pzt

    def log_px_x(self, x: torch.Tensor, ts: torch.Tensor) -> torch.Tensor:
        return torch.logsumexp(self.log_pxt_x(x, ts), dim=0)  # (B,1)

    def dx_dt_x(self, x: torch.Tensor, ts: torch.Tensor):
        """∇_x q(x,t) and ∂_t q(x,t) per time step.
        IMPORTANT: we need a per-time gradient wrt x, so we replicate x across T
        and differentiate wrt that replicated tensor (x_rep). This preserves the
        leading T dimension in the gradient.
        """
        T = ts.shape[0]
        B, D = x.shape

        # Replicate x across T and enable grad wrt the replicated variable
        x_rep = x.unsqueeze(0).expand(T, B, D).contiguous()
        x_rep = x_rep.clone().requires_grad_(True)

        # Whiten in one affine (broadcast matmul works on (T,B,D) @ (D,D))
        z_rep = self._whiten(x_rep)  # (T,B,D)
        tss = ts.view(T, 1, 1).expand(T, B, 1)  # (T,B,1)
        zts = torch.cat([z_rep, tss], dim=-1)  # (T,B,D+1)

        q = self.pxt.forward(zts)  # (T,B,1)

        # ∇_x q for each time step
        dq_dx = grad(
            q,
            x_rep,
            grad_outputs=torch.ones_like(q),
            create_graph=True,
            retain_graph=True,
        )[
            0
        ]  # (T,B,D)

        # ∂_t q from the time channel of zts
        g_xts = grad(
            q,
            zts,
            grad_outputs=torch.ones_like(q),
            create_graph=True,
            retain_graph=True,
        )[
            0
        ]  # (T,B,D+1)
        dq_dt_internal = g_xts[..., -1:]  # (T,B,1)

        if self.divide_time_by_tscale:
            s_t = self.pxt.tscale[-1]
            dq_dt = dq_dt_internal / s_t
        else:
            dq_dt = dq_dt_internal
        return dq_dx, dq_dt

    # ---------------- x-space wrappers for φ ----------------
    def u_x(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone().requires_grad_(True)
        z = self._whiten(x)
        y = self.phi.forward(z)
        u = grad(
            y, x, grad_outputs=torch.ones_like(y), create_graph=True, retain_graph=True
        )[0]
        return u

    def grad_and_laplacian_x(self, x: torch.Tensor, estimator="exact", n_probe=8):
        x = x.clone().requires_grad_(True)
        z = self._whiten(x)
        y = self.phi.forward(z)
        u = grad(
            y,
            x,
            grad_outputs=torch.ones_like(y),
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )[0]

        if u is None:
            # φ does not depend on x (e.g., degenerate case); u=0, lap=0
            u = torch.zeros_like(x)
            lap = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            return u, lap

        if estimator == "exact":
            # If φ is linear, u is constant in x and second derivatives are zero.
            if not u.requires_grad:
                lap = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
                return u, lap
            lap = 0.0
            for i in range(u.shape[-1]):
                gi = u[..., i]
                dgi_dx = grad(
                    gi,
                    x,
                    grad_outputs=torch.ones_like(gi),
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if dgi_dx is None:
                    lap_i = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
                else:
                    lap_i = dgi_dx[..., i : i + 1]
                lap = lap + lap_i
        elif estimator == "hutchinson":
            lap = 0.0
            for _ in range(n_probe):
                zprobe = torch.empty_like(x).bernoulli_(0.5).mul_(2).sub_(1)
                Hz = grad(
                    (u * zprobe).sum(),
                    x,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )[0]
                if Hz is None:
                    Hz = torch.zeros_like(x)
                lap = lap + (zprobe * Hz).sum(dim=-1, keepdim=True)
            lap = lap / float(n_probe)
        else:
            raise ValueError("estimator must be 'exact' or 'hutchinson'")
        return u, lap

    # ---------------- Losses (now using x-space wrappers) ----------------
    def nce_loss(self, x, noise, ts, scale):
        y = noise.sample((x.shape[0],))
        log_scale = torch.log(torch.tensor(scale, device=x.device, dtype=x.dtype))
        logp_x = self.log_px_x(x, ts) + log_scale
        logq_x = noise.log_prob(x).unsqueeze(1)
        logp_y = self.log_px_x(y, ts) + log_scale
        logq_y = noise.log_prob(y).unsqueeze(1)

        value_x = logp_x - torch.logaddexp(logp_x, logq_x)
        value_y = logq_y - torch.logaddexp(logp_y, logq_y)
        v = value_x.mean() + value_y.mean()

        r_x = torch.sigmoid(logp_x - logq_x)
        r_y = torch.sigmoid(logq_y - logp_y)
        acc = ((r_x > 0.5).sum() + (r_y > 0.5).sum()).item() / (len(x) + len(y))
        return -v, acc

    def fokker_planck_loss(self, x, ts):
        dq_dx, dq_dt = self.dx_dt_x(x, ts)
        u, lap = self.grad_and_laplacian_x(x)
        adv = (u.unsqueeze(0) * dq_dx).sum(dim=-1, keepdim=True)
        residual = dq_dt + adv + lap.unsqueeze(0).expand_as(dq_dt)
        return (residual**2).mean()

    def consistency_loss(self, X, ts):
        log_pxt = self.log_pxt_x(X, ts)
        l_cons = (log_pxt[1:].mean(1) - log_pxt[0].mean()) ** 2
        return l_cons.mean()

    def time_responsibilities(self, X, ts, temp=1.0):
        logp = self.log_pxt_x(X, ts).squeeze(-1) / temp
        logp = logp - logp.max(dim=0, keepdim=True).values
        w = torch.softmax(logp, dim=0)
        return w

    def time_prior_kl(self, X, ts, prior="uniform", temp=1.0):
        w = self.time_responsibilities(X, ts, temp=temp)
        T = w.size(0)
        if prior == "uniform":
            logu = -np.log(T)
            kl_b = (w * (torch.log(w + 1e-8) - logu)).sum(dim=0)
        else:
            raise NotImplementedError
        return kl_b.mean()

    def divergence_penalty(self, x, lam=1e-2):
        _, lap = self.grad_and_laplacian_x(x)
        return lam * (lap**2).mean()

    def u_penalized_loss(self, X, ts, beta=1e-3, gamma=1.0, temp=1.0, detach_pxt=True):
        T = ts.shape[0]
        B = X.shape[0]

        log_pxt = self.log_pxt_x(X, ts)
        if detach_pxt:
            log_pxt = log_pxt.detach()
        q_over_T = log_pxt / temp
        q_center = q_over_T - q_over_T.max(dim=0, keepdim=True).values
        w = torch.exp(q_center)
        w = w / (w.mean() + 1e-8)

        dq_dx, dq_dt = self.dx_dt_x(X, ts)
        if detach_pxt:
            dq_dx = dq_dx.detach()
            dq_dt = dq_dt.detach()

        u_x, lap = self.grad_and_laplacian_x(X)
        u_T = u_x.unsqueeze(0).expand(T, B, -1)

        adv_res = (u_T * dq_dx).sum(-1, keepdim=True) + dq_dt
        g2 = (dq_dx**2).sum(-1, keepdim=True)
        L_adv = (w * (adv_res**2) / (g2 + 1e-6)).mean()

        u_sq = (u_x**2).sum(dim=-1, keepdim=True)
        L_ke = beta * (w * u_sq.unsqueeze(0)).mean()

        L_div = gamma * (w * (lap.unsqueeze(0) ** 2)).mean()

        L_total = L_adv + L_ke + L_div
        return {"L_u": L_total, "L_adv": L_adv, "L_ke": L_ke, "L_div": L_div}

    # ---------------- Training APIs ----------------
    def optimize(
        self,
        X,
        X0,
        ts,
        px_noise,
        p0_noise,
        p0_alpha=1,
        pxt_lr=5e-4,
        ux_lr=1e-3,
        u_penalty_alpha=None,
        fokker_planck_alpha=1,
        consistency_alpha=None,
        time_prior_kl_alpha=None,
        div_penalty_alpha=None,
        u_and_p_detached=True,
        p_alpha=None,
        n_epochs=100,
        n_samples=1000,
        verbose=False,
    ):
        """Joint optimization"""
        # Fit whitener once from X (flattened)
        self._ensure_whitener(X)

        self.pxt_optimizer = torch.optim.Adam(
            self.pxt.parameters(), lr=pxt_lr, weight_decay=1e-3
        )
        self.phi_optimizer = torch.optim.Adam(
            self.phi.parameters(), lr=ux_lr, weight_decay=1e-3
        )

        zero = torch.zeros(1, device=self.device)
        l_nce_pxs = np.zeros(n_epochs)
        l_nce_p0s = np.zeros(n_epochs)
        l_fps = np.zeros(n_epochs)

        n_samples = min(n_samples, len(X))

        for epoch in range(n_epochs):
            rand_idxs = torch.randperm(len(X), device=X.device)[:n_samples]
            x = X[rand_idxs].detach()
            x0 = X0.detach()

            self.pxt_optimizer.zero_grad()
            self.phi_optimizer.zero_grad()

            if p_alpha is not None:
                l_nce_px, acc_px = self.nce_loss(
                    x, px_noise, ts=ts, scale=1 / ts.shape[0]
                )
                l_nce_px.backward()
            else:
                l_nce_px = zero
                acc_px = zero

            if p0_alpha is not None:
                l_nce_p0, acc_p0 = self.nce_loss(
                    x0, p0_noise, ts=torch.zeros(1, device=x.device), scale=1
                )
                l_nce_p0 = l_nce_p0 * p0_alpha
                l_nce_p0.backward()
            else:
                l_nce_p0 = zero
                acc_p0 = zero

            if fokker_planck_alpha is not None:
                l_fp = self.fokker_planck_loss(x, ts) * fokker_planck_alpha
                l_fp.backward()
            else:
                l_fp = zero

            if u_penalty_alpha is not None:
                losses = self.u_penalized_loss(x, ts, detach_pxt=u_and_p_detached)
                l_u_pen = losses["L_u"] * u_penalty_alpha
                l_adv_pen = losses["L_adv"] * u_penalty_alpha
                l_ke_pen = losses["L_ke"] * u_penalty_alpha
                l_div_pen = losses["L_div"] * u_penalty_alpha
                l_u_pen.backward()
            else:
                l_u_pen = zero
                l_adv_pen = zero
                l_ke_pen = zero
                l_div_pen = zero

            if consistency_alpha is not None:
                l_cons = self.consistency_loss(x, ts) * consistency_alpha
                l_cons.backward()
            else:
                l_cons = zero

            if time_prior_kl_alpha is not None:
                l_tpk = self.time_prior_kl(X, ts) * time_prior_kl_alpha
                l_tpk.backward()
            else:
                l_tpk = zero

            if div_penalty_alpha is not None:
                l_div = self.divergence_penalty(x) * div_penalty_alpha
                l_div.backward()
            else:
                l_div = zero

            self.pxt_optimizer.step()
            self.phi_optimizer.step()

            l_nce_pxs[epoch] = float(l_nce_px.mean())
            l_fps[epoch] = float(l_fp.mean())

            if verbose:
                print(
                    f"{epoch:6d} l_nce_px={float(l_nce_px):.5f}, acc_px={float(acc_px):.5f}, "
                    f"l_nce_p0={float(l_nce_p0): .5f}, acc_p0={float(acc_p0): .5f}, "
                    f"l_fp={float(l_fp):.5f}, l_u_pen={float(l_u_pen):.5f}, "
                    f"l_adv_pen={float(l_adv_pen):.5f}, l_ke_pen={float(l_ke_pen):.5f}, "
                    f"l_div_pen={float(l_div_pen):.5f}, l_cons={float(l_cons):.5f}, "
                    f"l_tpk={float(l_tpk):.5f}, l_div={float(l_div):.5f}"
                )

                if (epoch + 1) % 50 == 0:
                    fp = self.fp_terms(x, ts, lap_estimator="exact")
                    print(
                        f"[FP] {epoch} stats:",
                        {k: round(v, 5) for k, v in fp["stats"].items()},
                    )
                    gc = self.grad_conflict(x, x0, ts, px_noise, p0_noise)
                    print(f"[GRADS] {epoch}:", {k: round(v, 5) for k, v in gc.items()})
                    fpT = self.fp_per_time(x, ts)
                    print(
                        "[FP/T]", {k: np.round(v, 4).tolist() for k, v in fpT.items()}
                    )
                    vfit = self.eval_const_u_fit(x, ts)
                    print("[CONST-U]", {k: round(v, 5) for k, v in vfit.items()})
                    par = self.parallel_fraction(x, ts)
                    print("[U‖]", {k: round(v, 5) for k, v in par.items()})
                    tv_stats = self.time_variation_logp(x, ts)
                    fd_stats = self.fp_density_residual(x, ts)
                    div_stats = self.divergence_stats(x)
                    print("[TV]", {k: round(v, 5) for k, v in tv_stats.items()})
                    print("[FD]", {k: round(v, 5) for k, v in fd_stats.items()})
                    print("[DIV]", {k: round(v, 5) for k, v in div_stats.items()})

        return {
            "l_nce_px": l_nce_pxs,
            "l_nce_p0": l_nce_p0s,
            "l_fp": l_fps,
            "l_tpk": l_tpk,
        }

    def optimize_u_penalized(
        self,
        X,
        ts,
        ux_lr=1e-3,
        beta=1e-3,
        gamma=1e-3,
        temp=1.0,
        detach_pxt=True,
        noise=None,
        n_epochs=100,
        n_samples=1000,
        verbose=False,
    ):
        self._ensure_whitener(X)
        self.phi_optimizer = torch.optim.Adam(self.phi.parameters(), lr=ux_lr)

        L_u_hist = np.zeros(n_epochs, dtype=np.float64)
        L_adv_hist = np.zeros(n_epochs, dtype=np.float64)
        L_ke_hist = np.zeros(n_epochs, dtype=np.float64)
        L_div_hist = np.zeros(n_epochs, dtype=np.float64)

        n_samples = min(n_samples, len(X))
        for epoch in range(n_epochs):
            rand_idxs = torch.randperm(len(X), device=X.device)[:n_samples]
            x = X[rand_idxs].clone().detach()
            if noise is not None and noise > 0:
                x = x + torch.randn_like(x) * float(noise)

            self.phi_optimizer.zero_grad(set_to_none=True)
            losses = self.u_penalized_loss(
                X=x, ts=ts, beta=beta, gamma=gamma, temp=temp, detach_pxt=detach_pxt
            )
            losses["L_u"].backward()
            self.phi_optimizer.step()

            L_u_hist[epoch] = float(losses["L_u"].detach())
            L_adv_hist[epoch] = float(losses["L_adv"].detach())
            L_ke_hist[epoch] = float(losses["L_ke"].detach())
            L_div_hist[epoch] = float(losses["L_div"].detach())

            if verbose:
                print(
                    f"{epoch:4d} L_u={L_u_hist[epoch]:.6f} L_adv={L_adv_hist[epoch]:.6f} "
                    f"L_ke={L_ke_hist[epoch]:.6f} L_div={L_div_hist[epoch]:.6f}"
                )

        return {
            "L_u": L_u_hist,
            "L_adv": L_adv_hist,
            "L_ke": L_ke_hist,
            "L_div": L_div_hist,
        }

    def optimize_initial_conditions(
        self, X0, ts, p0_noise, pxt_lr=1e-3, n_epochs=100, verbose=False, scale=1
    ):
        if not self.whiten_ready:
            self._fit_whitener(X0)
        self.pxt_optimizer = torch.optim.Adam(self.pxt.parameters(), lr=pxt_lr)
        l_nce_p0s = np.zeros(n_epochs)
        for epoch in range(n_epochs):
            ti = torch.randint(0, len(ts), (1,), device=X0.device).item()
            t = ts[ti, None].detach()
            self.pxt_optimizer.zero_grad()
            l_nce_p0, acc_p0 = self.nce_loss(X0, p0_noise, ts=t, scale=scale)
            l_nce_p0.backward()
            self.pxt_optimizer.step()
            l_nce_p0s[epoch] = float(l_nce_p0.mean())
            if verbose:
                print(
                    f"{epoch} l_nce_p0={float(l_nce_p0):.5f}, acc_p0={float(acc_p0):.5f}"
                )
        return {"l_nce_p0": l_nce_p0s}

    # ---------------- Diagnostics (x-space) ----------------
    def simulate(self, X0, tsim, sigma=1, zero_boundary=True):
        x = X0.clone().detach()
        xts = torch.zeros((len(tsim), x.shape[0], x.shape[1]), device="cpu")
        ht = tsim[1] - tsim[0]
        for i in range(len(tsim)):
            u = self.u_x(x)
            dW = torch.randn_like(x) * torch.sqrt(ht)
            dx = u * ht + sigma * dW
            x = x + dx
            if zero_boundary:
                x[x < 0] = 0.0
            xts[i, :, :] = x.cpu().detach()
        return xts

    def fp_terms(self, x, ts, lap_estimator="exact", n_probe=8):
        with torch.enable_grad():
            dq_dx, dq_dt = self.dx_dt_x(x, ts)
            u, lap = self.grad_and_laplacian_x(
                x, estimator=lap_estimator, n_probe=n_probe
            )
            adv = (u * dq_dx).sum(dim=-1, keepdim=True)
            divu = lap.expand(dq_dt.shape)
            residual = dq_dt + adv + divu
        stats = {}
        stats.update(_tensor_stats("dq_dt", dq_dt))
        stats.update(_tensor_stats("adv", adv))
        stats.update(_tensor_stats("divu", divu))
        stats.update(_tensor_stats("res", residual))
        with torch.no_grad():
            a = adv.flatten()
            b = (-dq_dt).flatten()
            dot = float((a * b).mean())
            corr = (
                float(torch.corrcoef(torch.stack([a, b]))[0, 1])
                if a.numel() > 1
                else float("nan")
            )
            stats.update({"adv_vs_neg_dqdt_dot": dot, "adv_vs_neg_dqdt_corr": corr})
        return {
            "dq_dt": dq_dt.detach(),
            "adv": adv.detach(),
            "divu": divu.detach(),
            "residual": residual.detach(),
            "stats": stats,
        }

    def grad_conflict(self, x, x0, ts, px_noise, p0_noise):
        self.pxt_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()
        l_nce_px, _ = self.nce_loss(x, px_noise, ts=ts, scale=1 / ts.shape[0])
        l_nce_p0, _ = self.nce_loss(
            x0, p0_noise, ts=torch.zeros(1, device=x.device), scale=1
        )
        (l_nce_px + l_nce_p0).backward(retain_graph=True)
        g_pxt_nce = _flat_grads(self.pxt)
        g_phi_nce = _flat_grads(self.phi)
        self.pxt_optimizer.zero_grad()
        self.phi_optimizer.zero_grad()
        l_fp = self.fokker_planck_loss(x, ts)
        l_fp.backward(retain_graph=True)
        g_pxt_fp = _flat_grads(self.pxt)
        g_phi_fp = _flat_grads(self.phi)
        return {
            "cos_pxt": _cosine(g_pxt_nce, g_pxt_fp),
            "cos_phi": _cosine(g_phi_nce, g_phi_fp),
            "||g_pxt_nce||": float(g_pxt_nce.norm()),
            "||g_pxt_fp||": float(g_pxt_fp.norm()),
            "||g_phi_nce||": float(g_phi_nce.norm()),
            "||g_phi_fp||": float(g_phi_fp.norm()),
        }

    def fp_per_time(self, x, ts):
        dq_dx, dq_dt = self.dx_dt_x(x, ts)
        u, lap = self.grad_and_laplacian_x(x)
        adv = (u * dq_dx).sum(-1, keepdim=True)
        divu = lap.expand_as(dq_dt)
        res = dq_dt + adv + divu
        with torch.no_grad():
            return {
                "res_mean_per_T": res.mean(1).flatten().cpu().numpy()[::10],
                "res_abs_mean_per_T": res.abs().mean(1).flatten().cpu().numpy()[::10],
                "dq_dt_abs_mean_per_T": dq_dt.abs()
                .mean(1)
                .flatten()
                .cpu()
                .numpy()[::10],
                "adv_abs_mean_per_T": adv.abs().mean(1).flatten().cpu().numpy()[::10],
                "divu_abs_mean_per_T": divu.abs().mean(1).flatten().cpu().numpy()[::10],
            }

    def parallel_fraction(self, x, ts):
        dq_dx, _ = self.dx_dt_x(x, ts)
        u, _ = self.grad_and_laplacian_x(x)
        with torch.no_grad():
            T, B, D = dq_dx.shape
            u_expanded = u.unsqueeze(0).expand(T, -1, -1)
            num = (u_expanded * dq_dx).sum(-1)
            den = u_expanded.norm(dim=-1) * dq_dx.norm(dim=-1) + 1e-8
            cos = (num / den).abs()
        return {
            "u_parallel_frac_mean": float(cos.mean()),
            "u_parallel_frac_p25": float(cos.quantile(0.25)),
            "u_parallel_frac_p75": float(cos.quantile(0.75)),
        }

    def eval_const_u_fit(self, x, ts):
        dq_dx, dq_dt = self.dx_dt_x(x, ts)
        v_hat = torch.linalg.lstsq(
            (-dq_dx.detach()).reshape(-1, dq_dx.shape[-1]),
            dq_dt.detach().reshape(-1, 1),
        ).solution.squeeze(1)
        u_pred, _ = self.grad_and_laplacian_x(x)
        with torch.no_grad():
            cos = torch.nn.functional.cosine_similarity(u_pred, v_hat, dim=-1).mean()
            mse = ((u_pred - v_hat) ** 2).mean()
            norm = v_hat.norm()
        return {
            "cos_u_vhat": float(cos),
            "mse_u_vhat": float(mse),
            "v_hat_norm": float(norm),
        }

    def time_variation_logp(self, X, ts):
        logp = self.log_pxt_x(X, ts).squeeze(-1)
        var_t = logp.var(dim=0)
        return {
            "var_t_logp_mean": float(var_t.mean()),
            "var_t_logp_p25": float(var_t.quantile(0.25)),
            "var_t_logp_p75": float(var_t.quantile(0.75)),
        }

    def fp_density_residual(self, X, ts):
        # underflow-safe: center per-B before exp
        logp = self.log_pxt_x(X, ts)
        logp_c = logp - logp.max(dim=0, keepdim=True).values
        p = torch.exp(logp_c)
        dq_dx, dq_dt = self.dx_dt_x(X, ts)
        u, lap = self.grad_and_laplacian_x(X)
        adv = (u * dq_dx).sum(-1, keepdim=True)
        rlog = dq_dt + adv + lap.unsqueeze(0).expand_as(dq_dt)
        rden = p * rlog
        return {
            "rlog_abs_mean": float(rlog.abs().mean()),
            "rden_abs_mean": float(rden.abs().mean()),
        }

    def divergence_stats(self, X):
        _, lap = self.grad_and_laplacian_x(X)
        return {
            "div_abs_mean": float(lap.abs().mean()),
            "div_p95": float(lap.abs().quantile(0.95)),
        }


# --- utilities ---


def estimate_v_const(dq_dx, dq_dt):
    T, B, D = dq_dx.shape
    A = (-dq_dx).reshape(T * B, D)
    b = dq_dt.reshape(T * B, 1)
    v_hat = torch.linalg.lstsq(A, b).solution.squeeze(1)
    return v_hat


def _tensor_stats(name, t):
    with torch.no_grad():
        return {
            f"{name}_mean": float(t.mean()),
            f"{name}_std": float(t.std()),
            f"{name}_abs_mean": float(t.abs().mean()),
            f"{name}_p95": float(t.abs().quantile(0.95)),
        }


def _grad_norm(module):
    total = 0.0
    for p in module.parameters():
        if p.grad is not None:
            total += p.grad.detach().pow(2).sum().item()
    return total**0.5


def _flat_grads(module):
    vecs = []
    for p in module.parameters():
        if p.grad is None:
            vecs.append(torch.zeros_like(p).reshape(-1))
        else:
            vecs.append(p.grad.detach().reshape(-1))
    if not vecs:
        return torch.zeros(1)
    return torch.cat(vecs)


def _cosine(a, b, eps=1e-12):
    if a.numel() != b.numel():
        m = min(a.numel(), b.numel())
        a, b = a[:m], b[:m]
    an, bn = a.norm(), b.norm()
    if an < eps or bn < eps:
        return 0.0
    return float((a @ b) / (an * bn + eps))
