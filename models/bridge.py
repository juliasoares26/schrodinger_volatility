import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import sys
import multiprocessing

try:
    from numba import njit, prange
    import numba as nb
    _NUMBA = True
except ImportError:
    _NUMBA = False
    # stubs silenciosos para rodar sem numba instalado
    def njit(*a, **kw):
        return lambda f: f
    def prange(n):
        return range(n)

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

N_CORES = multiprocessing.cpu_count()
N_PHYSICAL = max(1, N_CORES // 2)  
torch.set_num_threads(N_PHYSICAL)
torch.set_num_interop_threads(max(1, N_PHYSICAL // 2))


@njit(cache=True, fastmath=True, parallel=True)
def _sinkhorn_plan_np(C: np.ndarray, a: np.ndarray, b: np.ndarray,
                      epsilon: float, n_iter: int) -> np.ndarray:
    B = C.shape[0]
    inv_eps = 1.0 / (epsilon + 1e-9)
    log_a = np.log(a + 1e-9)
    log_b = np.log(b + 1e-9)

    # M[i,j] = -C[i,j] / eps  
    M = np.empty((B, B), dtype=np.float64)
    for i in prange(B):
        for j in range(B):
            M[i, j] = -C[i, j] * inv_eps

    log_u = np.zeros(B, dtype=np.float64)
    log_v = np.zeros(B, dtype=np.float64)

    for _ in range(n_iter):
        # log_u: log_u[i] = log_a[i] - logsumexp_j(M[i,j] + log_v[j])
        for i in prange(B):
            mx = M[i, 0] + log_v[0]
            for j in range(1, B):
                val = M[i, j] + log_v[j]
                if val > mx:
                    mx = val
            s = 0.0
            for j in range(B):
                s += np.exp(M[i, j] + log_v[j] - mx)
            log_u[i] = log_a[i] - (np.log(s) + mx)

        # log_v: log_v[j] = log_b[j] - logsumexp_i(M[i,j] + log_u[i])
        for j in prange(B):
            mx = M[0, j] + log_u[0]
            for i in range(1, B):
                val = M[i, j] + log_u[i]
                if val > mx:
                    mx = val
            s = 0.0
            for i in range(B):
                s += np.exp(M[i, j] + log_u[i] - mx)
            log_v[j] = log_b[j] - (np.log(s) + mx)

    # retorna o plano P (B, B) — não o custo escalar
    P = np.empty((B, B), dtype=np.float64)
    for i in prange(B):
        for j in range(B):
            P[i, j] = np.exp(log_u[i] + M[i, j] + log_v[j])
    return P


def sinkhorn_loss_fast(pred: torch.Tensor, target: torch.Tensor,
                       weights: torch.Tensor, epsilon: float = 0.1,
                       n_iter: int = 20) -> torch.Tensor:
    B = pred.shape[0]

    # custo sem grad para Numba
    C_np = torch.cdist(pred.detach(), target.detach(), p=2).pow(2).cpu().numpy().astype(np.float64)
    a = np.full(B, 1.0 / B, dtype=np.float64)
    w_np = weights.detach().cpu().numpy().astype(np.float64)
    b = w_np / (w_np.sum() + 1e-9)

    if _NUMBA:
        P_np = _sinkhorn_plan_np(C_np, a, b, epsilon, n_iter)   # (B, B) float64
    else:
        # fallback torch puro: computa plano P sem grad
        C_t = torch.from_numpy(C_np).float()
        a_t = torch.full((B,), 1.0 / B)
        b_t = torch.from_numpy(b).float()
        P_np = sinkhorn(a_t, b_t, C_t, epsilon).detach().numpy()

    P_t = torch.from_numpy(P_np.astype(np.float32)).to(pred.device)  # (B,B) — sem grad

    # custo COM grad: o gradiente flui por esta cdist
    C_diff = torch.cdist(pred, target.detach(), p=2).pow(2)           # (B, B)
    return (P_t * C_diff).sum()


def sinkhorn(
    a: torch.Tensor,
    b: torch.Tensor,
    C: torch.Tensor,
    epsilon: float = 0.1,
    n_iter: int = 20,
) -> torch.Tensor:
    """Sinkhorn torch puro — mantido para compatibilidade."""
    M = (-C / (epsilon + 1e-9)).contiguous()
    log_a = torch.log(a + 1e-9)
    log_b = torch.log(b + 1e-9)
    log_u = torch.zeros_like(a)
    log_v = torch.zeros_like(b)
    for _ in range(n_iter):
        log_u = log_a - torch.logsumexp(M + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(M + log_u.unsqueeze(1), dim=0)
    return torch.exp(log_u.unsqueeze(1) + M + log_v.unsqueeze(0))


def sinkhorn_loss_torch(
    pred: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    epsilon: float = 0.1,
) -> torch.Tensor:
    B = pred.shape[0]
    C = torch.cdist(pred, target, p=2) ** 2
    a = torch.full((B,), 1.0 / B, device=pred.device)
    b = weights / (weights.sum() + 1e-9)
    P = sinkhorn(a, b, C, epsilon=epsilon)
    return (P * C).sum()


# alias padrão: usa numba se disponível
sinkhorn_loss = sinkhorn_loss_fast


class ResidualBlock(nn.Module):
    def __init__(self, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TransportNet(nn.Module):

    def __init__(
        self,
        d_in: int = 7,
        d_out: int = 7,
        d_noise: int = 8,   
        hidden: int = 256,
        n_blocks: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()

        self.x_proj = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, dropout) for _ in range(n_blocks)]
        )

        self.mu_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, d_out),
        )


        self.log_sigma_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, d_out),
        )
        nn.init.zeros_(self.log_sigma_head[-1].bias)
        nn.init.normal_(self.log_sigma_head[-1].weight, std=0.01)

    def forward(self, x: torch.Tensor,
                z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retorna (pred, mu, sigma) onde pred = mu + sigma * z.
        z : (B, d_noise) — usado apenas como N(0,I); só o shape[0] importa.
        """
        h = self.x_proj(x)
        for block in self.blocks:
            h = block(h)

        mu = self.mu_head(h)                                     
        log_sigma = torch.clamp(self.log_sigma_head(h), -2.3, -0.5)
        sigma = torch.exp(log_sigma)                       

        eps  = torch.randn_like(mu)
        pred = mu + sigma * eps

        return pred, mu, sigma


def _try_compile(net: nn.Module) -> nn.Module:
    if not hasattr(torch, "compile"):
        return net
    try:
        compiled = torch.compile(net, backend="aot_eager", fullgraph=False)
        return compiled
    except Exception:
        return net


class Normalizer:
    def __init__(self):
        self.center: np.ndarray | None = None
        self.scale:  np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self.center = np.median(X, axis=0)
        p05 = np.percentile(X, 5,  axis=0)
        p95 = np.percentile(X, 95, axis=0)
        self.scale  = (p95 - p05) / 2.0 + 1e-8   

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.center) / self.scale

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * self.scale + self.center

    def state(self) -> dict:
        return {"center": self.center, "scale": self.scale}

    def load(self, d: dict):
        if "center" in d:
            self.center = d["center"]
            self.scale  = d["scale"]
        else:
            self.center = d["mean"]
            self.scale  = d["std"]


class ConditionalBrenierSinkhorn:

    def __init__(
        self,
        d_in: int = 7,
        d_out: int = 7,
        d_noise: int = 8,
        hidden: int = 256,
        n_blocks: int = 4,
        device: str = "cpu",
    ):
        self.device  = torch.device(device)
        self.d_noise = d_noise
        self.net = TransportNet(d_in, d_out, d_noise, hidden, n_blocks).to(self.device)

        self.norm_src = Normalizer()
        self.norm_tgt = Normalizer()
        self._bias: np.ndarray | None = None
        
        self._pc_loss_weights = torch.ones(d_out)

        self._noise_cap  = 2048
        self._noise_buf  = torch.zeros(self._noise_cap, d_noise, device=self.device)

    def _to_tensor(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(X, dtype=torch.float32)

    def fit(
        self,
        X_source: np.ndarray,         
        X_target: np.ndarray,         
        weights:  np.ndarray | None = None,
        n_iter: int = 5000,
        batch: int = 512,
        lr: float = 3e-4,
        epsilon: float = 0.2,
        alpha_final: float = 0.6,
        warmup_mse_iters: int = 300,
        log_every: int  = 500,
        num_workers: int = 0,
        ot_every: int = 4,
        sinkhorn_iters: int = 30,
        var_expl_for_weights: np.ndarray | None = None,  
    ):
        self.norm_src.fit(X_source)
        self.norm_tgt.fit(X_target)

        X1_np = self.norm_src.transform(X_source).astype(np.float32)
        X2_np = self.norm_tgt.transform(X_target).astype(np.float32)

        K_out = X2_np.shape[1]
        self._pc_loss_weights = torch.ones(K_out)

        if weights is None:
            W_np = np.ones(len(X1_np), dtype=np.float32)
        else:
            W_np = weights.astype(np.float32)
            W_np = W_np / W_np.mean()

        _nw = max(0, num_workers)
        if _nw > 0:
            _train_threads = max(1, N_PHYSICAL - _nw)
            torch.set_num_threads(_train_threads)

        dataset = TensorDataset(
            torch.from_numpy(X1_np),
            torch.from_numpy(X2_np),
            torch.from_numpy(W_np),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=_nw,
            pin_memory=False,
            drop_last=True,
            persistent_workers=(_nw > 0),
            prefetch_factor=2 if _nw > 0 else None,
        )

        opt = optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=1e-4,
            fused=False,   
        )

        alpha_sched = np.ones(n_iter, dtype=np.float32)
        ot_phase = n_iter - warmup_mse_iters
        if ot_phase > 0:
            t = np.linspace(0.0, 1.0, ot_phase, dtype=np.float32)
            cosine = alpha_final + (1.0 - alpha_final) * 0.5 * (1.0 + np.cos(np.pi * t))
            alpha_sched[warmup_mse_iters:] = cosine

        def lr_lambda(it):
            wu = warmup_mse_iters
            if it < wu:
                return it / max(wu, 1)
            progress = (it - wu) / max(n_iter - wu, 1)
            return 0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        if batch > self._noise_cap:
            self._noise_cap = batch
            self._noise_buf = torch.zeros(batch, self.d_noise, device=self.device)

        self.net.train()
        losses_mse, losses_ot = [], []
        global_it = 0
        _ot_ema = None  

        pbar = tqdm(total=n_iter, desc="Bridge fit")

        while global_it < n_iter:
            for x1, x2, w in loader:
                if global_it >= n_iter:
                    break

                x1 = x1.to(self.device, non_blocking=True)
                x2 = x2.to(self.device, non_blocking=True)
                w  = w.to(self.device,  non_blocking=True)

                alpha = float(alpha_sched[global_it])   
               
                B = x1.shape[0]
                if B <= self._noise_cap:
                    z = self._noise_buf[:B].normal_()
                else:
                    z = torch.randn(B, self.d_noise, device=self.device)

                pred, mu, sigma = self.net(x1, z)

                loss_mu = (w.unsqueeze(1) * self._pc_loss_weights.to(x1.device) * (mu - x2) ** 2).mean()

                log_sigma_vals = torch.log(sigma + 1e-6)              
                log_sigma_tgt  = -1.20                                 
                loss_sigma = ((log_sigma_vals - log_sigma_tgt) ** 2).mean() * 0.5
                loss_mse   = loss_mu + loss_sigma

                if alpha < 1.0 and (global_it % ot_every == 0):
                    loss_ot = sinkhorn_loss(pred, x2, w, epsilon=epsilon,
                                            n_iter=sinkhorn_iters)
                    ot_val = loss_ot.detach().item()
                    _ot_ema = ot_val if _ot_ema is None else 0.98 * _ot_ema + 0.02 * ot_val
                    ot_scale  = max(_ot_ema, 1e-4)
                    mse_scale = max(loss_mse.detach().item(), 1e-4)
                    loss = alpha * loss_mse + (1.0 - alpha) * (loss_ot / ot_scale) * mse_scale
                    losses_ot.append(ot_val)
                else:
                    loss = alpha * loss_mse
                    losses_ot.append(0.0)

                losses_mse.append(loss_mse.item())

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
                scheduler.step()

                global_it += 1
                pbar.update(1)

                if global_it % log_every == 0:
                    avg_mse = np.mean(losses_mse[-log_every:])
                    avg_ot  = np.mean(losses_ot[-log_every:])
                    lr_now  = scheduler.get_last_lr()[0]
                    # sigma médio: indicador de colapso (sigma→0) ou explosão (sigma→∞)
                    with torch.no_grad():
                        _, _, _sig = self.net(x1[:16], z[:16])
                        avg_sigma = _sig.mean().item()
                    pbar.write(
                        f"  iter {global_it:5d} | MSE {avg_mse:.5f} "
                        f"| OT {avg_ot:.4f} | alpha {alpha:.3f} "
                        f"| sigma {avg_sigma:.3f} | lr {lr_now:.2e}"
                    )

        pbar.close()
        print(f"Final MSE (last 200): {np.mean(losses_mse[-200:]):.5f}")
        print(f"Final OT (last 200): {np.mean(losses_ot[-200:]):.4f}")

        self._bias = np.zeros(self.norm_tgt.center.shape, dtype=np.float32)

    def calibrate_bias_conditional(
        self,
        X_cal: np.ndarray,    
        Y_cal: np.ndarray,    
        n_bins: int = 10,
        n_z: int = 64,        
    ):
        from sklearn.isotonic import IsotonicRegression

        self.net.eval()
        N_cal = len(X_cal)

        with torch.no_grad():
            X_n = self.norm_src.transform(X_cal).astype(np.float32)
            x_t = torch.from_numpy(X_n).to(self.device)

            acc = np.zeros((N_cal, self.net.mu_head[-1].out_features), dtype=np.float64)
            for _ in range(n_z):
                z = torch.randn(N_cal, self.d_noise, device=self.device)
                pred_norm, _, _ = self.net(x_t, z)
                acc += pred_norm.cpu().numpy().astype(np.float64)
            mean_pred_norm = (acc / n_z).astype(np.float32)
            mu_pred = self.norm_tgt.inverse(mean_pred_norm)  

        residuals = mu_pred - Y_cal                           
        pc1_cal   = X_cal[:, 0]                              
        
        res_stds = np.maximum(residuals.std(axis=0), 1e-6)

        self._bias_models  = []
        self._bias_pc1_min = float(pc1_cal.min())
        self._bias_pc1_max = float(pc1_cal.max())
        self._bias_res_stds = res_stds  

        print(f" IsotonicRegression, n_z={n_z}):")
        for k in range(residuals.shape[1]):
            r_norm = residuals[:, k] / res_stds[k]
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(pc1_cal, r_norm)
            self._bias_models.append(ir)
            q = np.percentile(pc1_cal, [25, 50, 75])
            b_vals = ir.predict(q) * res_stds[k]
            print(f"    PC{k+1}: bias@Q1={b_vals[0]:+.4f}  "
                  f"Q2={b_vals[1]:+.4f}  Q3={b_vals[2]:+.4f}  "
                  f"(res_std={res_stds[k]:.4f})")

        self._bias = None

    def _conditional_bias(self, X_src_whitened: np.ndarray) -> np.ndarray:
        if not hasattr(self, "_bias_models") or self._bias_models is None:
            return np.zeros((len(X_src_whitened), 1), dtype=np.float32)
        pc1  = X_src_whitened[:, 0]                        
        stds = getattr(self, "_bias_res_stds",
                        np.ones(len(self._bias_models), dtype=np.float32))
        bias  = np.stack(
            [m.predict(pc1) * stds[k] for k, m in enumerate(self._bias_models)],
            axis=1,
        ).astype(np.float32)                                
        return bias


    @torch.no_grad()
    def sample(self, x: np.ndarray, n: int = 500) -> np.ndarray:
        self.net.eval()

        x_n = self.norm_src.transform(x).astype(np.float32)
        x_t = torch.tensor(x_n, dtype=torch.float32).to(self.device)
        if x_t.ndim == 1:
            x_t = x_t.unsqueeze(0).expand(n, -1)
        else:
            x_t = x_t.expand(n, -1)

        z = torch.randn(n, self.d_noise, device=self.device)
        pred, _, _ = self.net(x_t, z)
        pred = pred.cpu().numpy()

        out = self.norm_tgt.inverse(pred)
        if self._bias is not None:
            out = out - self._bias
        elif hasattr(self, "_bias_models") and self._bias_models:
            # bias condicional: x é (d_in,) → replica para (n, d_in) para _conditional_bias
            x_rep = np.tile(x if x.ndim == 1 else x[0], (n, 1))
            out   = out - self._conditional_bias(x_rep)
        return out

    @torch.no_grad()
    def sample_batch(self, X: np.ndarray, n: int = 200, chunk_m: int = 64) -> np.ndarray:
        self.net.eval()
        M   = len(X)
        X_n = self.norm_src.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X_n)   # (M, d_in)

        out_chunks = []
        for start in range(0, M, chunk_m):
            end = min(start + chunk_m, M)
            m  = end - start                         
            X_sub = X_t[start:end]                     
            X_rep = X_sub.unsqueeze(1).expand(m, n, -1).reshape(m * n, -1).to(self.device)
            Z = torch.randn(m * n, self.d_noise, device=self.device)
            pred, _, _ = self.net(X_rep, Z)
            pred  = pred.cpu().numpy()   # (m*n, d_out)
            out_chunks.append(pred.reshape(m, n, -1))

        stacked = np.concatenate(out_chunks, axis=0)    # (M, n, d_out)
        shape = stacked.shape
        flat  = stacked.reshape(-1, shape[-1])
        out   = self.norm_tgt.inverse(flat).reshape(shape)
        if self._bias is not None:
            out = out - self._bias
        elif hasattr(self, "_bias_models") and self._bias_models:
            bias_m = self._conditional_bias(X)           # (M, K)
            out = out - bias_m[:, np.newaxis, :]      # broadcast (M,1,K)
        return out

    def save(self, path: str):
        sd = self.net.state_dict()
        torch.save({
            "state_dict": sd,
            "norm_src": self.norm_src.state(),
            "norm_tgt": self.norm_tgt.state(),
            "d_noise": self.d_noise,
            "bias": self._bias if self._bias is not None else np.zeros(1),
            "bias_models": getattr(self, "_bias_models",    None),
            "bias_pc1_min": getattr(self, "_bias_pc1_min",   None),
            "bias_pc1_max": getattr(self, "_bias_pc1_max",   None),
            "bias_res_stds": getattr(self, "_bias_res_stds",  None),  
            "pc_loss_weights": getattr(self, "_pc_loss_weights", None), 
        }, path)
        print(f"Bridge salvo em: {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.norm_src.load(ckpt["norm_src"])
        self.norm_tgt.load(ckpt["norm_tgt"])
        self.d_noise = ckpt["d_noise"]
        self._bias = ckpt.get("bias", None)
        self._bias_models= ckpt.get("bias_models", None)
        self._bias_pc1_min = ckpt.get("bias_pc1_min", None)
        self._bias_pc1_max = ckpt.get("bias_pc1_max", None)
        self._bias_res_stds = ckpt.get("bias_res_stds", None)   
        w = ckpt.get("pc_loss_weights", None)                    
        if w is not None:
            self._pc_loss_weights = w if isinstance(w, torch.Tensor) else torch.from_numpy(w).float()
        print(f"Bridge carregado de: {path}")


def load_surface_bridge_data(
    npz_path:     str,
    surface_model_path: str,
    horizon:      int = 5,
) -> dict:

    import importlib.util, os

    _sm_dir = Path(__file__).resolve().parent
    _sm_file = _sm_dir / "surface_model.py"
    if not _sm_file.exists():
        _sm_file = Path("surface_model.py")
    if not _sm_file.exists():
        raise FileNotFoundError(
            "surface_model.py not found"
        )

    spec = importlib.util.spec_from_file_location("surface_model", str(_sm_file))
    sm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sm_mod)

    print(f"\nNPZ: {npz_path}")
    print(f"SurfaceModel: {surface_model_path}")

    # carrega ou treina surface_model
    sm = sm_mod.SurfaceModel()
    if Path(surface_model_path).exists():
        sm.load(surface_model_path)
        print("  SurfaceModel carregado do checkpoint.")
    else:
        print("Checkpoint not found — training SurfaceModel...")
        surf_hist_path = str(Path(npz_path).parent / "surface_history.npz")
        sm.fit(npz_path, surface_history_path=surf_hist_path)
        Path(surface_model_path).parent.mkdir(parents=True, exist_ok=True)
        sm.save(surface_model_path)

    scores_full = sm.pc_history.astype(np.float32)   # (N, K_full)

    var_expl  = sm.pca.explained_variance_ratio_
    pc_stds   = scores_full.std(axis=0)
    meaningful = int(np.sum(pc_stds > 0.01))
    meaningful = max(meaningful, 2)
    scores = scores_full[:, :meaningful].copy()
    N, K   = scores.shape
    print(f"  PCs retidas (std > 0.01): {meaningful} de {len(var_expl)}"
          f"  ({var_expl[:meaningful].sum()*100:.1f}% da variância)")

    pc_stds_k = scores.std(axis=0)
    pc_stds_k = np.maximum(pc_stds_k, 1e-6)
    scores    = scores / pc_stds_k
    print(f"  Whitening: pc_stds = {np.round(pc_stds_k, 4).tolist()}")

    X_source = scores[:N - horizon].copy()       # (N-h, K)
    X_target = scores[horizon:].copy()           # (N-h, K)
    N_pairs  = len(X_source)

    pc1 = X_source[:, 0]
    pc1_z = (pc1 - pc1.mean()) / (pc1.std() + 1e-8)
    log_w = -0.5 * pc1_z ** 2 * 0.5 
    w = np.exp(log_w - log_w.max())
    w = np.clip(w, 0.1, None)
    weights = (w / w.mean()).astype(np.float32)

    w_norm    = weights / (weights.mean() + 1e-9)
    eff_N     = int(np.clip(1.0 / (np.mean(w_norm ** 2) + 1e-9), 1, N_pairs))

    print(f"  Scores PCA: N={N_pairs:,}  K={K}  horizon={horizon}d")
    print(f"  X_source: {X_source.shape}  X_target: {X_target.shape}")
    print(f"  Weights: min={weights.min():.3f}  max={weights.max():.3f}  eff_N={eff_N:,}")

    print(f"\n  Variância por componente PCA (source):")
    for k in range(K):
        print(f"    PC{k+1}: std={X_source[:,k].std():.4f}  "
              f"range=[{X_source[:,k].min():.3f}, {X_source[:,k].max():.3f}]")

    return {
        "X_source": X_source,
        "X_target": X_target,
        "weights": weights,
        "surface_model": sm,
        "pca": sm.pca,
        "K": K,
        "horizon": horizon,
        "pc_stds_k": pc_stds_k,
    }


def main():


    NPZ_PATH = r"C:\volatility-options\data\live_spx_data_extended.npz"
    SM_PATH  = r"C:\volatility-options\data\surface_model.npz"
    BRIDGE_PATH = r"C:\volatility-options\data\bridge_surface.pt"

    bundle = load_surface_bridge_data(NPZ_PATH, SM_PATH, ho=5)
    X_source = bundle["X_source"]
    X_target = bundle["X_target"]
    weights = bundle["weights"]
    surface_model = bundle["surface_model"]
    K = bundle["K"]
    pc_stds_k = bundle["pc_stds_k"]
    N = len(X_source)

    print(f"\nDimensões: N={N:,}  K={K}  (espaço PCA da superfície)")

    split = int(0.8 * N)
    X_src_train, X_src_test = X_source[:split], X_source[split:]
    X_tgt_train, X_tgt_test = X_target[:split], X_target[split:]
    W_train = weights[:split]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    hidden = 128
    n_blocks = 3
    d_noise = K * 2        
    n_iter = 10000        
    batch = 256         
    ot_every = 3
    warmup = 800         

    print(f"\nArquitetura: hidden={hidden}  n_blocks={n_blocks}  "
          f"d_noise={d_noise}  n_iter={n_iter}")

    bridge = ConditionalBrenierSinkhorn(
        d_in=K, d_out=K,
        d_noise=d_noise,
        hidden=hidden,
        n_blocks=n_blocks,
        device=device,
    )

    bridge.fit(
        X_src_train, X_tgt_train,
        weights = W_train,
        n_iter = n_iter,
        batch = batch,
        lr = 3e-4,
        epsilon = 0.20,
        alpha_final = 0.70,       
        warmup_mse_iters = warmup,     
        log_every = 500,
        num_workers = min(2, max(0, N_PHYSICAL - 1)),
        ot_every = ot_every,
        sinkhorn_iters = 30,
        var_expl_for_weights = surface_model.pca.explained_variance_ratio_[:K],  
    )

    cal_start = int(0.75 * split)
    bridge.calibrate_bias_conditional(
        X_src_train[cal_start:],
        X_tgt_train[cal_start:],
    )


    n_eval = min(200, len(X_src_test))
    idx  = np.random.choice(len(X_src_test), n_eval, replace=False)
    X_eval = X_src_test[idx]
    X_true = X_tgt_test[idx]

    samps_pc = bridge.sample_batch(X_eval, n=500)       
    means_pc = samps_pc.mean(axis=1)                 

    errs_pc = np.abs(means_pc - X_true).mean(axis=1)
    print(f"MAE médio (espaço PCA):  {errs_pc.mean():.5f}")
    print(f"MAE p25/p75:             {np.percentile(errs_pc,25):.5f} / {np.percentile(errs_pc,75):.5f}")

    K_full = surface_model.pca.n_components
    def _pad_and_reconstruct(scores_k):
        """scores_k: (..., K) whitened → (..., 4, 21) via dewhiten + inverse_transform"""
        shape = scores_k.shape[:-1]
        flat  = scores_k.reshape(-1, K)
        flat  = flat * pc_stds_k
        pad   = np.zeros((len(flat), K_full - K), dtype=np.float32)
        full  = np.concatenate([flat, pad], axis=1)
        surfs = surface_model.pca.inverse_transform(full)
        return surfs.reshape(*shape, 4, 21) if shape else surfs[0]

    pred_surfs = _pad_and_reconstruct(means_pc)         # (n_eval, 4, 21)
    true_surfs = _pad_and_reconstruct(X_true)           # (n_eval, 4, 21)
    errs_iv    = np.abs(pred_surfs - true_surfs).mean(axis=(1, 2))
    print(f"\nMAE médio (escala IV):   {errs_iv.mean():.6f}")
    print(f"MAE p25/p75 (IV):  {np.percentile(errs_iv,25):.6f} / {np.percentile(errs_iv,75):.6f}")
    print(f"MAE p90 (IV):  {np.percentile(errs_iv,90):.6f}")

    per_pc_mae = np.abs(means_pc - X_true).mean(axis=0)
    per_pc_std = samps_pc.std(axis=1).mean(axis=0)
    print(f"\n  MAE e std por componente PCA:")
    print(f"  {'PC':5s}  {'MAE':>8}  {'avg_std':>8}  {'var_expl':>9}")
    var_expl = surface_model.pca.explained_variance_ratio_
    for k in range(K):
        flag = " ⚠" if per_pc_std[k] < 0.20 else ""
        print(f"  PC{k+1:<3d}  {per_pc_mae[k]:8.5f}  {per_pc_std[k]:8.5f}  "
              f"{var_expl[k]*100:8.1f}%{flag}")

    # diagnóstico de bias por regime de PC1
    pc1_test = X_eval[:, 0]
    atm_bias = pred_surfs[:, 0, 10] - true_surfs[:, 0, 10]
    quartis  = np.percentile(pc1_test, [0, 25, 50, 75, 100])
    print(f"\n PC1 Bias")
    print(f"  {'Regime':12s}  {'N':>5}  {'bias_ATM':>10}  {'bias_std':>10}  {'MAE_IV':>8}")
    labels = ["Q1 (baixo)", "Q2", "Q3", "Q4 (alto)"]
    for i, label in enumerate(labels):
        mask = (pc1_test >= quartis[i]) & (pc1_test < quartis[i + 1])
        if mask.sum() == 0:
            continue
        print(f"  {label:12s}  {mask.sum():5d}  "
              f"{atm_bias[mask].mean():+10.5f}  "
              f"{atm_bias[mask].std():10.5f}  "
              f"{errs_iv[mask].mean():8.5f}")
    print(f"  {'TOTAL':12s}  {len(atm_bias):5d}  "
          f"{atm_bias.mean():+10.5f}  {atm_bias.std():10.5f}  {errs_iv.mean():8.5f}")
    
    x_demo = X_src_test[0]
    score_samples = bridge.sample(x_demo, n=1000)           # (1000, K)
    surf_samples  = _pad_and_reconstruct(score_samples)     # (1000, 4, 21)

    print(f"\nDemo (x_test[0]) — distribuição de 1000 superfícies:")
    true_surf_demo = _pad_and_reconstruct(X_tgt_test[0])
    pred_surf_demo = _pad_and_reconstruct(score_samples.mean(axis=0))

    mats = [0.25, 0.50, 0.75, 1.00]
    print(f"  {'Mat':>5}  {'ATM_true':>9}  {'ATM_mean':>9}  {'ATM_std':>8}  "
          f"{'skew_true':>10}  {'skew_mean':>10}")
    for i, T in enumerate(mats):
        atm_true = true_surf_demo[i, 10]
        atm_mean = surf_samples[:, i, 10].mean()
        atm_std = surf_samples[:, i, 10].std()
        sk_true = true_surf_demo[i, 0]  - true_surf_demo[i, 20]
        sk_mean = (surf_samples[:, i, 0] - surf_samples[:, i, 20]).mean()
        print(f"  {T:.2f}Y  {atm_true:9.4f}  {atm_mean:9.4f}  {atm_std:8.4f}  "
              f"{sk_true:10.4f}  {sk_mean:10.4f}")

    Path(BRIDGE_PATH).parent.mkdir(parents=True, exist_ok=True)
    bridge.save(BRIDGE_PATH)


class MartingaleSchrodingerBridge:

    def __init__(self, model) -> None:
        self._model  = model
        self._bridge = None
        self._T: float | None = None

    def train(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        T: float,
        n_iterations: int = 50,
        batch_size: int = 256,
        patience: int = 10,
        l2_reg: float = 1e-3,
        seed: int = 42,
    ):
    
        rng = np.random.default_rng(seed)
        self._T = float(T)
        strikes = np.asarray(strikes, dtype=float)
        market_prices = np.asarray(market_prices, dtype=float)

        try:
            self._model.calibrate_schrodinger_potential(
                market_prices, strikes, self._T, l2_reg=l2_reg,
            )
            calibrated = True
        except Exception:
            calibrated = False

        losses = []
        best_loss  = np.inf
        no_improve = 0

        for _ in range(n_iterations):
            n_paths = max(batch_size, 512)
            prices_cal = self._model.option_prices_mc(
                strikes, T=self._T,
                n_paths=n_paths,
                n_steps=50,
            )

            loss = float(np.sqrt(np.mean((prices_cal - market_prices) ** 2)))
            losses.append(loss)

            if loss < best_loss - 1e-6:
                best_loss  = loss
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        return losses

    def price_option(
        self,
        strike: float,
        T: float,
        n_paths: int = 10_000,
        n_steps: int  = 50,
    ):
 
        K = np.atleast_1d(float(strike))

        try:
            prices = self._model.option_prices_mc(
                K, T=T, n_paths=n_paths, n_steps=n_steps,
            )
        except Exception:
            prices = self._model.option_prices_mc(
                K, T=T, n_paths=n_paths, n_steps=n_steps,
            )

        price = float(prices[0])

        draws = []
        for _ in range(30):
            p_b = self._model.option_prices_mc(
                K, T=T, n_paths=min(n_paths, 500), n_steps=n_steps,
            )
            draws.append(float(p_b[0]))
        se = float(np.std(draws) / np.sqrt(len(draws)))

        return price, se


class SchrodingerPotentialOptimizer:

    def __init__(
        self,
        strikes: np.ndarray,
        market_prices: np.ndarray,
        model,
        n_paths: int = 3_000,
        T: float = 1.0,
        seed: int = 0,
    ):
        self.strikes = np.asarray(strikes, dtype=np.float64)
        self.market_prices = np.asarray(market_prices, dtype=np.float64)
        self.model = model
        self.n_paths = n_paths
        self.T = T
        self._rng = np.random.default_rng(seed)
        self.weights_ = np.zeros(len(strikes), dtype=np.float64)
        self._payoffs = self._simulate_payoffs()

    def _simulate_payoffs(self) -> np.ndarray:
        h = self.model
        dt = self.T / 252
        rho = h.rho
        s1m = np.sqrt(max(1.0 - rho ** 2, 0.0))
        sdt = np.sqrt(dt)

        logS = np.zeros(self.n_paths)
        v = np.full(self.n_paths, max(h.v0, 1e-4))

        for _ in range(252):
            vc = np.maximum(v, 0.0)
            Z1 = self._rng.standard_normal(self.n_paths)
            Z2 = self._rng.standard_normal(self.n_paths)
            v = vc + h.kappa * (h.theta - vc) * dt \
                + h.sigma * np.sqrt(vc) * Z1 * sdt
            np.maximum(v, 1e-6, out=v)
            logS += (h.r - 0.5 * vc) * dt \
                + np.sqrt(vc) * (rho * Z1 + s1m * Z2) * sdt

        S_T = h.S0 * np.exp(logS)
        df = np.exp(-h.r * self.T)
        return df * np.maximum(
            S_T[:, np.newaxis] - self.strikes[np.newaxis, :], 0.0
        )

    def optimize(
        self,
        max_iter: int = 100,
        lr: float = 0.05,
        clip: float = 5.0,
    ) -> np.ndarray:
        lam = np.zeros(len(self.strikes))

        for _ in range(max_iter):
            log_w = self._payoffs @ lam
            log_w -= log_w.max()
            w = np.exp(log_w)
            w /= w.sum()

            grad = self.market_prices \
                - (w[:, np.newaxis] * self._payoffs).sum(axis=0)
            lam = np.clip(lam + lr * grad, -clip, clip)

        self.weights_ = lam
        return lam


if __name__ == "__main__":
    import multiprocessing as _mp
    _mp.freeze_support()
    main()