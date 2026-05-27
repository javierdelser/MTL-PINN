from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn

from mtl_pinn.utils import set_seed

# -------------------------
# Default experiment settings
# -------------------------
TASK_P = [0.05, 0.1, 0.2, 0.3]
device = "cpu"


# -------------------------
# Analytical solution for Heat equation
# -------------------------
def heat_analytical_numpy(x, t, p):
    """
    Analytical solution for the 1D parametric heat equation:
      u(x,t;p) = sin(pi x) * exp(-pi^2 * alpha(p) * t)
    with alpha(p) = 0.1 * (1 + p)

    x: array-like (...,) or (...,1)
    t: scalar or array-like (broadcastable w.r.t x)
    p: scalar or array-like (broadcastable)
    returns: numpy array with broadcasted shape
    """
    x = np.asarray(x)
    t = np.asarray(t)
    p = np.asarray(p)
    return np.sin(np.pi * x) * np.exp(- (np.pi ** 2) * p * t)

def plot_heat_analytical(p=1.0, nx=100, nt=100, T_max=1.0, cmap='jet', show=True, save_path=None):
    """
    Dibuja la solución analítica definida por heat_analytical_numpy_2:
      u(x,t) = sin(pi x) * exp(-pi^2 * p * t)

    Parámetros:
      - p: valor del parámetro p en la fórmula
      - nx, nt: resolución en x y t
      - T_max: tiempo máximo (t ∈ [0, T_max])
      - cmap: colormap para el contourf
      - show: si True llama a plt.show()
      - save_path: si no es None guarda la figura en la ruta indicada
    """
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, T_max, nt)
    # Mantener indexing='ij' para que la primera dimensión sea x (nx, nt)
    X, T = np.meshgrid(x, t, indexing='xy')
    # heat_analytical_numpy_2 espera arrays broadcastables; hacemos la llamada directamente
    U = heat_analytical_numpy(X, T, p)

    plt.figure(figsize=(8, 4))
    plt.contourf(X, T, U, 100, cmap=cmap)
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Analytical solution (heat_analytical_numpy_2)  p = {p}')
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    if show:
        plt.show()
    else:
        plt.close()





# -------------------------
# MTNN with Multi-Heads
# -------------------------
class BatchedMultiHeadPINN_Heat_Attention(torch.nn.Module):
    """
    Multi-head PINN for the parametric 1D heat equation with an attention network
    that mixes head outputs based on the parameter p.
    API:
      - forward(x, t, p, mode='inference') -> combined u (N,1)
      - forward_head(x, t, head_idx) -> output of a single head (N,1)
      - get_attention_weights(p) -> attention weights (N, T) for batch p
    """
    def __init__(self, task_p, hidden_dim=128, layers=5, attn_hidden=64):
        super().__init__()
        self.task_p = torch.tensor(task_p, dtype=torch.float32).view(1, -1).to(device)
        # Shared layers: input is (x, t)
        self.shared = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            *[layer for _ in range(layers - 1)
              for layer in (nn.Linear(hidden_dim, hidden_dim), nn.Tanh())]
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in task_p])
        # Attention network: input is (nu, task_p) pair
        self.attn_net = nn.Sequential(
            nn.Linear(2, attn_hidden), nn.ReLU(),
            nn.Linear(attn_hidden, attn_hidden), nn.ReLU(),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, x, t, p, mode='inference'):
        """
        x: (N,1), t: (N,1), p: (N,1)
        returns u_out: (N,1)
        """
        # shared encoding
        xt = torch.cat([x, t], dim=1)  # (N,2)
        z = self.shared(xt)            # (N, hidden_dim)

        # heads outputs concatenated -> (N, T)
        head_outs = torch.cat([h(z) for h in self.heads], dim=1)  # (N, T)

        # attention logits: build pairs (p_query, task_p) per sample
        N = p.shape[0]
        T = self.task_p.shape[1]
        p_exp = p.expand(-1, T)                       # (N,T)
        task_p_exp = self.task_p.expand(N, -1) # (N,T)
        attn_inputs = torch.stack([p_exp, task_p_exp], dim=2).view(-1, 2)  # (N*T, 2)
        attn_logits = self.attn_net(attn_inputs).view(N, T)                # (N, T)
        weights = torch.softmax(attn_logits, dim=1)  # (N, T)

        # weighted sum over heads
        u_out = (head_outs * weights).sum(dim=1, keepdim=True)  # (N,1)
        return u_out

    def forward_head(self, x, t, head_idx):
        """
        Evaluate a single head (without attention).
        head_idx: integer index
        """
        xt = torch.cat([x, t], dim=1)
        z = self.shared(xt)
        return self.heads[head_idx](z)

    def get_attention_weights(self, p):
        """
        Given p (N,1) return attention weights (N, T) consistent with forward.
        """
        N = p.shape[0]
        T = self.task_p.shape[1]
        p_exp = p.expand(-1, T)
        task_p_exp = self.task_p.expand(N, -1)
        attn_inputs = torch.stack([p_exp, task_p_exp], dim=2).view(-1, 2)
        attn_logits = self.attn_net(attn_inputs).view(N, T)
        weights = torch.softmax(attn_logits, dim=1)
        return weights



# -------------------------
# TRAINING FUNCTIONS
# -------------------------
# HEADS AND ATTENTION NETWORK TRAINED TOGETHER. HEADS ARE NOT PREVIOUSLY SPECIALIZED.
def train_heat_multitask_attention_head_spec(
    model, optimizer, epochs=3000, task_p=None, weights=None, attn_reg=0.0, device='cpu',
    loss_filename="loss_heat_evolution.txt", model_save_path="Heat/model", save_evolution_model=False,
    T_max=1.0
):
    """
    Multi-head PINN training adapted to the parametric 1D heat equation:
        u_t = alpha(p) * u_xx,  alpha(p) = 0.1 * (1 + p)
    - Each head is expected to specialize for one p in task_p.
    - Attention network is trained together (same style as train_burgers_multitask_attention_head_spec).
    """
    if task_p is None:
        task_p = [0.0, 0.5, 1.0, 1.5]  # example default values
    if weights is None:
        weights = [1.0] * len(task_p)
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        for i, p_val in enumerate(task_p):
            # --- Residual points for this head/parameter ---
            N_r = 400
            x_r = torch.rand((N_r, 1), device=device) * 1.0            # x in [0,1]
            t_r = torch.rand((N_r, 1), device=device) * T_max         # t in [0,T_max]
            p_r = torch.full((N_r, 1), p_val, dtype=torch.float32, device=device)
            x_r.requires_grad_(True)
            t_r.requires_grad_(True)

            # PDE residual: u_t - alpha(p) * u_xx = 0
            u = model(x_r, t_r, p_r, mode='inference')
            u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
            u_t = autograd.grad(u, t_r, torch.ones_like(u), create_graph=True)[0]
            u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
            residual = u_t - p_val * u_xx
            loss_r = torch.mean(residual ** 2)

            # --- Initial condition: u(x,0) = sin(pi x) ---
            N_ic = 100
            x_ic = torch.rand((N_ic, 1), device=device) * 1.0
            t_ic = torch.zeros_like(x_ic)
            p_ic = torch.full((N_ic, 1), p_val, dtype=torch.float32, device=device)
            u_ic_pred = model(x_ic, t_ic, p_ic, mode='inference')
            u_ic_true = torch.sin(np.pi * x_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

            # --- Boundary conditions: u(0,t)=u(1,t)=0 ---
            N_bc = 100
            t_bc = torch.rand((N_bc, 1), device=device) * T_max
            x_l = torch.zeros_like(t_bc)
            x_r_bc = torch.ones_like(t_bc)
            p_bc = torch.full((N_bc, 1), p_val, dtype=torch.float32, device=device)
            u_l = model(x_l, t_bc, p_bc, mode='inference')
            u_r = model(x_r_bc, t_bc, p_bc, mode='inference')
            loss_bc = torch.mean(u_l ** 2 + u_r ** 2)

            # --- (Optional) Regularization: encourage attention to focus on the correct head ---
            attn_reg_loss = 0.0
            if attn_reg > 0 and hasattr(model, "get_attention_weights"):
                att_weights = model.get_attention_weights(p_r)
                target = torch.zeros_like(att_weights)
                target[:, i] = 1.0
                attn_reg_loss = attn_reg * torch.mean((att_weights - target) ** 2)

            total_loss = total_loss + weights[i] * (loss_r + loss_ic + loss_bc + attn_reg_loss)

        # --- Backpropagation and optimizer step ---
        total_loss.backward()
        optimizer.step()

        # Save loss value
        loss_history.append(total_loss.item())

        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Total Loss: {total_loss.item():.2e}")

        if save_evolution_model:
            # Save model every 3000 epochs (and at the last epoch)
            if (epoch + 1) % 3000 == 0 or (epoch + 1) == epochs:
                torch.save(model.state_dict(), f"{model_save_path}_epoch{epoch+1}.pth")
                print(f"Model saved at epoch {epoch+1}")

    # Save loss history to txt file
    with open(loss_filename, "w") as f:
        for loss in loss_history:
            f.write(f"{loss}\n")



# -------------------------
# TRAINING IN 2 STEPS. FIX HEADS AND THEN TRAIN ATTENTION NETWORK.
# Each head only affected by its own loss, shared layers updated by all losses. Attention network NOT trained. Attention network trained in second phase.

def train_heat_multitask_head_spec_isolated_attention(
    model, optimizer, epochs=3000, task_p=None, weights=None, device='cpu',
    loss_filename="loss_heat_head_spec.txt", model_save_path="Heat/model", save_evolution_model=False):
    """
    Multi-head PINN training for attention model (Burgers' equation):
    - Each head is specialized for one nu (only its weights updated by its loss).
    - Per-task losses are computed once per epoch; backbone is updated by their mean,
      then each head is updated from its own stored loss (same forward graph, retain_graph).
    - Attention network is NOT trained in this phase.
    - Saves loss evolution to a txt file and model every 3000 epochs.
    """
    if device is None:
        device = next(model.parameters()).device
    if task_p is None:
        task_p = [0.0, 0.5, 1.0, 1.5]
    if weights is None:
        weights = [1.0] * len(task_p)
    loss_history = []

    def compute_task_loss(i, p_val):
        # PDE residual using only head i
        N_r = 400
        x_r = torch.rand((N_r, 1), device=device) * 1.0
        t_r = torch.rand((N_r, 1), device=device) * 1.0
        x_r.requires_grad_(True)
        t_r.requires_grad_(True)
        z = model.shared(torch.cat([x_r, t_r], dim=1))
        u = model.heads[i](z)
        u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
        u_t = autograd.grad(u, t_r, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
        residual = u_t - p_val * u_xx
        loss_r = torch.mean(residual ** 2)

        # --- Initial condition: u(x,0) = sin(pi x) ---
        N_ic = 100
        x_ic = torch.rand((N_ic, 1), device=device) * 1.0
        t_ic = torch.zeros_like(x_ic)
        z_ic = model.shared(torch.cat([x_ic, t_ic], dim=1))
        u_ic_pred = model.heads[i](z_ic)
        u_ic_true = torch.sin(np.pi * x_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

        # --- Boundary condition: u(0,t) = u(1,t) = 0 ---
        N_bc = 100
        t_bc = torch.rand((N_bc, 1), device=device) * 1.0
        x_l = torch.zeros_like(t_bc)
        x_r_bc = torch.ones_like(t_bc)
        z_l = model.shared(torch.cat([x_l, t_bc], dim=1))
        z_r = model.shared(torch.cat([x_r_bc, t_bc], dim=1))
        u_l = model.heads[i](z_l)
        u_r = model.heads[i](z_r)
        loss_bc = torch.mean(u_l ** 2 + u_r ** 2)

        return weights[i] * (loss_r + loss_ic + loss_bc)

    n_tasks = len(task_p)
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Freeze attention network for this phase
        for param in model.attn_net.parameters():
            param.requires_grad = False

        # Forward: compute each task loss once (shared + heads in graph, attn frozen)
        for param in model.shared.parameters():
            param.requires_grad = True
        for head in model.heads:
            for param in head.parameters():
                param.requires_grad = True
        task_losses = [compute_task_loss(i, p_val) for i, p_val in enumerate(task_p)]

        # Phase 1: backbone step from mean of stored per-task losses
        for head in model.heads:
            for param in head.parameters():
                param.requires_grad = False
        backbone_loss = torch.stack(task_losses).mean()
        backbone_loss.backward(retain_graph=True)
        total_loss = backbone_loss

        # Phase 2: each head from its own stored loss (shared frozen, no re-forward)
        for param in model.shared.parameters():
            param.requires_grad = False
        for i, _ in enumerate(task_p):
            for j, head in enumerate(model.heads):
                for param in head.parameters():
                    param.requires_grad = (j == i)
            task_losses[i].backward(retain_graph=(i < n_tasks - 1))

        optimizer.step()

        # Restore all heads and attention to trainable for next epoch
        for head in model.heads:
            for param in head.parameters():
                param.requires_grad = True
        for param in model.attn_net.parameters():
            param.requires_grad = True

        # Save loss value
        loss_history.append(total_loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Total Loss: {total_loss.item():.2e}")

        if save_evolution_model == True:
            # Save model every 3000 epochs (and at the last epoch)
            if (epoch + 1) % 3000 == 0 or (epoch + 1) == epochs:
                torch.save(model.state_dict(), f"{model_save_path}_epoch{epoch+1}.pth")
                print(f"Model saved at epoch {epoch+1}")

    # Save loss history to txt file
    with open(loss_filename, "w") as f:
        for loss in loss_history:
            f.write(f"{loss}\n")


# After that use this function to train de Atention network
def train_heat_pinn(
    model, optimizer, epochs=3000, task_p=None, weights=None, device='cpu', T_max=1.0, loss_filename="loss_evolution.txt"):
    """
    General training loop for heat PINN (can be used to train attention only
    if shared+heads are frozen and optimizer targets attn_net.params()).
    Samples parameter p randomly from task_p for each residual batch.
    """
    loss_history = []

    if task_p is None:
        task_p = [0.0, 0.5, 1.0, 1.5]
    if weights is None:
        weights = [1.0] * len(task_p)
    for epoch in range(epochs):
        optimizer.zero_grad()
        # --- Residual points (sampled for all p) ---
        N_r = 400
        x_r = torch.rand((N_r, 1), device=device) * 1.0
        t_r = torch.rand((N_r, 1), device=device) * T_max
        p_r_np = np.random.choice(task_p, size=(N_r, 1), replace=True)
        p_r = torch.tensor(p_r_np, dtype=torch.float32, device=device)
        x_r.requires_grad_(True)
        t_r.requires_grad_(True)

        # --- PDE residual: u_t - alpha(p) * u_xx = 0 ---
        u = model(x_r, t_r, p_r, mode='train')
        u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
        u_t = autograd.grad(u, t_r, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
        residual = u_t - p_r * u_xx
        loss_r = torch.mean(residual ** 2)

        # --- Initial condition: u(x, 0) = sin(pi x) ---
        N_ic = 100
        x_ic = torch.rand((N_ic, 1), device=device) * 1.0
        t_ic = torch.zeros_like(x_ic)
        p_ic_np = np.random.choice(task_p, size=(N_ic, 1), replace=True)
        p_ic = torch.tensor(p_ic_np, dtype=torch.float32, device=device)
        u_ic_pred = model(x_ic, t_ic, p_ic, mode='train')
        u_ic_true = torch.sin(np.pi * x_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

        # --- Boundary conditions: u(0,t) = u(1,t) = 0 ---
        N_bc = 100
        t_bc = torch.rand((N_bc, 1), device=device) * T_max
        p_bc_np = np.random.choice(task_p, size=(N_bc, 1), replace=True)
        p_bc = torch.tensor(p_bc_np, dtype=torch.float32, device=device)
        u_l = model(torch.zeros_like(t_bc), t_bc, p_bc, mode='train')
        u_r = model(torch.ones_like(t_bc), t_bc, p_bc, mode='train')
        loss_bc = torch.mean(u_l ** 2 + u_r ** 2)

        # --- Total loss ---
        loss = loss_r + loss_ic + loss_bc
        loss.backward()
        optimizer.step()

        # Save loss value
        loss_history.append(loss.item())

        if epoch % 200 == 0:
            print(f"Epoch {epoch} | Residual: {loss_r.item():.2e}, Total: {loss.item():.2e}")

    # Save loss history to txt file
    with open(loss_filename, "w") as f:
        for loss_value in loss_history:
            f.write(f"{loss_value}\n")




def plot_heads_vs_true_one_to_one_heat(model_spec, model_no_spec, nx=50, nt=50, task_p=None, T_max=1.0):
    """
    Para cada cabeza i plotea su salida frente a la solución analítica
    para p = task_p[i]. Atención: task_p debe tener al menos tantas entradas
    como heads en el modelo (o se truncará).
    """
    if task_p is None:
        task_p = TASK_P
    n_heads = min(len(model_spec.heads), len(task_p))
    x = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)
    t = torch.linspace(0.0, T_max, nt, device=device).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)

    #plt.figure(figsize=(4 * n_heads, 8))
    
    mins = []
    maxs = []
    
    for i in range(n_heads):
        p_val = task_p[i]

        # Head output
        u_head_spec = model_spec.forward_head(x_input, t_input, i).detach().cpu().numpy().reshape(nx, nt)
        u_head_no_spec = model_no_spec.forward_head(x_input, t_input, i).detach().cpu().numpy().reshape(nx, nt)
        
        min_val = min(u_head_spec.min(), u_head_no_spec.min())
        max_val = max(u_head_spec.max(), u_head_no_spec.max())

        mins.append(min_val)
        maxs.append(max_val)
        
    total_min = min(mins)
    total_max = max(maxs)    
    
    fig = plt.figure(figsize=(12, 8))
    axes = fig.subplots(3, 4)
    for i in range(n_heads):
        p_val = task_p[i]

        # Head output (N,1) -> reshape (nx, nt)
        u_head_spec = model_spec.forward_head(x_input, t_input, i).detach().cpu().numpy().reshape(nx, nt)
        u_head_no_spec = model_no_spec.forward_head(x_input, t_input, i).detach().cpu().numpy().reshape(nx, nt)

        # True analytical solution for this p
        u_true = heat_analytical_numpy(X.cpu().numpy(), T.cpu().numpy(), p_val)

        # Plot head output
        #plt.subplot(3, n_heads, i + 1)
        axes[0,i].contourf(X.cpu(), T.cpu(), u_head_spec, levels=100, cmap='RdBu_r', vmin=total_min, vmax=total_max)
        axes[0,i].set_title(f'Head {i} output (μ = {p_val}), two-phase training')
        if i==0:
            axes[0,i].set_ylabel('t', fontsize=15)

        axes[0,i].tick_params(axis='both', labelsize=12)

        #plt.subplot(3, n_heads, n_heads + i + 1)
        axes[1,i].contourf(X.cpu(), T.cpu(), u_head_no_spec, levels=100, cmap='RdBu_r', vmin=total_min, vmax=total_max)
        axes[1,i].set_title(f'Head {i} output (μ = {p_val}), one-phase training')
        if i==0:
            axes[1,i].set_ylabel('t', fontsize=15)

        axes[1,i].tick_params(axis='both', labelsize=12)

        # Plot true solution
        #plt.subplot(3, n_heads, 2*n_heads + i + 1)
        axes[2,i].contourf(X.cpu(), T.cpu(), u_true, levels=100, cmap='RdBu_r', vmin=total_min, vmax=total_max)
        axes[2,i].set_title(f'Analytical solution (μ = {p_val})')
        axes[2,i].tick_params(axis='both', labelsize=12)
        
        axes[2,i].set_xlabel('x', fontsize=15)
        if i==0:
            axes[2,i].set_ylabel('t', fontsize=15)
        
    cmap = matplotlib.cm.RdBu_r
    norm = matplotlib.colors.Normalize(vmin=total_min, vmax=total_max)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, location='right', orientation='vertical',pad=0.02)
    #plt.tight_layout()
    plt.show()


def plot_heads_only_one_to_one_heat(model, nx=50, nt=50, task_p=None, T_max=1.0, file_save=None):
    """
    Para cada cabeza i plotea solo su salida para p = task_p[i].
    Atención: task_p debe tener al menos tantas entradas como heads en el modelo
    (o se truncará).
    """
    if task_p is None:
        task_p = TASK_P
    n_heads = min(len(model.heads), len(task_p))

    x = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)
    t = torch.linspace(0.0, T_max, nt, device=device).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)

    plt.figure(figsize=(4 * n_heads, 4))
    for i in range(n_heads):
        p_val = task_p[i]
        u_head = model.forward_head(x_input, t_input, i).detach().cpu().numpy().reshape(nx, nt)

        plt.subplot(1, n_heads, i + 1)
        plt.contourf(X.cpu(), T.cpu(), u_head, 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'Head {i} output (μ = {p_val})')

    plt.tight_layout()

    plt.suptitle("Heads obtained after training")
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(file_save, format="pdf", bbox_inches="tight")
    plt.show()


def plot_true_solutions_one_to_one_heat(nx=50, nt=50, task_p=None, T_max=1.0, file_save=None):
    """
    Plotea solo la solución analítica de la PDE para cada parámetro de entrenamiento.
    Por defecto usa TASK_P.
    """
    if task_p is None:
        task_p = TASK_P

    n_params = len(task_p)
    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, T_max, nt)
    X, T = np.meshgrid(x, t, indexing='ij')

    plt.figure(figsize=(4 * n_params, 4))
    for i, p_val in enumerate(task_p):
        u_true = heat_analytical_numpy(X, T, p_val)

        plt.subplot(1, n_params, i + 1)
        plt.contourf(X, T, u_true, 100, cmap='jet')
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'μ = {p_val}')

    plt.tight_layout()

    plt.suptitle("Analytical solutions for different μ values")
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(file_save, format="pdf", bbox_inches="tight")
    plt.show()


def plot_pred_and_real_heat(model, p=1, nx=100, nt=100, T_max=1.0):
    """
    Plot the PINN prediction and the analytical solution for the 1D parametric
    heat equation for a given parameter p.
    """
    x = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)
    t = torch.linspace(0.0, T_max, nt, device=device).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    p_input = torch.full_like(x_input, float(p))

    # PINN prediction
    u_pred = model(x_input, t_input, p_input, mode='inference').reshape(nx, nt).detach().cpu().numpy()

    # Analytical solution
    u_real = heat_analytical_numpy(X.cpu().numpy(), T.cpu().numpy(), p)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.contourf(X.cpu(), T.cpu(), u_pred, 100, cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f"PINN Prediction (p = {p})")

    plt.subplot(1, 2, 2)
    plt.contourf(X.cpu(), T.cpu(), u_real, 100, cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f"Analytical Solution (p = {p})")

    plt.tight_layout()
    plt.suptitle(f"PINN vs Analytical Solution (p = {p})", y=1.02)
    plt.show()


def plot_pred_heat(model, p=1, nx=100, nt=100, T_max=1.0):
    """
    Plot the PINN prediction and the analytical solution for the 1D parametric
    heat equation for a given parameter p.
    """
    x = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)
    t = torch.linspace(0.0, T_max, nt, device=device).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    p_input = torch.full_like(x_input, float(p))

    # PINN prediction
    u_pred = model(x_input, t_input, p_input, mode='inference').reshape(nx, nt).detach().cpu().numpy()

    plt.contourf(X.cpu(), T.cpu(), u_pred, 100, cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    # plt.title(f"PINN Prediction (p = {p})")
    plt.tight_layout()
    plt.savefig('Heat/test.pdf', format="pdf", bbox_inches="tight")
    plt.show()


def plot_real_heat(model, p=1, nx=100, nt=100, T_max=1.0):
    """
    Plot the PINN prediction and the analytical solution for the 1D parametric
    heat equation for a given parameter p.
    """
    x = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)
    t = torch.linspace(0.0, T_max, nt, device=device).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    p_input = torch.full_like(x_input, float(p))

    # Analytical solution
    u_real = heat_analytical_numpy(X.cpu().numpy(), T.cpu().numpy(), p)

    plt.contourf(X.cpu(), T.cpu(), u_real, 100, cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    # plt.title(f"Analytical Solution (p = {p})")
    plt.tight_layout()
    plt.savefig('Heat/real.pdf', format="pdf", bbox_inches="tight")
    plt.show()


@torch.no_grad()
def plot_absolute_error_heat(model, p=1.0, nx=100, nt=100, T_max=1.0,
                                                         cmap='hot', file_save=None):
        """
        Compute and plot the absolute error between the model approximation and
        the analytical solution for a given parameter p.

        Returns:
            X, T         : grid coordinates as numpy arrays
            abs_error    : pointwise absolute error on the grid
            mae          : mean absolute error over the grid
        """
        x = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)
        t = torch.linspace(0.0, T_max, nt, device=device).reshape(-1, 1)
        X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

        x_input = X.reshape(-1, 1)
        t_input = T.reshape(-1, 1)
        p_input = torch.full_like(x_input, float(p))

        u_pred = model(x_input, t_input, p_input, mode='inference').reshape(nx, nt).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        T_np = T.detach().cpu().numpy()
        u_true = heat_analytical_numpy(X_np, T_np, p)

        abs_error = np.abs(u_pred - u_true)
        mae = np.mean(abs_error)

        contour = plt.contourf(X_np, T_np, abs_error, 100, cmap=cmap)
        plt.colorbar(contour, label='|u_pred - u_true|')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'Absolute Error of the Approximation (p = {p})')
        plt.tight_layout()

        plt.savefig(file_save, format='pdf', bbox_inches='tight')
        plt.show()
        print(f"MAE for p = {p}: {mae:.6e}")

        return X_np, T_np, abs_error, mae


@torch.no_grad()
def visualize(model_spec, model_no_spec, lam_val=[1.0], nx=100, nt=100, T_max=1.0):

    x = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)
    t = torch.linspace(0.0, T_max, nt, device=device).reshape(-1, 1)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    
    u_pred_spec = {}
    u_pred_no_spec = {}
    u_true = {}
    abs_error_spec = {}
    abs_error_no_spec = {}
    row0_min, row0_max = None, None
    row1_min, row1_max = None, None
    
    fig, axs = plt.subplots(2, 2*len(lam_val), figsize=(16, 14))
    cs_top = None
    cs_bottom = None
    
    for lam in lam_val:
        p_input = torch.full_like(x_input, lam)

        u_pred_spec[lam] = model_spec(x_input, t_input, p_input, mode='inference').reshape(nx, nt).detach().cpu().numpy()
        u_pred_no_spec[lam] = model_no_spec(x_input, t_input, p_input, mode='inference').reshape(nx, nt).detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        T_np = T.detach().cpu().numpy()
        u_true[lam] = heat_analytical_numpy(X_np, T_np, lam)
        
        abs_error_spec[lam] = np.abs(u_pred_spec[lam] - u_true[lam])
        abs_error_no_spec[lam] = np.abs(u_pred_no_spec[lam] - u_true[lam])
        current_row0_min = min(u_pred_spec[lam].min(), u_pred_no_spec[lam].min())
        current_row0_max = max(u_pred_spec[lam].max(), u_pred_no_spec[lam].max())
        current_row1_min = min(abs_error_spec[lam].min(), abs_error_no_spec[lam].min())
        current_row1_max = max(abs_error_spec[lam].max(), abs_error_no_spec[lam].max())

        row0_min = current_row0_min if row0_min is None else min(row0_min, current_row0_min)
        row0_max = current_row0_max if row0_max is None else max(row0_max, current_row0_max)
        row1_min = current_row1_min if row1_min is None else min(row1_min, current_row1_min)
        row1_max = current_row1_max if row1_max is None else max(row1_max, current_row1_max)
        
    
    
    for lam in lam_val:
        idx = lam_val.index(lam)
        u_levels = np.linspace(row0_min, row0_max, 100) if row0_min is not None and row0_max is not None else 100
        err_levels = np.linspace(row1_min, row1_max, 100) if row1_min is not None and row1_max is not None else 100

        cs_top = axs[0,2*idx].contourf(X_np, T_np, u_pred_spec[lam], levels=u_levels, cmap='RdBu_r', label='U(x; $\\lambda$), one-phase training')
        cs_top.set_label('')
        cs_bottom = axs[1,2*idx].contourf(X_np, T_np, abs_error_spec[lam], levels=err_levels, cmap='YlOrRd', label='Absolute error, one-phase training')
        cs_bottom.set_label('Absolute error, one-phase training')

        axs[0,2*idx+1].contourf(X_np, T_np, u_pred_no_spec[lam], levels=u_levels, cmap='RdBu_r', label='U(x; $\\lambda$), two-phase training')
        axs[1,2*idx+1].contourf(X_np, T_np, abs_error_no_spec[lam], levels=err_levels, cmap='YlOrRd', label='Absolute error, two-phase training')

        #axs[0,2*lam_val.index(lam)].set_xlabel('x', fontsize=15)
        axs[0,2*lam_val.index(lam)].set_ylabel('t', fontsize=15)
        axs[0,2*lam_val.index(lam)].tick_params(axis='both', labelsize=12)
        #axs[0,2*lam_val.index(lam)].legend(fontsize=12)
        axs[0,2*lam_val.index(lam)].set_title(f"U(x; $\\lambda$), one-phase training, μ = {lam}", fontsize=15)
        
        axs[1,2*lam_val.index(lam)].set_xlabel('x', fontsize=15)
        axs[1,2*lam_val.index(lam)].set_ylabel('t', fontsize=15)
        axs[1,2*lam_val.index(lam)].tick_params(axis='both', labelsize=12)
        axs[1,2*lam_val.index(lam)].set_title(f"Absolute error, one-phase training, μ = {lam}", fontsize=15)
        #axs[1,2*lam_val.index(lam)].legend(fontsize=12)
        
        #axs[0,2*lam_val.index(lam)+1].set_xlabel('x', fontsize=15)
        axs[0,2*lam_val.index(lam)+1].set_ylabel('t', fontsize=15)
        axs[0,2*lam_val.index(lam)+1].tick_params(axis='both', labelsize=12)
        #axs[0,2*lam_val.index(lam)+1].legend(fontsize=12)
        axs[0,2*lam_val.index(lam)+1].set_title(f"U(x; $\\lambda$), two-phase training, μ = {lam}", fontsize=15)
        
        axs[1,2*lam_val.index(lam)+1].set_xlabel('x', fontsize=15)
        axs[1,2*lam_val.index(lam)+1].set_ylabel('t', fontsize=15)
        axs[1,2*lam_val.index(lam)+1].tick_params(axis='both', labelsize=12)
        axs[1,2*lam_val.index(lam)+1].set_title(f"Absolute error, two-phase training, μ = {lam}", fontsize=15)
        #axs[1,2*lam_val.index(lam)+1].legend(fontsize=12)
            

    if cs_top is not None:
        fig.colorbar(cs_top, ax=axs[0, :].ravel().tolist(), fraction=0.046, pad=0.01)
    if cs_bottom is not None:
        fig.colorbar(cs_bottom, ax=axs[1, :].ravel().tolist(), fraction=0.046, pad=0.01)

    plt.show()



def train(
    strategy: str,
    output_dir: Path,
    task_p: list[float] | None = None,
    head_epochs: int = 50_000,
    attn_epochs: int = 10_000,
    seed: int = 50,
) -> Path:
    """Train a heat-equation multitask PINN (one-phase or two-phase strategy)."""
    global device
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = task_p or TASK_P
    two_phase = strategy == "two-phase"

    if not two_phase:
        model = BatchedMultiHeadPINN_Heat_Attention(tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_heat_multitask_attention_head_spec(
            model,
            optimizer,
            epochs=head_epochs,
            attn_reg=0.1,
            task_p=tasks,
            device=device,
            loss_filename=str(output_dir / "loss_one_phase.txt"),
            model_save_path=str(output_dir / "evolutions/one_phase"),
            save_evolution_model=False,
        )
        model_path = output_dir / "model_one_phase.pth"
    else:
        model = BatchedMultiHeadPINN_Heat_Attention(tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_heat_multitask_head_spec_isolated_attention(
            model,
            optimizer,
            epochs=head_epochs,
            task_p=tasks,
            device=device,
            loss_filename=str(output_dir / "loss_heads_two_phase.txt"),
            model_save_path=str(output_dir / "evolutions/two_phase_heads"),
            save_evolution_model=False,
        )
        for param in model.shared.parameters():
            param.requires_grad = False
        for head in model.heads:
            for param in head.parameters():
                param.requires_grad = False
        for param in model.attn_net.parameters():
            param.requires_grad = True
        optimizer_attn = torch.optim.Adam(model.attn_net.parameters(), lr=1e-3)
        train_heat_pinn(
            model,
            optimizer_attn,
            task_p=tasks,
            epochs=attn_epochs,
            device=device,
            loss_filename=str(output_dir / "loss_voter_two_phase.txt"),
        )
        model_path = output_dir / "model_two_phase.pth"

    torch.save(model, model_path)
    torch.save(model.shared.state_dict(), output_dir / f"backbone_{model_path.name}")
    return model_path


def load_models(output_dir: Path) -> tuple:
    output_dir = Path(output_dir)
    model_two = torch.load(output_dir / "model_two_phase.pth", weights_only=False)
    model_one = torch.load(output_dir / "model_one_phase.pth", weights_only=False)
    return model_two, model_one


def predict(
    output_dir: Path,
    param_values: list[float] | None = None,
    nx: int = 100,
    nt: int = 100,
    t_max: float = 1.0,
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """Compare one-phase and two-phase models on interpolation parameters."""
    model_two, model_one = load_models(output_dir)
    visualize(
        model_two,
        model_one,
        lam_val=param_values or [0.15, 0.8],
        nx=nx,
        nt=nt,
        T_max=t_max,
    )
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close("all")