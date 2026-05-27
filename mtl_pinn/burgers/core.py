from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
from scipy.integrate import simpson

from mtl_pinn.utils import set_seed

# -------------------------
# Default experiment settings
# -------------------------
TASK_NU = [0.005, 0.01, 0.02, 0.03]
device = "cpu"


# -------------------------
# Analytical solution for Burgers' equation
# -------------------------
def burgers_analytical(x_vals, t, lambda_=1, nu=0.01/np.pi):
    y_vals = np.linspace(-1, 1, 1000)
    # Initial condition: u(x,0) = -sin(pi x)
    phi_0 = np.exp(-(lambda_ / (2 * nu)) * (1 / np.pi) * (np.cos(np.pi * y_vals) - 1))

    u_vals = []
    for x in x_vals:
        kernel = np.exp(-(x - y_vals) ** 2 / (4 * nu * t)) / np.sqrt(4 * np.pi * nu * t)
        theta = simpson(kernel * phi_0, y_vals)
        dtheta_dx = simpson((-(x - y_vals) / (2 * nu * t)) * kernel * phi_0, y_vals)
        u = -2 * nu / lambda_ * (dtheta_dx / theta)
        u_vals.append(u)

    return np.array(u_vals)


def plot_burgers_analytical_for_nu(nu, nx=100, nt=100, t_min=0.1, t_max=1.0,
                                   lambda_=1.0, levels=100, cmap='jet',
                                   file_save=None):
    """
    Plot the analytical solution of Burgers' equation for a given nu value.
    """
    x = np.linspace(-1, 1, nx)
    t = np.linspace(t_min, t_max, nt)
    X, T = np.meshgrid(x, t, indexing='ij')
    U = np.zeros((nx, nt))

    for j, t_val in enumerate(t):
        U[:, j] = burgers_analytical(x, t_val, lambda_=lambda_, nu=nu)

    plt.figure(figsize=(8, 5))
    contour = plt.contourf(X, T, U, levels, cmap=cmap)
    plt.colorbar(contour, label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f"Analytical Burgers Solution at μ = {nu}")
    plt.tight_layout()

    plt.savefig(file_save, format='pdf', bbox_inches='tight')
    plt.show()



# -------------------------
# MTNN with Multi-Heads
# -------------------------
class BatchedMultiHeadPINN_Burgers_Attention(nn.Module):
    """Learnable attention weighting for parametric Burgers' equation (nu only)."""
    def __init__(self, task_nu, hidden_dim=128, layers=5, gamma=20.0, attn_hidden=64):
        super().__init__()
        self.task_nu = torch.tensor(task_nu, dtype=torch.float32).view(1, -1).to(device)
        self.gamma = gamma
        # Shared layers: input is (x, t)
        self.shared = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            *[layer for _ in range(layers - 1)
              for layer in (nn.Linear(hidden_dim, hidden_dim), nn.Tanh())]
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in task_nu])
        # Attention network: input is (nu, task_nu) pair
        self.attn_net = nn.Sequential(
            nn.Linear(2, attn_hidden), nn.ReLU(),
            nn.Linear(attn_hidden, attn_hidden), nn.ReLU(),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, x, t, nu, mode='inference'):
        # x: (N, 1), t: (N, 1), nu: (N, 1)
        z = self.shared(torch.cat([x, t], dim=1))  # (N, D)
        head_outputs = torch.stack([head(z) for head in self.heads], dim=2)  # (N, 1, T)
        if mode in ['train', 'inference']:
            N = nu.shape[0]
            T = self.task_nu.shape[1]
            nu_expanded = nu.expand(-1, T)  # (N, T)
            task_nu_expanded = self.task_nu.expand(N, -1)  # (N, T)
            attn_inputs = torch.stack([nu_expanded, task_nu_expanded], dim=2)  # (N, T, 2)
            attn_inputs = attn_inputs.view(-1, 2)  # (N*T, 2)
            attn_logits = self.attn_net(attn_inputs).view(N, T)  # (N, T)
            weights = torch.softmax(attn_logits, dim=1)  # (N, T)
            # weights = attn_logits
            u_out = (head_outputs * weights.unsqueeze(1)).sum(dim=2)
            return u_out
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def get_attention_weights(self, x, t, nu):
        z = self.shared(torch.cat([x, t], dim=1)) # No se usa
        N = nu.shape[0]
        T = self.task_nu.shape[1]
        nu_expanded = nu.expand(-1, T)  # (N, T)
        task_nu_expanded = self.task_nu.expand(N, -1)  # (N, T)
        attn_inputs = torch.stack([nu_expanded, task_nu_expanded], dim=2)  # (N, T, 2)
        attn_inputs = attn_inputs.view(-1, 2)  # (N*T, 2)
        attn_logits = self.attn_net(attn_inputs).view(N, T)  # (N, T)
        weights = torch.softmax(attn_logits, dim=1)  # (N, T)
        # weights = attn_logits
        return weights

    def forward_head(self, x, t, head_idx):
        z = self.shared(torch.cat([x, t], dim=1))
        return self.heads[head_idx](z)




# -------------------------
# -------------------------
@torch.no_grad()
def plot_pred_and_real(model, nu=0.01, nx=100, nt=100):
    """
    Plot the PINN prediction and the analytical solution for Burgers' equation
    with nu as the only parameter.
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    nu_input = torch.full_like(x_input, nu)

    # PINN prediction
    u_pred = model(x_input, t_input, nu_input, mode='inference').reshape(nx, nt).cpu().numpy()

    # Analytical solution (lambda=1, nu=nu)
    u_vals = []
    for t_val in t_input.unique():
        mask = (t_input == t_val).squeeze()
        x_vals = x_input[mask].cpu().numpy().flatten()
        t_scalar = t_val.item()
        u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu)
        u_vals.append(u_vals_t)
    u_real = np.array(u_vals).T  # shape (nx, nt)

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.contourf(X.cpu(), T.cpu(), u_pred, 100, cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f"PINN Prediction at μ = {nu}")
    plt.subplot(1, 2, 2)
    plt.contourf(X.cpu(), T.cpu(), u_real.reshape(nx, nt), 100, cmap='jet')
    plt.colorbar(label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f"Analytical Solution at μ = {nu}")
    plt.tight_layout()
    plt.suptitle(f"PINN vs Analytical Solution at μ = {nu}", y=1.05)
    plt.subplots_adjust(top=0.85)
    plt.show()

@torch.no_grad()
def plot_abs_error_pred_vs_real(model, nu=0.01, nx=100, nt=100):
    """
    Plot the absolute error between the PINN prediction and the analytical solution
    for the parametric Burgers' equation, and print mean absolute and mean squared error.
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    nu_input = torch.full_like(x_input, nu)

    # PINN prediction
    u_pred = model(x_input, t_input, nu_input, mode='inference').reshape(nx, nt).cpu().numpy()

    # Analytical solution (lambda=1, nu=nu)
    u_vals = []
    for t_val in t_input.unique():
        mask = (t_input == t_val).squeeze()
        x_vals = x_input[mask].cpu().numpy().flatten()
        t_scalar = t_val.item()
        u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu)
        u_vals.append(u_vals_t)
    u_real = np.array(u_vals).T  # shape (nx, nt)

    # Absolute error
    error = np.abs(u_pred - u_real)
    # Mean squared error
    mse = np.mean((u_pred - u_real) ** 2)

    # Plot error
    plt.figure(figsize=(7, 5))
    plt.contourf(X.cpu(), T.cpu(), error, 100, cmap='hot')
    plt.colorbar(label='|u_pred - u_real|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Absolute Error |u_pred - u_real| at μ = {nu}')
    plt.tight_layout()
    plt.show()

    print(f"Mean absolute error: {np.mean(error):.4e}")
    print(f"Mean squared error: {mse:.4e}")

@torch.no_grad()
def plot_relative_error_pred_vs_real(model, nu=0.01, nx=100, nt=100, eps=1e-8):
    """
    Plot the relative error between the PINN prediction and the analytical solution
    for the parametric Burgers' equation, and print mean relative and mean squared error.
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    nu_input = torch.full_like(x_input, nu)

    # PINN prediction
    u_pred = model(x_input, t_input, nu_input, mode='inference').reshape(nx, nt).cpu().numpy()

    # Analytical solution (lambda=1, nu=nu)
    u_vals = []
    for t_val in t_input.unique():
        mask = (t_input == t_val).squeeze()
        x_vals = x_input[mask].cpu().numpy().flatten()
        t_scalar = t_val.item()
        u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu)
        u_vals.append(u_vals_t)
    u_real = np.array(u_vals).T  # shape (nx, nt)

    # Relative error
    relative_error = 100 * np.abs(u_pred - u_real) / (np.abs(u_real) + eps)
    mse = np.mean((u_pred - u_real) ** 2)

    # Plot relative error
    plt.figure(figsize=(7, 5))
    plt.contourf(X.cpu(), T.cpu(), relative_error, 100, cmap='hot')
    plt.colorbar(label='Relative Error % |u_pred - u_real| / |u_real|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Relative Error at μ = {nu}')
    plt.tight_layout()
    plt.show()

    print(f"Mean relative error: {np.mean(relative_error):.4e}")
    print(f"Mean squared error: {mse:.4e}")

@torch.no_grad()
def plot_relative_error_above_threshold(model, nu=0.01, nx=100, nt=100, threshold=30, eps=1e-8):
    """
    Plot the relative error and highlight points where the relative error > threshold (%).
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    nu_input = torch.full_like(x_input, nu)

    # PINN prediction
    u_pred = model(x_input, t_input, nu_input, mode='inference').reshape(nx, nt).cpu().numpy()

    # Analytical solution (lambda=1, nu=nu)
    u_vals = []
    for t_val in t_input.unique():
        mask = (t_input == t_val).squeeze()
        x_vals = x_input[mask].cpu().numpy().flatten()
        t_scalar = t_val.item()
        u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu)
        u_vals.append(u_vals_t)
    u_real = np.array(u_vals).T  # shape (nx, nt)

    # Relative error (%)
    relative_error = 100 * np.abs(u_pred - u_real) / (np.abs(u_real) + eps)

    # Mask for points above threshold
    mask = relative_error > threshold

    # Plot relative error
    plt.figure(figsize=(7, 5))
    plt.contourf(X.cpu(), T.cpu(), relative_error, 100, cmap='hot')
    plt.colorbar(label='Relative Error %')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Relative Error at μ = {nu}')

    # Overlay points where relative error > threshold
    plt.scatter(X.cpu()[mask], T.cpu()[mask], color='blue', s=10, label=f'> {threshold}% error')

    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Points with relative error > {threshold}%: {np.sum(mask)}")
    print(f"Mean relative error: {np.mean(relative_error):.4e}")

@torch.no_grad()
def plot_each_head_pred(model, nu=0.01, nx=100, nt=100, file_save=None):
    """
    Plot the output of each head over the (x, t) grid for a given nu value.
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    nu_input = torch.full_like(x_input, nu)

    n_heads = len(model.heads)
    plt.figure(figsize=(4 * n_heads, 4))
    for i in range(n_heads):
        # Evaluate only head i
        u_head = model.forward_head(x_input, t_input, i).reshape(nx, nt).cpu().numpy()
        plt.subplot(1, n_heads, i + 1)
        plt.contourf(X.cpu(), T.cpu(), u_head, 100, cmap='jet')
        plt.colorbar(label=f'Head {i} output')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'Head {i} output (μ = {nu})')
    plt.tight_layout()

    plt.show()

@torch.no_grad()
def plot_heads_vs_true_one_to_one(model_spec, model_no_spec, nx=30, nt=30, task_nu=None):
    """
    For each head, plot its output and the true solution for the corresponding nu.
    Head 0 vs nu=task_nu[0], Head 1 vs nu=task_nu[1], etc. (lambda is always 1)
    """
    if task_nu is None:
        task_nu = TASK_NU
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    # print(f"t_input: {t_input}")

    n_heads = min(10, len(model_spec.heads))
    
    mins = []
    maxs = []
    
    for i in range(n_heads):
        nu_val = task_nu[i]

        # Head output
        u_head_spec = model_spec.forward_head(x_input, t_input, i).reshape(nx, nt).cpu().numpy()
        u_head_no_spec = model_no_spec.forward_head(x_input, t_input, i).reshape(nx, nt).cpu().numpy()
        
        min_val = min(u_head_spec.min(), u_head_no_spec.min())
        max_val = max(u_head_spec.max(), u_head_no_spec.max())

        mins.append(min_val)
        maxs.append(max_val)
    
    min_global = -1#min(mins)
    max_global = 1#max(maxs)

    fig = plt.figure(figsize=(12, 8))
    axes = fig.subplots(3, 4)
    for i in range(n_heads):
        nu_val = task_nu[i]

        # Head output
        u_head_spec = model_spec.forward_head(x_input, t_input, i).reshape(nx, nt).cpu().numpy()
        u_head_no_spec = model_no_spec.forward_head(x_input, t_input, i).reshape(nx, nt).cpu().numpy()
        
        # True solution (lambda=1, nu=nu_val)
        u_vals = []
        for t_val in t_input.unique():
            mask = (t_input == t_val).squeeze()
            x_vals = x_input[mask].cpu().numpy().flatten()
            t_scalar = t_val.item()
            u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu_val)
            u_vals.append(u_vals_t)
        u_true = np.array(u_vals).T  # shape (nx, nt)

        axes[0,i].contourf(X.cpu(), T.cpu(), u_head_spec, levels=100, cmap='RdBu_r', vmin=min_global, vmax=max_global)
        axes[0,i].set_title(f'Head {i} output (μ = {nu_val}), two-phase training')
        if i==0:
            axes[0,i].set_ylabel('t', fontsize=15)

        axes[0,i].tick_params(axis='both', labelsize=12)

        #plt.subplot(3, n_heads, n_heads + i + 1)
        axes[1,i].contourf(X.cpu(), T.cpu(), u_head_no_spec, levels=200, cmap='RdBu_r', vmin=min_global, vmax=max_global)
        axes[1,i].set_title(f'Head {i} output (μ = {nu_val}), one-phase training')
        if i==0:
            axes[1,i].set_ylabel('t', fontsize=15)

        axes[1,i].tick_params(axis='both', labelsize=12)

        # Plot true solution
        #plt.subplot(3, n_heads, 2*n_heads + i + 1)
        contour = axes[2,i].contourf(X.cpu(), T.cpu(), u_true, levels=100, cmap='RdBu_r', vmin=min_global, vmax=max_global)
        axes[2,i].set_title(f'Analytical solution (μ = {nu_val})')
        axes[2,i].tick_params(axis='both', labelsize=12)
        
        axes[2,i].set_xlabel('x', fontsize=15)
        if i==0:
            axes[2,i].set_ylabel('t', fontsize=15)
        
    cmap = matplotlib.cm.RdBu_r
    norm = matplotlib.colors.Normalize(vmin=min_global, vmax=max_global)

    fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, location='right', orientation='vertical',pad=0.02)
    #plt.tight_layout()
    plt.show()


@torch.no_grad()
def plot_true_solutions_one_to_one(nx=30, nt=30, task_nu=None, file_save=None):
    """
    Plot only the true analytical solution for each training nu.
    """
    if task_nu is None:
        task_nu = TASK_NU

    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)

    n_nu = min(10, len(task_nu))
    plt.figure(figsize=(4 * n_nu, 4))
    for i in range(n_nu):
        nu_val = task_nu[i]

        u_vals = []
        for t_val in t_input.unique():
            mask = (t_input == t_val).squeeze()
            x_vals = x_input[mask].cpu().numpy().flatten()
            t_scalar = t_val.item()
            u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu_val)
            u_vals.append(u_vals_t)
        u_true = np.array(u_vals).T

        plt.subplot(1, n_nu, i + 1)
        plt.contourf(X.cpu(), T.cpu(), u_true, 100, cmap='jet')
        plt.colorbar(label='True solution')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'μ = {nu_val}')

    plt.tight_layout()
    plt.suptitle("Analytical solutions for different μ values")
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.savefig(file_save, format="pdf", bbox_inches="tight")
    plt.show()

@torch.no_grad()
def plot_heads_only_one_to_one(model, nx=30, nt=30, task_nu=None, file_save=None):
    """
    Plot only the output of each head for the corresponding training nu.
    Head 0 vs nu=task_nu[0], Head 1 vs nu=task_nu[1], etc.
    """
    if task_nu is None:
        task_nu = TASK_NU

    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)

    n_heads = min(10, len(model.heads), len(task_nu))
    plt.figure(figsize=(4 * n_heads, 4))
    for i in range(n_heads):
        nu_val = task_nu[i]
        u_head = model.forward_head(x_input, t_input, i).reshape(nx, nt).cpu().numpy()

        plt.subplot(1, n_heads, i + 1)
        plt.contourf(X.cpu(), T.cpu(), u_head, 100, cmap='jet')
        plt.colorbar(label=f'Head {i} output')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title(f'Head {i} output (μ = {nu_val})')

    plt.tight_layout()
    plt.suptitle("Heads obtained after training")
    plt.tight_layout(rect=[0, 0, 1, 1])

    if file_save is not None:
        plt.savefig(file_save, format="pdf", bbox_inches="tight")
    plt.show()

@torch.no_grad()
def plot_model_approximation_for_nu(model, nu, nx=100, nt=100, t_min=0.1, t_max=1.0,
                                    levels=100, cmap='jet', file_save=None):
    """
    Plot the model approximation of Burgers' equation for a given nu value.
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(t_min, t_max, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    nu_input = torch.full_like(x_input, nu)

    u_pred = model(x_input, t_input, nu_input, mode='inference').reshape(nx, nt).cpu().numpy()

    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()

    plt.figure(figsize=(8, 5))
    contour = plt.contourf(X_np, T_np, u_pred, levels, cmap=cmap)
    plt.colorbar(contour, label='u(x,t)')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f"Model Approximation at μ = {nu}")
    plt.tight_layout()

    plt.savefig(file_save, format='pdf', bbox_inches='tight')
    plt.show()

    return X_np, T_np, u_pred


@torch.no_grad()
def plot_absolute_error_burgers(model, nu=0.01, nx=100, nt=100, t_min=0.1, t_max=1.0,
                                lambda_=1.0, cmap='hot', file_save=None):
    """
    Compute and plot the absolute error between the model approximation and
    the analytical Burgers solution for a given nu.

    Returns:
        X, T         : grid coordinates as numpy arrays
        abs_error    : pointwise absolute error on the grid
        mae          : mean absolute error over the grid
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(t_min, t_max, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')

    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    nu_input = torch.full_like(x_input, float(nu))

    u_pred = model(x_input, t_input, nu_input, mode='inference').reshape(nx, nt).cpu().numpy()

    X_np = X.cpu().numpy()
    T_np = T.cpu().numpy()
    u_true = np.zeros((nx, nt))
    t_values = t.squeeze().cpu().numpy()
    x_values = x.squeeze().cpu().numpy()

    for j, t_val in enumerate(t_values):
        u_true[:, j] = burgers_analytical(x_values, t_val, lambda_=lambda_, nu=nu)

    abs_error = np.abs(u_pred - u_true)
    mae = np.mean(abs_error)

    plt.figure(figsize=(8, 5))
    contour = plt.contourf(X_np, T_np, abs_error, 100, cmap=cmap)
    plt.colorbar(contour, label='|u_pred - u_true|')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'Absolute Error of the Approximation (μ = {nu})')
    plt.tight_layout()

    plt.savefig(file_save, format='pdf', bbox_inches='tight')
    plt.show()
    print(f"MAE for μ = {nu}: {mae:.6e}")





# Function to plot multiple loss evolution curves from txt files
def plot_multiple_loss_evolutions(loss_files, labels=None):
    """
    Plot multiple loss evolution curves from txt files in the same figure.
    Args:
        loss_files: List of file paths to loss txt files.
        labels: List of labels for each curve (optional).
    """
    plt.figure(figsize=(8, 5))
    for i, loss_file in enumerate(loss_files):
        data = np.loadtxt(loss_file)
        losses = data
        epochs = np.arange(1, len(losses) + 1)
        label = labels[i] if labels is not None else f"Run {i+1}"
        if i == 1:
            plt.plot(epochs, losses, lw=2, label=label, alpha=0.5)  # Dashed line for the second curve
        else:
            plt.plot(epochs, losses, lw=2, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Evolution Comparison')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()



@torch.no_grad()
def plot_burgers_heads_grid_for_saved_models(model_class, model_paths, task_nu=TASK_NU, nx=50, nt=50, device='cpu'):
    """
    For each saved Burgers model, plot the output of each head and the true solution for each training nu.
    Arranges the plots in two grids: each with 5 rows (models) and 5 columns (heads).
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    n_models = len(model_paths)
    n_heads = len(task_nu)
    n_figs = 2
    models_per_fig = n_models // n_figs

    for fig_idx in range(n_figs):
        fig, axes = plt.subplots(models_per_fig, n_heads, figsize=(4*n_heads, 2.5*models_per_fig), sharex=True, sharey=True)
        if models_per_fig == 1:
            axes = axes[np.newaxis, :]
        if n_heads == 1:
            axes = axes[:, np.newaxis]
        for row in range(models_per_fig):
            model_idx = fig_idx * models_per_fig + row
            model_path = model_paths[model_idx]
            model = model_class(task_nu).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            for col, nu_val in enumerate(task_nu):
                ax = axes[row, col]
                nu_input = torch.full_like(x_input, nu_val)
                u_head = model.forward_head(x_input, t_input, col).reshape(nx, nt).cpu().numpy()
                # True solution
                u_vals = []
                for t_val in t_input.unique():
                    mask = (t_input == t_val).squeeze()
                    x_vals = x_input[mask].cpu().numpy().flatten()
                    t_scalar = t_val.item()
                    u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu_val)
                    u_vals.append(u_vals_t)
                u_true = np.array(u_vals).T  # shape (nx, nt)
                # Plot head output and true solution contours
                cf = ax.contourf(X.cpu(), T.cpu(), u_head, 50, cmap='jet', alpha=0.7)
                # cs = ax.contour(X.cpu(), T.cpu(), u_true, 10, colors='k', linewidths=0.7)
                if row == 0:
                    ax.set_title(f'Head {col+1}\nμ={nu_val}')
                if col == 0:
                    # Show model name or epoch in the first column
                    model_label = model_path.split('/')[-1]
                    ax.set_ylabel(f'{model_label}', fontsize=9)
                ax.set_xlabel('x')
                ax.set_ylabel('t')
        plt.tight_layout()
        plt.suptitle(f'Burgers PINN Heads - Models {fig_idx*models_per_fig+1} to {(fig_idx+1)*models_per_fig}', y=1.02, fontsize=16)
        plt.subplots_adjust(top=0.92)
        plt.show()

@torch.no_grad()
def plot_true_and_heads_comparison_grid(
    model_class, model_A_paths, model_B_paths, task_nu=TASK_NU, nx=50, nt=50, device='cpu'
):
    """
    For each epoch, plot a 3xN_heads grid:
    Row 1: True solution for each nu
    Row 2: Heads of model A (specialized, two-stage)
    Row 3: Heads of model B (no specialization, all-together)
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    n_heads = len(task_nu)
    n_models = len(model_A_paths)
    assert len(model_A_paths) == len(model_B_paths), "Model A and B path lists must have the same length"

    for idx in range(n_models):
        title = (idx + 1) * 3000
        # Load models for this epoch
        model_A = model_class(task_nu).to(device)
        model_A.load_state_dict(torch.load(model_A_paths[idx], map_location=device))
        model_A.eval()
        model_B = model_class(task_nu).to(device)
        model_B.load_state_dict(torch.load(model_B_paths[idx], map_location=device))
        model_B.eval()

        fig, axes = plt.subplots(3, n_heads, figsize=(4 * n_heads, 9), sharex=True, sharey=True)
        for col, nu_val in enumerate(task_nu):
            nu_input = torch.full_like(x_input, nu_val)

            # --- Row 1: True solution ---
            u_vals = []
            for t_val in t_input.unique():
                mask = (t_input == t_val).squeeze()
                x_vals = x_input[mask].cpu().numpy().flatten()
                t_scalar = t_val.item()
                u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu_val)
                u_vals.append(u_vals_t)
            u_true = np.array(u_vals).T  # shape (nx, nt)
            ax = axes[0, col]
            cf = ax.contourf(X.cpu(), T.cpu(), u_true, 50, cmap='jet')
            ax.set_title(f'μ={nu_val}')
            if col == 0:
                ax.set_ylabel('True')

            # --- Row 2: Model A head ---
            u_head_A = model_A.forward_head(x_input, t_input, col).reshape(nx, nt).cpu().numpy()
            ax = axes[1, col]
            cf = ax.contourf(X.cpu(), T.cpu(), u_head_A, 50, cmap='jet')
            if col == 0:
                ax.set_ylabel(f'Model A\n({model_A_paths[idx].split("/")[-1]})')

            # --- Row 3: Model B head ---
            u_head_B = model_B.forward_head(x_input, t_input, col).reshape(nx, nt).cpu().numpy()
            ax = axes[2, col]
            cf = ax.contourf(X.cpu(), T.cpu(), u_head_B, 50, cmap='jet')
            if col == 0:
                ax.set_ylabel(f'Model B\n({model_B_paths[idx].split("/")[-1]})')

            for row in range(3):
                axes[row, col].set_xlabel('x')
                axes[row, col].set_ylabel('t')

        plt.tight_layout()
        plt.suptitle(f'Epoch {title}', y=0.97, fontsize=16)
        plt.subplots_adjust(top=0.90)
        plt.show()

@torch.no_grad()
def plot_error_difference(model1, model2, nu=0.01, nx=100, nt=100, error_type='relative', eps=1e-8):
    """
    Compute and plot the difference of approximation errors (model1 - model2) for two models.
    nu is the parameter (lambda is always 1 in analytical solution).
    error_type: 'absolute' or 'relative'
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    t = torch.linspace(0.1, 1, nt).reshape(-1, 1).to(device)
    X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
    x_input = X.reshape(-1, 1)
    t_input = T.reshape(-1, 1)
    nu_input = torch.full_like(x_input, nu)

    # Predictions
    u_pred1 = model1(x_input, t_input, nu_input, mode='inference').reshape(nx, nt).cpu().numpy()
    u_pred2 = model2(x_input, t_input, nu_input, mode='inference').reshape(nx, nt).cpu().numpy()

    # Analytical solution (lambda=1, nu=nu)
    u_vals = []
    for t_val in t_input.unique():
        mask = (t_input == t_val).squeeze()
        x_vals = x_input[mask].cpu().numpy().flatten()
        t_scalar = t_val.item()
        u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, nu)
        u_vals.append(u_vals_t)
    u_real = np.array(u_vals).T  # shape (nx, nt)

    # Error grids
    if error_type == 'absolute':
        error1 = np.abs(u_pred1 - u_real)
        error2 = np.abs(u_pred2 - u_real)
        label = 'Absolute Error Difference'
    else:
        error1 = 100 * np.abs(u_pred1 - u_real) / (np.abs(u_real) + eps)
        error2 = 100 * np.abs(u_pred2 - u_real) / (np.abs(u_real) + eps)
        label = 'Relative Error Difference (%)'

    diff = error1 - error2
    vmax = np.max(np.abs(diff))

    plt.figure(figsize=(7, 5))
    plt.contourf(X.cpu(), T.cpu(), diff, 100, cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar(label=label)
    plt.xlabel('x')
    plt.ylabel('t')
    plt.title(f'{label} (Head Spec - Head NO Spec) at μ = {nu}')
    plt.tight_layout()
    plt.show()

    print(f"Mean {label}: {np.mean(diff):.4e}")





# -------------------------
# TRAINING FUNCTIONS
# -------------------------
# HEADS AND ATTENTION NETWORK TRAINED TOGETHER. HEADS ARE NOT PREVIOUSLY SPECIALIZED.
def train_burgers_multitask_attention_head_spec(
    model, optimizer, epochs=3000, task_nu=None, weights=None, attn_reg=0.0, device='cpu',
    loss_filename="loss_evolution.txt", model_save_path="Burger_nu/model", save_evolution_model=False):
    """
    Multi-head PINN training for parametric Burgers' equation:
    - Each head specializes in one nu (viscosity),
    - The attention network is trained as well.
    - Optionally, add a regularization to encourage attention to focus on the correct head.
    - Saves loss evolution to a txt file and model every 3000 epochs.
    """
    if task_nu is None:
        task_nu = [0.005, 0.01, 0.02, 0.03]
    if weights is None:
        weights = [1.0] * len(task_nu)
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        for i, nu_val in enumerate(task_nu):
            # --- Residual points for this head/nu ---
            N_r = 400
            x_r = torch.rand((N_r, 1), device=device) * 2 - 1
            t_r = torch.rand((N_r, 1), device=device) * 1.0  # t in [0, 1]
            nu_r = torch.full((N_r, 1), nu_val, dtype=torch.float32, device=device)
            x_r.requires_grad_(True)
            t_r.requires_grad_(True)

            # --- PDE residual: u_t + u * u_x - nu * u_xx = 0 ---
            u = model(x_r, t_r, nu_r, mode='inference')
            u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
            u_t = autograd.grad(u, t_r, torch.ones_like(u), create_graph=True)[0]
            u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
            residual = u_t + u * u_x - nu_val * u_xx
            loss_r = torch.mean(residual ** 2)

            # --- Initial condition: u(x, 0) = -sin(pi x) ---
            N_ic = 100
            x_ic = torch.rand((N_ic, 1), device=device) * 2 - 1
            t_ic = torch.zeros_like(x_ic)
            nu_ic = torch.full((N_ic, 1), nu_val, dtype=torch.float32, device=device)
            u_ic_pred = model(x_ic, t_ic, nu_ic, mode='inference')
            u_ic_true = -torch.sin(np.pi * x_ic)
            loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

            # --- Boundary condition: u(-1, t) = u(1, t) = 0 ---
            N_bc = 100
            t_bc = torch.rand((N_bc, 1), device=device) * 1.0
            x_l = -torch.ones_like(t_bc)
            x_r_bc = torch.ones_like(t_bc)
            nu_bc = torch.full((N_bc, 1), nu_val, dtype=torch.float32, device=device)
            u_l = model(x_l, t_bc, nu_bc, mode='inference')
            u_r = model(x_r_bc, t_bc, nu_bc, mode='inference')
            loss_bc = torch.mean(u_l ** 2 + u_r ** 2)

            # --- (Optional) Regularization: encourage attention to focus on the correct head ---
            attn_reg_loss = 0.0
            if attn_reg > 0 and hasattr(model, "get_attention_weights"):
                att_weights = model.get_attention_weights(x_r, t_r, nu_r)
                target = torch.zeros_like(att_weights)
                target[:, i] = 1.0
                attn_reg_loss = attn_reg * torch.mean((att_weights - target) ** 2)

            # --- Weighted sum for this head ---
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
def train_burgers_multitask_head_spec_isolated_attention(
    model, optimizer, epochs=3000, task_nu=None, weights=None, device=None,
    loss_filename="loss_evolution.txt", model_save_path="Burger_nu/model", save_evolution_model=False
):
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
    if task_nu is None:
        task_nu = [0.005, 0.01, 0.02, 0.03]
    if weights is None:
        weights = [1.0] * len(task_nu)
    loss_history = []

    def compute_task_loss(i, nu_val):
        N_r = 400
        x_r = torch.rand((N_r, 1), device=device) * 2 - 1
        t_r = torch.rand((N_r, 1), device=device) * 1.0
        x_r.requires_grad_(True)
        t_r.requires_grad_(True)

        z = model.shared(torch.cat([x_r, t_r], dim=1))
        u = model.heads[i](z)
        u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
        u_t = autograd.grad(u, t_r, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
        residual = u_t + u * u_x - nu_val * u_xx
        loss_r = torch.mean(residual ** 2)

        N_ic = 100
        x_ic = torch.rand((N_ic, 1), device=device) * 2 - 1
        t_ic = torch.zeros_like(x_ic)
        z_ic = model.shared(torch.cat([x_ic, t_ic], dim=1))
        u_ic_pred = model.heads[i](z_ic)
        u_ic_true = -torch.sin(np.pi * x_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

        N_bc = 100
        t_bc = torch.rand((N_bc, 1), device=device) * 1.0
        x_l = -torch.ones_like(t_bc)
        x_r_bc = torch.ones_like(t_bc)
        z_l = model.shared(torch.cat([x_l, t_bc], dim=1))
        z_r = model.shared(torch.cat([x_r_bc, t_bc], dim=1))
        u_l = model.heads[i](z_l)
        u_r = model.heads[i](z_r)
        loss_bc = torch.mean(u_l ** 2 + u_r ** 2)

        return weights[i] * (loss_r + loss_ic + loss_bc)

    n_tasks = len(task_nu)
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
        task_losses = [compute_task_loss(i, nu_val) for i, nu_val in enumerate(task_nu)]

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
        for i, _ in enumerate(task_nu):
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
def train_burgers_pinn(
    model, optimizer, epochs=3000, task_nu=None, weights=None, device='cpu', loss_filename="loss_evolution.txt"):
    """
    Train a PINN (single or multi-head/attention) for the parametric Burgers' equation:
        u_t + u * u_x - nu * u_xx = 0
    with initial condition u(x,0) = -sin(pi x) and boundary conditions u(-1,t) = u(1,t) = 0.
    Only nu is the parameter.
    """
    loss_history = []
    if task_nu is None:
        task_nu = [0.005, 0.01, 0.02, 0.03]
    if weights is None:
        weights = [1.0] * len(task_nu)
    for epoch in range(epochs):
        optimizer.zero_grad()
        # --- Residual points (sampled for all nu) ---
        N_r = 400
        x_r = torch.rand((N_r, 1), device=device) * 2 - 1
        t_r = torch.rand((N_r, 1), device=device) * 1.0
        nu_r_np = np.random.choice(task_nu, size=(N_r, 1), replace=True)
        nu_r = torch.tensor(nu_r_np, dtype=torch.float32, device=device)
        x_r.requires_grad_(True)
        t_r.requires_grad_(True)

        # --- PDE residual: u_t + u * u_x - nu * u_xx = 0 ---
        u = model(x_r, t_r, nu_r, mode='train')
        u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
        u_t = autograd.grad(u, t_r, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
        residual = u_t + u * u_x - nu_r * u_xx
        loss_r = torch.mean(residual ** 2)

        # --- Initial condition: u(x, 0) = -sin(pi x) ---
        N_ic = 100
        x_ic = torch.rand((N_ic, 1), device=device) * 2 - 1
        t_ic = torch.zeros_like(x_ic)
        nu_ic_np = np.random.choice(task_nu, size=(N_ic, 1), replace=True)
        nu_ic = torch.tensor(nu_ic_np, dtype=torch.float32, device=device)
        u_ic_pred = model(x_ic, t_ic, nu_ic, mode='train')
        u_ic_true = -torch.sin(np.pi * x_ic)
        loss_ic = torch.mean((u_ic_pred - u_ic_true) ** 2)

        # --- Boundary conditions: u(-1, t) = u(1, t) = 0 ---
        N_bc = 100
        t_bc = torch.rand((N_bc, 1), device=device) * 1.0
        nu_bc_np = np.random.choice(task_nu, size=(N_bc, 1), replace=True)
        nu_bc = torch.tensor(nu_bc_np, dtype=torch.float32, device=device)
        u_l = model(-torch.ones_like(t_bc), t_bc, nu_bc, mode='train')
        u_r = model(torch.ones_like(t_bc), t_bc, nu_bc, mode='train')
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
        for loss in loss_history:
            f.write(f"{loss}\n")

@torch.no_grad()
def visualize(model_spec, model_no_spec, lam_val=[1.0], nx=100, nt=100, T_max=1.0):

    x = torch.linspace(0.0, 1.0, nx, device=device).reshape(-1, 1)*2-1
    t = torch.linspace(0.1, T_max, nt, device=device).reshape(-1, 1)
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
        
        # Analytical solution (lambda=1, nu=nu)
        u_vals = []
        for t_val in t_input.unique():
            mask = (t_input == t_val).squeeze()
            x_vals = x_input[mask].cpu().numpy().flatten()
            t_scalar = t_val.item()
            u_vals_t = burgers_analytical(x_vals, t_scalar, 1.0, lam)
            u_vals.append(u_vals_t)
        u_true[lam] = np.array(u_vals).T  # shape (nx, nt)
        
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
    task_nu: list[float] | None = None,
    head_epochs: int = 100_000,
    attn_epochs: int = 20_000,
    seed: int = 50,
) -> Path:
    """Train a Burgers multitask PINN (one-phase or two-phase strategy)."""
    global device
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = task_nu or TASK_NU
    two_phase = strategy == "two-phase"

    if not two_phase:
        model = BatchedMultiHeadPINN_Burgers_Attention(tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_burgers_multitask_attention_head_spec(
            model,
            optimizer,
            epochs=head_epochs,
            attn_reg=0.1,
            device=device,
            loss_filename=str(output_dir / "loss_one_phase.txt"),
            model_save_path=str(output_dir / "evolutions/one_phase"),
            save_evolution_model=False,
        )
        model_path = output_dir / "model_one_phase.pth"
    else:
        model = BatchedMultiHeadPINN_Burgers_Attention(tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_burgers_multitask_head_spec_isolated_attention(
            model,
            optimizer,
            epochs=head_epochs,
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
        train_burgers_pinn(
            model,
            optimizer_attn,
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
    """Compare one-phase and two-phase models on viscosity values."""
    model_two, model_one = load_models(output_dir)
    visualize(
        model_two,
        model_one,
        lam_val=param_values or [0.015, 0.08],
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
