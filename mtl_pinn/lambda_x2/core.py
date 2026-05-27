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
TASK_LAMBDAS = [0.5, 1.5, 2.5, 3.5]
device = "cpu"




# -------------------------
# MTNN with Multi-Heads
# -------------------------
 
class BatchedMultiHeadPINN_Attention(nn.Module):
    """Learnable attention weighting."""
    def __init__(self, task_lambdas, hidden_dim=64, layers=3, gamma=20.0, attn_hidden=32):
        super().__init__()
        self.task_lambdas = torch.tensor(task_lambdas, dtype=torch.float32).view(1, -1).to(device)
        self.gamma = gamma
        self.shared = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.Tanh(),
            *[layer for _ in range(layers - 1)
              for layer in (nn.Linear(hidden_dim, hidden_dim), nn.Tanh())]
        )
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in task_lambdas])
        # Attention network: input is (lam, task_lambda) pair
        self.attn_net = nn.Sequential(
            nn.Linear(2, attn_hidden), nn.ReLU(),
            nn.Linear(attn_hidden, attn_hidden), nn.ReLU(),
            nn.Linear(attn_hidden, 1)
        )

    def forward(self, x, lam, mode='inference'):
        # Shared feature extraction
        z = self.shared(x)
        # Get outputs from ALL heads
        head_outputs = torch.stack([head(z) for head in self.heads], dim=2)  # (N, 1, T)
        if mode in ['train', 'inference']:
            # Compute attention weights based on (μ, μ_task) pairs
            N, _ = lam.shape
            T = self.task_lambdas.shape[1]
            # Create pairs: for each input μ, pair it with EACH task μ
            lam_expanded = lam.expand(-1, T)  # (N, T)
            task_lambdas_expanded = self.task_lambdas.expand(N, -1)  # (N, T)
            # Stack into (μ, μ_task) pairs
            attn_inputs = torch.stack([lam_expanded, task_lambdas_expanded], dim=2)  # (N, T, 2)
            attn_inputs = attn_inputs.view(-1, 2)  # (N*T, 2)
            # Attention network predicts which heads to use
            attn_logits = self.attn_net(attn_inputs).view(N, T)  # (N, T)
            weights = torch.softmax(attn_logits, dim=1)  # (N, T)
            # weights = attn_logits
            # Weighted combination of head outputs
            u_out = (head_outputs * weights.unsqueeze(1)).sum(dim=2)
            return u_out
        else:
            raise ValueError(f"Unsupported mode: {mode}")
    
    def get_attention_weights(self, x, lam):
        z = self.shared(x)
        N, _ = lam.shape
        T = self.task_lambdas.shape[1]
        lam_expanded = lam.expand(-1, T)  # (N, T)
        task_lambdas_expanded = self.task_lambdas.expand(N, -1)  # (N, T)
        attn_inputs = torch.stack([lam_expanded, task_lambdas_expanded], dim=2)  # (N, T, 2)
        attn_inputs = attn_inputs.view(-1, 2)  # (N*T, 2)
        attn_logits = self.attn_net(attn_inputs).view(N, T)  # (N, T)
        weights = torch.softmax(attn_logits, dim=1)  # (N, T)
        # weights = attn_logits
        return weights

    def forward_head(self, x, head_idx):
        z = self.shared(x)
        return self.heads[head_idx](z)


# -------------------------
# -------------------------
# def pde_residual(model, x, lam, nu=1.0, mode='train'):
#     """
#     PDE: -nu * u_xx + 2 * lambda = 0
#     Solution: u(x; lambda) = lambda * x^2
#     """
#     x.requires_grad_(True)

#     u = model(x, lam, mode=mode)           # (N, 1)
#     u_x = autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
#     u_xx = autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]

#     residual = -u_xx + 2 * lam
#     return residual



# -------------------------
##### LAS SIGUIENTES DOS FUNCIONES SON LAS QUE ME INTERESAN #####
# Model with head specialization, to train attention network as well
def train_lambda_x2_pinn_multitask_attention_head_spec(
    model, optimizer, epochs=3000, task_lambdas=TASK_LAMBDAS, weights=None, attn_reg=0.0, loss_filename="loss_evolution.txt"):
    """
    Multi-head PINN training: each head specializes in one lambda,
    but the attention network is trained as well.
    Optionally, add a regularization to encourage attention to focus on the correct head.
    Saves loss evolution to a txt file.
    """
    if weights is None:
        weights = [1.0] * len(task_lambdas)
    loss_history = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        for i, lam_val in enumerate(task_lambdas):
            # Residual points for this head/lambda
            N_r = 100
            x_r = torch.rand((N_r, 1), device=device) * 2 - 1
            lam_r = torch.full((N_r, 1), lam_val, dtype=torch.float32, device=device)
            x_r.requires_grad_(True)

            # Use the full model (with attention) for this μ
            u = model(x_r, lam_r, mode='inference')
            u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
            u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
            residual = -u_xx + 2 * lam_r
            loss_r = torch.mean(residual ** 2)

            # --- Left boundary: u(-1) = μ ---
            lam_bc = torch.full((100, 1), lam_val, dtype=torch.float32, device=device)
            x_l = -torch.ones_like(lam_bc)
            u_l = model(x_l, lam_bc, mode='inference')
            u_l_true = lam_bc
            loss_bc_left = torch.mean((u_l - u_l_true) ** 2)

            # --- Right boundary: u'(1) = 2μ ---
            x_r_bc = torch.ones_like(lam_bc)
            x_r_bc.requires_grad_(True)
            u_r = model(x_r_bc, lam_bc, mode='inference')
            u_r_x = autograd.grad(u_r, x_r_bc, torch.ones_like(u_r), create_graph=True)[0]
            u_r_x_true = 2 * lam_bc
            loss_bc_right = torch.mean((u_r_x - u_r_x_true) ** 2)

            loss_bc = loss_bc_left + loss_bc_right

            # (Optional) Regularization: encourage attention to focus on the correct head
            attn_reg_loss = 0.0
            if attn_reg > 0 and hasattr(model, "get_attention_weights"):
                att_weights = model.get_attention_weights(x_r, lam_r)
                # Encourage attention to be high on the i-th head for this μ
                target = torch.zeros_like(att_weights)
                target[:, i] = 1.0
                attn_reg_loss = attn_reg * torch.mean((att_weights - target) ** 2)

            total_loss = total_loss + weights[i] * (loss_r + loss_bc + attn_reg_loss)

        total_loss.backward()
        optimizer.step()

        # Save loss value
        loss_history.append(total_loss.item())

        if epoch % 500 == 0:
            print(f"Epoch {epoch} | Total Loss: {total_loss.item():.2e}")

    # Save loss history to txt file
    with open(loss_filename, "w") as f:
        for loss in loss_history:
            f.write(f"{loss}\n")


# Each head only affected by its own loss, shared layers updated by all losses. Attention network NOT trained. Attention network trained in second phase.
# def train_lambda_x2_pinn_multitask_head_spec_isolated_attention(
#     model, optimizer, epochs=3000, nu=1.0, task_lambdas=TASK_LAMBDAS, weights=None, 
#     loss_filename="loss_evolution.txt"):
#     """
#     Multi-head PINN training for attention model:
#     - Each head is specialized for one lambda (only its weights updated by its loss).
#     - Shared layers are updated by all losses.
#     - Attention network is NOT trained in this phase.
#     - Saves loss evolution to a txt file and model every 3000 epochs.
#     """
#     if weights is None:
#         weights = [1.0] * len(task_lambdas)
#     loss_history = []
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         total_loss = 0.0
#         for i, lam_val in enumerate(task_lambdas):
#             # Freeze all heads except the current one
#             for j, head in enumerate(model.heads):
#                 for param in head.parameters():
#                     param.requires_grad = (j == i)
#             # Shared always trainable
#             for param in model.shared.parameters():
#                 param.requires_grad = True
#             # Attention network frozen
#             for param in model.attn_net.parameters():
#                 param.requires_grad = False

#             # Residual points for this head/lambda
#             N_r = 100
#             x_r = torch.rand((N_r, 1), device=device) * 2 - 1
#             lam_r = torch.full((N_r, 1), lam_val, dtype=torch.float32, device=device)
#             x_r.requires_grad_(True)

#             # PDE residual using only head i
#             z = model.shared(x_r)
#             u = model.heads[i](z)
#             u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
#             u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
#             residual = -u_xx + 2 * lam_r
#             loss_r = torch.mean(residual ** 2)

#             # --- Left boundary: u(-1) = μ ---
#             lam_bc = torch.full((100, 1), lam_val, dtype=torch.float32, device=device)
#             x_l = -torch.ones_like(lam_bc)
#             z_l = model.shared(x_l)
#             u_l = model.heads[i](z_l)
#             u_l_true = lam_bc
#             loss_bc_left = torch.mean((u_l - u_l_true) ** 2)

#             # --- Right boundary: u'(1) = 2μ ---
#             x_r_bc = torch.ones_like(lam_bc)
#             x_r_bc.requires_grad_(True)
#             z_r = model.shared(x_r_bc)
#             u_r = model.heads[i](z_r)
#             u_r_x = autograd.grad(u_r, x_r_bc, torch.ones_like(u_r), create_graph=True)[0]
#             u_r_x_true = 2 * lam_bc
#             loss_bc_right = torch.mean((u_r_x - u_r_x_true) ** 2)

#             loss_bc = loss_bc_left + loss_bc_right

#             loss = weights[i] * (loss_r + loss_bc)
#             total_loss = total_loss + loss

#             # Backward for this head (accumulate gradients)
#             loss.backward(retain_graph=True)

#         optimizer.step()

#         # Restore all heads and attention to trainable for next epoch
#         for head in model.heads:
#             for param in head.parameters():
#                 param.requires_grad = True
#         for param in model.attn_net.parameters():
#             param.requires_grad = True

#         # Save loss value
#         loss_history.append(total_loss.item())

#         if epoch % 500 == 0:
#             print(f"Epoch {epoch} | Total Loss: {total_loss.item():.2e}")

#     # Save loss history to txt file
#     with open(loss_filename, "w") as f:
#         for loss in loss_history:
#             f.write(f"{loss}\n")


def train_lambda_x2_pinn_multitask_head_spec_isolated_attention(
    model, optimizer, epochs=3000, task_lambdas=None, weights=None, device='cpu',
    loss_filename="loss_heat_head_spec.txt"):
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
    if task_lambdas is None:
        task_lambdas = [0.5, 1.5, 2.5, 3.5]
    if weights is None:
        weights = [1.0] * len(task_lambdas)
    loss_history = []

    def compute_task_loss(i, lamda_val):
        # Residual points for this head/lambda
        N_r = 100
        x_r = torch.rand((N_r, 1), device=device) * 2 - 1
        x_r.requires_grad_(True)
        # PDE residual using only head i
        z = model.shared(x_r)
        u = model.heads[i](z)
        u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_r, torch.ones_like(u_x), create_graph=True)[0]
        residual = -u_xx + 2 * lamda_val
        loss_r = torch.mean(residual ** 2)

        # --- Left boundary: u(-1) = μ ---
        lam_bc = torch.full((100, 1), lamda_val, dtype=torch.float32, device=device)
        x_l = -torch.ones_like(lam_bc)
        z_l = model.shared(x_l)
        u_l = model.heads[i](z_l)
        u_l_true = lamda_val
        loss_bc_left = torch.mean((u_l - u_l_true) ** 2)

        # --- Right boundary: u'(1) = 2μ ---
        x_r_bc = torch.ones_like(lam_bc)
        x_r_bc.requires_grad_(True)
        z_r = model.shared(x_r_bc)
        u_r = model.heads[i](z_r)
        u_r_x = autograd.grad(u_r, x_r_bc, torch.ones_like(u_r), create_graph=True)[0]
        u_r_x_true = 2 * lamda_val
        loss_bc_right = torch.mean((u_r_x - u_r_x_true) ** 2)

        return weights[i] * (loss_r + loss_bc_left + loss_bc_right)

    n_tasks = len(task_lambdas)
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
        task_losses = [compute_task_loss(i, lamda_val) for i, lamda_val in enumerate(task_lambdas)]

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
        for i, _ in enumerate(task_lambdas):
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


    # Save loss history to txt file
    with open(loss_filename, "w") as f:
        for loss in loss_history:
            f.write(f"{loss}\n")


# Global model, no head specialization
def train_lambda_x2_pinn(model, optimizer, epochs=3000, nu=1.0, loss_filename="loss_evolution.txt"):
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Residual points
        N_r = 100
        x_r = torch.rand((N_r, 1), device=device) * 2 - 1
        lam_r_np = np.random.choice(TASK_LAMBDAS, size=(N_r, 1), replace=True)
        lam_r = torch.tensor(lam_r_np, dtype=torch.float32, device=device)

        x_r.requires_grad_(True)

        u = model(x_r, lam_r, mode='inference')           # (N, 1)
        u_x = autograd.grad(u, x_r, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_r, torch.ones_like(u), create_graph=True)[0]

        res = -u_xx + 2 * lam_r

        loss_r = torch.mean(res ** 2)

        # Boundary condition: u(-1) = 2μ, u(1) = 2μ
        lam_bc_np = np.random.choice(TASK_LAMBDAS, size=(N_r, 1), replace=True)
        lam_bc = torch.tensor(lam_bc_np, dtype=torch.float32, device=device)

        # --- Left boundary: u(-1) = μ ---
        x_l = -torch.ones_like(lam_bc)
        u_l = model(x_l, lam_bc, mode='inference')
        u_l_true = lam_bc
        loss_bc_left = torch.mean((u_l - u_l_true) ** 2)

        # --- Right boundary: u'(1) = 2μ ---
        x_r = torch.ones_like(lam_bc)
        x_r.requires_grad_(True)
        u_r = model(x_r, lam_bc, mode='inference')
        u_r_x = autograd.grad(u_r, x_r, torch.ones_like(u_r), create_graph=True)[0]
        u_r_x_true = 2 * lam_bc
        loss_bc_right = torch.mean((u_r_x - u_r_x_true) ** 2)

        loss_bc = loss_bc_left + loss_bc_right

        loss = loss_r + loss_bc
        loss.backward()
        optimizer.step()

        # Save loss value
        loss_history.append(loss.item())

        if epoch % 100 == 0:
            print(f"Epoch {epoch} | Residual: {loss_r.item():.2e}, Total: {loss.item():.2e}")

    # Save loss history to txt file
    with open(loss_filename, "w") as f:
        for loss_value in loss_history:
            f.write(f"{loss_value}\n")





def gaussian_weights(model, x_val, lam_val):
    """
    Get the Gaussian weights for a specific x and lambda.
    Args:
        model: The trained model.
        x_val: float or 1-element tensor, the x value.
        lam_val: float or 1-element tensor, the lambda value.
    Returns:
        weights: Tensor of shape (1, T), where T is the number of task lambdas.
    """
    x = torch.tensor([[x_val]], dtype=torch.float32).to(next(model.parameters()).device)
    lam = torch.tensor([[lam_val]], dtype=torch.float32).to(next(model.parameters()).device)
    return model.get_weights(x, lam)






# -------------------------
# -------------------------
@torch.no_grad()
def evaluate_single(model, x_val, lam_val):
    """
    Evaluate the model at a single x and lambda value.
    Args:
        model: The trained model.
        x_val: float or 1-element tensor, the x value.
        lam_val: float or 1-element tensor, the lambda value.
    Returns:
        u_pred: float, the predicted u(x, lambda).
    """
    x = torch.tensor([[x_val]], dtype=torch.float32).to(next(model.parameters()).device)
    lam = torch.tensor([[lam_val]], dtype=torch.float32).to(next(model.parameters()).device)
    model.eval()  # Set model to evaluation mode
    u_pred = model(x, lam, mode='inference').item()
    # Print attention weights if the model has that method
    if hasattr(model, "get_attention_weights"):
        att_weights = model.get_attention_weights(x, lam)
        print(f"Attention weights for x={x_val}, μ={lam_val}: {att_weights.cpu().numpy()}")
    return u_pred

@torch.no_grad()
def visualize(model_spec, model_no_spec, lam_val=[1.0], nx=100):

    u_pred_spec = {}
    u_pred_no_spec = {}
    u_true = {}
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    
    fig, axs = plt.subplots(1, 2*len(lam_val), figsize=(6, 4))
    colors = ["#1B4F9EFF", "#0C582AFF", "#7570B3FF", "#E7298AFF"]
    
    for lam in lam_val:
        lam_input = torch.full_like(x, lam)

        u_pred_spec[lam] = model_spec(x, lam_input, mode='inference').reshape(-1).cpu()
        u_pred_no_spec[lam] = model_no_spec(x, lam_input, mode='inference').reshape(-1).cpu()
        u_true[lam] = (lam * x.cpu().reshape(-1) ** 2)
        
        axs[2*lam_val.index(lam)].plot(x.cpu(), u_pred_spec[lam], linestyle='--', marker='o',linewidth=2, color=colors[0], label='Predicted (Two-Phase strategy)')
        axs[2*lam_val.index(lam)].plot(x.cpu(), u_true[lam], linewidth=2, color=colors[0], label='Analytical solution')

        axs[2*lam_val.index(lam)+1].plot(x.cpu(), u_pred_no_spec[lam], linestyle='--', marker='o', linewidth=2, color=colors[1], label='Predicted (One-Phase strategy)')
        axs[2*lam_val.index(lam)+1].plot(x.cpu(), u_true[lam], linewidth=2, color=colors[1], label='Analytical solution')

        axs[2*lam_val.index(lam)].set_xlabel('x', fontsize=15)
        axs[2*lam_val.index(lam)].set_ylabel('U(x; $\\lambda$)', fontsize=15)
        axs[2*lam_val.index(lam)].tick_params(axis='both', labelsize=12)
        axs[2*lam_val.index(lam)].legend(fontsize=12)
        axs[2*lam_val.index(lam)].set_title(f"μ = {lam}", fontsize=15)
        
        axs[2*lam_val.index(lam)+1].set_xlabel('x', fontsize=15)
        axs[2*lam_val.index(lam)+1].set_ylabel('U(x; $\\lambda$)', fontsize=15)
        axs[2*lam_val.index(lam)+1].tick_params(axis='both', labelsize=12)
        axs[2*lam_val.index(lam)+1].legend(fontsize=12)
        axs[2*lam_val.index(lam)+1].set_title(f"μ = {lam}", fontsize=15)

    plt.show()

@torch.no_grad()
def visualize_multiple_lambdas(model, lambda_values, nx=100, filename=None):
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)

    plt.figure(figsize=(8, 5))
    
    for lam_val in lambda_values:
        lam_input = torch.full_like(x, lam_val)
        u_pred = model(x, lam_input, mode='inference').reshape(-1).cpu()
        # True solution for comparison
        u_true = (lam_val * x.cpu().reshape(-1) ** 2)

        plt.plot(x.cpu(), u_pred, label=f'Predicted μ={lam_val:.2f}')
        plt.plot(x.cpu(), u_true, '--', label=f'True μ={lam_val:.2f}')

    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Model Prediction for Multiple μ Values')
    plt.legend()
    plt.grid(True)

    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")

    plt.show()

@torch.no_grad()
def plot_heads_of_saved_models_grid(model_class, model_paths, task_lambdas=TASK_LAMBDAS, nx=100, device='cpu'):
    """
    For each saved model, plot the output of each head and the true solution for each training lambda.
    Each model gets its own subplot.
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    n_models = len(model_paths)
    n_cols = 5
    n_rows = (n_models + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 5, n_rows * 4))
    for idx, model_path in enumerate(model_paths):
        model = model_class(task_lambdas).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        plt.subplot(n_rows, n_cols, idx + 1)
        for i, lam_val in enumerate(task_lambdas):
            u_pred = model.forward_head(x, i).detach().cpu().numpy().flatten()
            u_true = (lam_val * x.cpu().numpy().flatten() ** 2)
            plt.plot(x.cpu(), u_pred, label=f'Head {i+1} μ={lam_val}')
            plt.plot(x.cpu(), u_true, '--', label=f'True μ={lam_val}')
        plt.title(f"{model_path.split('/')[-1]}")
        plt.xlabel('x')
        plt.ylabel('u(x)')
        plt.grid(True)
        if idx == 0:
            plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

@torch.no_grad()
def plot_heads_vs_true_x2(model, task_lambdas=TASK_LAMBDAS, nx=100, file_save=None):
    """
    Plot the output of each head and the true solution for each training lambda.
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    plt.figure(figsize=(8, 5))
    colors = ["#1B9E77FF", "#D95F02FF", "#7570B3FF", "#E7298AFF"]
    for i, lam_val in enumerate(task_lambdas):
        u_pred = model.forward_head(x, i).detach().cpu().numpy().flatten()
        u_true = (lam_val * x.cpu().numpy().flatten() ** 2)
        plt.plot(x.cpu(), u_pred, linestyle='--', marker='o', label=f'Solution learned by head {i+1}, μ={lam_val}', linewidth=2, color=colors[i])
        plt.plot(x.cpu(), u_true, label=f'Analytical solution, μ={lam_val}',  linewidth=2, color=colors[i])
    ax = plt.gca()
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('U(x; $\\lambda$)', fontsize=15)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_ylim(-0.05, 4)
    ax.legend(fontsize=12)
    # inset Axes....
    x1, x2, y1, y2 = 0.25, 0.35, 0.001, 0.4  # subregion of the original image
    axins = ax.inset_axes(
        [0.6, 0.5, 0.37, 0.47],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
    for i, lam_val in enumerate(task_lambdas):
        u_pred = model.forward_head(x, i).detach().cpu().numpy().flatten()
        u_true = (lam_val * x.cpu().numpy().flatten() ** 2)
        axins.plot(x.cpu(), u_pred, linestyle='--', marker='o', label=f'Predicted μ={lam_val:.2f}', linewidth=2, color=colors[i])
        axins.plot(x.cpu(), u_true, label=f'True μ={lam_val:.2f}',  linewidth=2, color=colors[i])
    
    ax.indicate_inset_zoom(axins, edgecolor="black", linestyle='--')
    #plt.title('Head outputs vs True solutions for each training μ')
    plt.legend(loc='upper left', fontsize=10)
    #plt.grid(True)
    #plt.savefig(file_save, format="pdf", bbox_inches="tight")
    plt.show()

@torch.no_grad()
def plot_true_solutions_x2(task_lambdas=TASK_LAMBDAS, nx=100, file_save=None):
    """
    Plot only the true solutions (μx²) for each training lambda.
    """
    x = torch.linspace(-1, 1, nx).reshape(-1, 1).to(device)
    plt.figure(figsize=(8, 5))
    for i, lam_val in enumerate(task_lambdas):
        u_true = (lam_val * x.cpu().numpy().flatten() ** 2)
        plt.plot(x.cpu(), u_true, label=f'μ={lam_val}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('Analytical solutions for different μ values')
    plt.legend()
    plt.grid(True)
    if file_save:
        plt.savefig(file_save, format="pdf", bbox_inches="tight")
    plt.show()


def train(
    strategy: str,
    output_dir: Path,
    task_lambdas: list[float] | None = None,
    head_epochs: int = 30_000,
    attn_epochs: int = 5_000,
    seed: int = 50,
) -> Path:
    """Train the λx² elliptic benchmark (one-phase or two-phase strategy)."""
    global device
    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = task_lambdas or TASK_LAMBDAS
    two_phase = strategy == "two-phase"

    if not two_phase:
        model = BatchedMultiHeadPINN_Attention(tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_lambda_x2_pinn_multitask_attention_head_spec(
            model,
            optimizer,
            epochs=head_epochs,
            attn_reg=0.1,
            task_lambdas=tasks,
            loss_filename=str(output_dir / "loss_one_phase.txt"),
        )
        model_path = output_dir / "model_one_phase.pth"
    else:
        model = BatchedMultiHeadPINN_Attention(tasks).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_lambda_x2_pinn_multitask_head_spec_isolated_attention(
            model,
            optimizer,
            task_lambdas=tasks,
            epochs=head_epochs,
            device=device,
            loss_filename=str(output_dir / "loss_heads_two_phase.txt"),
        )
        for param in model.shared.parameters():
            param.requires_grad = False
        for head in model.heads:
            for param in head.parameters():
                param.requires_grad = False
        for param in model.attn_net.parameters():
            param.requires_grad = True
        optimizer_attn = torch.optim.Adam(model.attn_net.parameters(), lr=1e-3)
        train_lambda_x2_pinn(
            model,
            optimizer_attn,
            epochs=attn_epochs,
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
    nx: int = 30,
    show: bool = True,
    save_path: Path | None = None,
) -> None:
    """Compare one-phase and two-phase models on λ values."""
    model_two, model_one = load_models(output_dir)
    visualize(model_two, model_one, lam_val=param_values or [2.0, 8.0], nx=nx)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close("all")
