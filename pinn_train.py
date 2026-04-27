from matplotlib import colors
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.autograd import grad
import os

# ==========================================
# OUTPUT CONFIGURATION
# Save everything here so pinn_plot.py can load it
# ==========================================
SAVE_DIR = "pinn_results"
MODEL_PATH = os.path.join(SAVE_DIR, "model.pt")
DATA_PATH  = os.path.join(SAVE_DIR, "data.npz")


# ==========================================
# 1. Model Definition
# ==========================================
class CrowdPINN(nn.Module):
    def __init__(self, layers=[3, 128, 128, 128, 3]):
        super(CrowdPINN, self).__init__()

        # Neural Network Architecture
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)

        # Trainable Physics Parameters (mu1 to mu3, and mu5 to mu7)
        self.mu_x = nn.Parameter(torch.randn(3) * 0.1)
        self.mu_y = nn.Parameter(torch.randn(3) * 0.1)

        # Store architecture so we can reconstruct this class when loading
        self.layers = layers

    def forward(self, t, x, y):
        inputs = torch.cat([t, x, y], dim=1)
        outputs = self.net(inputs)

        rho   = outputs[:, 0:1]
        phi_x = outputs[:, 1:2]
        phi_y = outputs[:, 2:3]
        return rho, phi_x, phi_y

    def get_poly_flux(self, rho, rho_max=1.0):
        mu4_x = -(self.mu_x[0] * rho_max + self.mu_x[1] * (rho_max ** 2) + self.mu_x[2] * (rho_max ** 3)) / (rho_max ** 4)
        mu4_y = -(self.mu_y[0] * rho_max + self.mu_y[1] * (rho_max ** 2) + self.mu_y[2] * (rho_max ** 3)) / (rho_max ** 4)

        phi_x_poly = (self.mu_x[0] * rho + self.mu_x[1] * rho ** 2 + self.mu_x[2] * rho ** 3 + mu4_x * rho ** 4)
        phi_y_poly = (self.mu_y[0] * rho + self.mu_y[1] * rho ** 2 + self.mu_y[2] * rho ** 3 + mu4_y * rho ** 4)

        return phi_x_poly, phi_y_poly


# ==========================================
# 2. Loss Function and Physics Constraints
# ==========================================
def compute_loss(model, t, x, y, rho_true, phix_true, phiy_true,
                 rho_max=1.0, weights=(1.0, 1.0, 1.0)):
    w_data, w_pde, w_poly = weights

    # Enable gradients for inputs (needed for PDE residual computation)
    t.requires_grad_(True)
    x.requires_grad_(True)
    y.requires_grad_(True)

    # Forward pass
    rho_pred, phix_pred, phiy_pred = model(t, x, y)
    phix_poly, phiy_poly = model.get_poly_flux(rho_pred, rho_max)

    # 1. Data Loss — MSE against simulation data
    loss_data = torch.mean(
        (rho_pred   - rho_true)  ** 2 +
        (phix_pred  - phix_true) ** 2 +
        (phiy_pred  - phiy_true) ** 2
    )

    # 2. Flux Parameterization Consistency Loss
    #    Forces the NN flux output to stay consistent with the polynomial law
    loss_poly = torch.mean(
        (phix_pred - phix_poly) ** 2 +
        (phiy_pred - phiy_poly) ** 2
    )

    # 3. PDE Loss — Conservation of Mass: d(rho)/dt + d(phi_x)/dx + d(phi_y)/dy = 0
    drho_dt   = grad(rho_pred,   t, grad_outputs=torch.ones_like(rho_pred),   create_graph=True)[0]
    dphix_dx  = grad(phix_pred,  x, grad_outputs=torch.ones_like(phix_pred),  create_graph=True)[0]
    dphiy_dy  = grad(phiy_pred,  y, grad_outputs=torch.ones_like(phiy_pred),  create_graph=True)[0]

    pde_residual = drho_dt + dphix_dx + dphiy_dy
    loss_pde = torch.mean(pde_residual ** 2)

    total_loss = w_data * loss_data + w_pde * loss_pde + w_poly * loss_poly

    return total_loss, loss_data, loss_pde, loss_poly


# ==========================================
# 3. Data Loading & Preprocessing
# ==========================================
def load_and_prep_data(filepath, device):
    df = pd.read_csv(filepath)

    t    = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).view(-1, 1).to(device)
    x    = torch.tensor(df.iloc[:, 1].values, dtype=torch.float32).view(-1, 1).to(device)
    y    = torch.tensor(df.iloc[:, 2].values, dtype=torch.float32).view(-1, 1).to(device)
    rho  = torch.tensor(df.iloc[:, 3].values, dtype=torch.float32).view(-1, 1).to(device)
    phix = torch.tensor(df.iloc[:, 4].values, dtype=torch.float32).view(-1, 1).to(device)
    phiy = torch.tensor(df.iloc[:, 5].values, dtype=torch.float32).view(-1, 1).to(device)

    # Scale t, x, y to [0, 1] to prevent gradient vanishing
    if t.max()   > t.min():   t   = (t   - t.min())   / (t.max()   - t.min())
    if x.max()   > x.min():   x   = (x   - x.min())   / (x.max()   - x.min())
    if y.max()   > y.min():   y   = (y   - y.min())   / (y.max()   - y.min())

    # Normalize density to [0, 1]
    if rho.max() > rho.min():
        rho = (rho - rho.min()) / (rho.max() - rho.min())

    return t, x, y, rho, phix, phiy


# ==========================================
# 4. Training Loop (Adam + L-BFGS)
# ==========================================
def train_pinn(filepath, epochs_adam=3500, epochs_lbfgs=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    t, x, y, rho, phix, phiy = load_and_prep_data(filepath, device)

    model = CrowdPINN().to(device)

    optimizer_adam  = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer_lbfgs = torch.optim.LBFGS(
        model.parameters(),
        max_iter=epochs_lbfgs,
        tolerance_grad=1e-7,
        tolerance_change=1e-9
    )

    batch_size      = 10000
    occupied_indices = torch.where(rho > 0.001)[0]
    empty_indices    = torch.where(rho <= 0.001)[0]

    print(f"Dataset breakdown: {len(occupied_indices)} occupied points, {len(empty_indices)} empty points.")
    print("Starting Adam Optimization...")

    # ---- Adam Phase ----
    for epoch in range(epochs_adam):
        optimizer_adam.zero_grad()

        # Balanced batch: 50% occupied, 50% empty
        n_occ = min(batch_size // 2, len(occupied_indices))
        n_emp = batch_size - n_occ

        idx_occ = occupied_indices[torch.randperm(len(occupied_indices))[:n_occ]]
        idx_emp = empty_indices[torch.randperm(len(empty_indices))[:n_emp]]
        idx = torch.cat([idx_occ, idx_emp])

        loss, l_data, l_pde, l_poly = compute_loss(
            model, t[idx], x[idx], y[idx], rho[idx], phix[idx], phiy[idx]
        )
        loss.backward()
        optimizer_adam.step()

        if epoch % 250 == 0:
            print(f"  Adam Epoch {epoch:>5d}: "
                  f"Total={loss.item():.4e}  "
                  f"Data={l_data.item():.4e}  "
                  f"PDE={l_pde.item():.4e}  "
                  f"Poly={l_poly.item():.4e}")

    # ---- L-BFGS Phase ----
    print("\nStarting L-BFGS Optimization...")

    # L-BFGS needs a fixed (static) batch to converge cleanly
    n_occ_lbfgs  = min(10000 // 2, len(occupied_indices))
    n_emp_lbfgs  = 10000 - n_occ_lbfgs
    idx_occ_lbfgs = occupied_indices[torch.randperm(len(occupied_indices))[:n_occ_lbfgs]]
    idx_emp_lbfgs = empty_indices[torch.randperm(len(empty_indices))[:n_emp_lbfgs]]
    idx_lbfgs = torch.cat([idx_occ_lbfgs, idx_emp_lbfgs])

    def closure():
        optimizer_lbfgs.zero_grad()
        loss, _, _, _ = compute_loss(
            model,
            t[idx_lbfgs], x[idx_lbfgs], y[idx_lbfgs],
            rho[idx_lbfgs], phix[idx_lbfgs], phiy[idx_lbfgs]
        )
        loss.backward()
        return loss

    optimizer_lbfgs.step(closure)

    # ---- Final Metrics ----
    final_loss, _, _, _ = compute_loss(model, t, x, y, rho, phix, phiy)
    print(f"\nOptimization finished. Final Total Loss: {final_loss.item():.4e}")

    print("\nLearned Flux Parameters:")
    print(f"  Phi_x (mu1-3): {model.mu_x.data.cpu().numpy()}")
    print(f"  Phi_y (mu5-7): {model.mu_y.data.cpu().numpy()}")

    return model, t, x, y, rho, phix, phiy


# ==========================================
# 5. Save Results to Disk
# ==========================================
def save_results(model, t, x, y, rho, phix, phiy, save_dir=SAVE_DIR):
    """
    Persists everything pinn_plot.py will need:
      - model weights + architecture info  → model.pt
      - preprocessed tensors (as numpy)    → data.npz
    """
    os.makedirs(save_dir, exist_ok=True)

    # --- Save model ---
    torch.save({
        "model_state_dict": model.state_dict(),
        "layers":           model.layers,        # so plot script can rebuild the architecture
        "mu_x":             model.mu_x.data.cpu().numpy(),
        "mu_y":             model.mu_y.data.cpu().numpy(),
    }, MODEL_PATH)
    print(f"Model saved  → {MODEL_PATH}")

    # --- Save tensors as numpy arrays ---
    np.savez(
        DATA_PATH,
        t    = t.detach().cpu().numpy(),
        x    = x.detach().cpu().numpy(),
        y    = y.detach().cpu().numpy(),
        rho  = rho.detach().cpu().numpy(),
        phix = phix.detach().cpu().numpy(),
        phiy = phiy.detach().cpu().numpy(),
    )
    print(f"Data saved   → {DATA_PATH}")
    print(f"\nAll results saved to '{save_dir}/'. Run pinn_plot.py to generate plots.")


# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    filepath = r"C:\Users\stefa\Documents\Grenoble INP\RISK\code\micro_data\Model4_cylinder_pop_500_density_flux_normalized.csv"

    trained_model, t_data, x_data, y_data, rho_data, phix_data, phiy_data = train_pinn(filepath)
    save_results(trained_model, t_data, x_data, y_data, rho_data, phix_data, phiy_data)
