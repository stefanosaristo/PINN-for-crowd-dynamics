"""
pinn_plot.py
------------
Loads a trained CrowdPINN model and the preprocessed dataset from disk
(written by pinn_train.py) and generates all visualisations.

Run this script whenever you want to change / re-generate plots
WITHOUT re-training the model.

Expected files in ./pinn_results/:
    model.pt   – model weights + architecture info
    data.npz   – preprocessed tensors (t, x, y, rho, phix, phiy)
"""

from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.nn as nn
import numpy as np
import os

# ==========================================
# INPUT CONFIGURATION — edit only these paths
# ==========================================
SAVE_DIR   = "pinn_results"
MODEL_PATH = os.path.join(SAVE_DIR, "model.pt")
DATA_PATH  = os.path.join(SAVE_DIR, "data.npz")


# ==========================================
# 1. Model Definition (must match pinn_train.py)
# ==========================================
class CrowdPINN(nn.Module):
    def __init__(self, layers=[3, 128, 128, 128, 3]):
        super(CrowdPINN, self).__init__()

        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i + 1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)

        self.mu_x  = nn.Parameter(torch.randn(3) * 0.1)
        self.mu_y  = nn.Parameter(torch.randn(3) * 0.1)
        self.layers = layers

    def forward(self, t, x, y):
        inputs  = torch.cat([t, x, y], dim=1)
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
# 2. Load Model and Data from Disk
# ==========================================
def load_model(model_path=MODEL_PATH, device=None):
    """
    Reconstructs the CrowdPINN from the saved checkpoint.
    The architecture (layer sizes) is read from the checkpoint itself,
    so you never have to manually keep them in sync between files.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    layers     = checkpoint["layers"]

    model = CrowdPINN(layers=layers).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Model loaded from '{model_path}'")
    print(f"  Architecture : {layers}")
    print(f"  mu_x (Phi_x) : {checkpoint['mu_x']}")
    print(f"  mu_y (Phi_y) : {checkpoint['mu_y']}")

    return model, device


def load_data(data_path=DATA_PATH, device=None):
    """
    Loads the preprocessed numpy arrays saved by pinn_train.py and
    converts them back to PyTorch tensors on the correct device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    arrays = np.load(data_path)

    def to_tensor(key):
        return torch.tensor(arrays[key], dtype=torch.float32).to(device)

    t    = to_tensor("t")
    x    = to_tensor("x")
    y    = to_tensor("y")
    rho  = to_tensor("rho")
    phix = to_tensor("phix")
    phiy = to_tensor("phiy")

    print(f"Data loaded from '{data_path}'  —  {t.shape[0]} points")
    return t, x, y, rho, phix, phiy


# ==========================================
# 3. Snapshot Evolution Plots
# ==========================================
def plot_evolution(model, t, x, y, rho_true, phix_true, phiy_true,
                   num_frames=10, out_dir="frames"):
    """
    Saves a series of PNG snapshots showing True Density, PINN Prediction,
    and the absolute error map at evenly-spaced time slices.

    Parameters
    ----------
    num_frames : int
        Number of time slices to visualise.
    out_dir : str
        Directory where the PNG files are written.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    time_steps = np.linspace(0.0, 1.0, num_frames)

    for i, target_t in enumerate(time_steps):
        # Grab all data points within a small time window around target_t
        mask = ((t >= target_t - 0.02) & (t <= target_t + 0.02)).flatten()

        if mask.sum() == 0:
            print(f"Frame {i}: no data near t={target_t:.2f}, skipping.")
            continue

        t_s, x_s, y_s     = t[mask], x[mask], y[mask]
        rho_true_s         = rho_true[mask]

        with torch.no_grad():
            rho_pred_s, _, _ = model(t_s, x_s, y_s)

        # Convert to numpy
        x_np      = x_s.cpu().numpy().flatten()
        y_np      = y_s.cpu().numpy().flatten()
        rho_t_np  = rho_true_s.cpu().numpy().flatten()
        rho_p_np  = rho_pred_s.cpu().numpy().flatten()
        error_np  = np.abs(rho_t_np - rho_p_np)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)
        fig.suptitle(f"t = {target_t:.2f}", fontsize=13)

        axes[0].scatter(x_np, y_np, c=rho_t_np,  cmap='viridis', s=8, marker='s', vmin=0, vmax=1)
        axes[0].set_title("True Density")
        plt.colorbar(axes[0].collections[0], ax=axes[0], label="Density")

        axes[1].scatter(x_np, y_np, c=rho_p_np,  cmap='viridis', s=8, marker='s', vmin=0, vmax=1)
        axes[1].set_title("PINN Prediction")
        plt.colorbar(axes[1].collections[0], ax=axes[1], label="Density")

        axes[2].scatter(x_np, y_np, c=error_np,  cmap='magma',   s=8, marker='s')
        axes[2].set_title("Absolute Error")
        plt.colorbar(axes[2].collections[0], ax=axes[2], label="|Error|")

        for ax in axes:
            ax.set_xlabel("x (normalised)")
            ax.set_ylabel("y (normalised)")

        plt.tight_layout()
        fname = os.path.join(out_dir, f"snapshot_{i:03d}.png")
        plt.savefig(fname)
        plt.close()
        print(f"  Saved {fname}")

    print(f"All frames saved to '{out_dir}/'")


# ==========================================
# 4. Animated Evolution (Video / GIF)
# ==========================================
def animate_evolution(model, t, x, y, rho_true,
                      filename="crowd_evolution_high_contrast.mp4",
                      num_frames=50, fps=4,
                      time_window=0.02):
    """
    Renders a side-by-side animation of True Density, PINN Prediction,
    and Relative Error (%) and saves it as MP4 (or GIF as fallback).

    Parameters
    ----------
    num_frames   : int   — number of animation frames
    fps          : int   — frames per second in the saved video
    time_window  : float — half-width of the time slice for each frame
    """
    model.eval()
    unique_times = np.linspace(0.0, 1.0, num_frames)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(22, 6), dpi=150)
    plt.subplots_adjust(wspace=0.4, bottom=0.15)

    # PowerNorm: gamma < 1 stretches low-density values for better contrast
    norm_density = colors.PowerNorm(gamma=0.5, vmin=0, vmax=1)
    norm_error   = colors.Normalize(vmin=0, vmax=100)

    # Dummy scatter objects so colorbars can be created once
    sc1 = ax1.scatter([], [], c=[], cmap='magma', norm=norm_density)
    sc2 = ax2.scatter([], [], c=[], cmap='magma', norm=norm_density)
    sc3 = ax3.scatter([], [], c=[], cmap='Reds',  norm=norm_error)

    fig.colorbar(sc1, ax=ax1, label='Density',             fraction=0.046, pad=0.04)
    fig.colorbar(sc2, ax=ax2, label='Density',             fraction=0.046, pad=0.04)
    fig.colorbar(sc3, ax=ax3, label='Relative Error (%)',  fraction=0.046, pad=0.04)

    def update(frame):
        target_t = unique_times[frame]
        for ax in [ax1, ax2, ax3]:
            ax.clear()

        mask = ((t >= target_t - time_window) & (t <= target_t + time_window)).flatten()
        if mask.sum() == 0:
            return

        t_s, x_s, y_s = t[mask], x[mask], y[mask]
        rho_true_s     = rho_true[mask]

        with torch.no_grad():
            rho_pred_s, _, _ = model(t_s, x_s, y_s)

        x_np     = x_s.cpu().numpy().flatten()
        y_np     = y_s.cpu().numpy().flatten()
        rho_t_np = rho_true_s.cpu().numpy().flatten()
        rho_p_np = rho_pred_s.cpu().numpy().flatten()

        # Relative error capped at 100% (0.01 added to avoid div-by-zero in empty areas)
        error_pct = np.clip(
            (np.abs(rho_t_np - rho_p_np) / (rho_t_np + 0.01)) * 100,
            0, 100
        )

        ax1.scatter(x_np, y_np, c=rho_t_np,  cmap='magma', s=12, marker='s', norm=norm_density)
        ax1.set_title(f"True Density  (t={target_t:.2f})")
        ax1.set_xlabel("x"); ax1.set_ylabel("y")

        ax2.scatter(x_np, y_np, c=rho_p_np,  cmap='magma', s=12, marker='s', norm=norm_density)
        ax2.set_title("PINN Predicted Density")
        ax2.set_xlabel("x")

        ax3.scatter(x_np, y_np, c=error_pct,  cmap='Reds',  s=12, marker='s', norm=norm_error)
        ax3.set_title("Relative Error (%)")
        ax3.set_xlabel("x")

        print(f"  Rendering frame {frame + 1}/{num_frames}...", end="\r")

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=int(1000 / fps))

    try:
        ani.save(filename, writer='ffmpeg', fps=fps)
        print(f"\nVideo saved → {filename}")
    except Exception as e:
        print(f"\nffmpeg not available ({e}). Falling back to GIF...")
        gif_path = filename.replace(".mp4", ".gif")
        ani.save(gif_path, writer='pillow', fps=fps)
        print(f"GIF saved   → {gif_path}")

    plt.close()


# ==========================================
# 5. Fundamental Diagram (Flux vs Density)
# ==========================================
def plot_fundamental_diagram(model, t, x, y, rho_true, phix_true, phiy_true,
                             out_path="fundamental_diagram.png"):
    """
    Plots the learned polynomial flux law on top of the raw scatter data.

    Three series are shown:
      - True Data   (blue scatter)
      - PINN Prediction (red scatter)
      - Learned Polynomial Law (black line)

    Parameters
    ----------
    out_path : str — where to save the PNG
    """
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        rho_pred, phix_pred, phiy_pred = model(t, x, y)
        rho_continuous = torch.linspace(0, 1.0, 500).view(-1, 1).to(device)
        phix_poly, phiy_poly = model.get_poly_flux(rho_continuous, rho_max=1.0)

    rho_true_np   = rho_true.cpu().numpy().flatten()
    flux_mag_true = np.sqrt(
        phix_true.cpu().numpy().flatten() ** 2 +
        phiy_true.cpu().numpy().flatten() ** 2
    )

    rho_pred_np   = rho_pred.cpu().numpy().flatten()
    flux_mag_pred = np.sqrt(
        phix_pred.cpu().numpy().flatten() ** 2 +
        phiy_pred.cpu().numpy().flatten() ** 2
    )

    rho_line_np   = rho_continuous.cpu().numpy().flatten()
    flux_mag_line = np.sqrt(
        phix_poly.cpu().numpy().flatten() ** 2 +
        phiy_poly.cpu().numpy().flatten() ** 2
    )

    plt.figure(figsize=(8, 6), dpi=150)
    plt.scatter(rho_true_np,  flux_mag_true, alpha=0.2, label='True Data',            s=1, c='blue')
    plt.scatter(rho_pred_np,  flux_mag_pred, alpha=0.2, label='PINN Prediction',       s=1, c='red')
    plt.plot(   rho_line_np,  flux_mag_line, color='black', linewidth=2,
                label='Learned Polynomial Law')

    plt.xlabel("Density (normalised)")
    plt.ylabel("Flux Magnitude")
    plt.title("Fundamental Diagram: Flux vs. Density")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.show()
    print(f"Fundamental diagram saved → {out_path}")


# ==========================================
# Entry Point — edit the flags below to
# control which plots are generated
# ==========================================
if __name__ == "__main__":

    # ---- Load ----
    model, device = load_model(MODEL_PATH)
    t, x, y, rho, phix, phiy = load_data(DATA_PATH, device)

    # ---- Toggle plots ----
    RUN_SNAPSHOTS         = True   # saves frames/snapshot_*.png
    RUN_ANIMATION         = True   # saves crowd_evolution_high_contrast.mp4
    RUN_FUNDAMENTAL       = True   # saves fundamental_diagram.png

    # ---- Snapshot parameters ----
    SNAPSHOT_NUM_FRAMES   = 10
    SNAPSHOT_OUT_DIR      = "frames"

    # ---- Animation parameters ----
    ANIM_NUM_FRAMES       = 50
    ANIM_FPS              = 4
    ANIM_TIME_WINDOW      = 0.02   # half-width of the time slice per frame
    ANIM_FILENAME         = "crowd_evolution_high_contrast.mp4"

    # ---- Fundamental diagram parameters ----
    FUNDAMENTAL_OUT_PATH  = "fundamental_diagram.png"

    # ---- Run ----
    if RUN_SNAPSHOTS:
        print("\n[1/3] Generating snapshots...")
        plot_evolution(
            model, t, x, y, rho, phix, phiy,
            num_frames=SNAPSHOT_NUM_FRAMES,
            out_dir=SNAPSHOT_OUT_DIR
        )

    if RUN_ANIMATION:
        print("\n[2/3] Rendering animation...")
        animate_evolution(
            model, t, x, y, rho,
            filename=ANIM_FILENAME,
            num_frames=ANIM_NUM_FRAMES,
            fps=ANIM_FPS,
            time_window=ANIM_TIME_WINDOW
        )

    if RUN_FUNDAMENTAL:
        print("\n[3/3] Plotting fundamental diagram...")
        plot_fundamental_diagram(
            model, t, x, y, rho, phix, phiy,
            out_path=FUNDAMENTAL_OUT_PATH
        )

    print("\nDone.")