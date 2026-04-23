# CrowdPINN — Code Documentation

> A Physics-Informed Neural Network for crowd dynamics. Learning density and flux fields from simulated ABM data while enforcing conservation of mass as a physical constraint.

---

## Table of Contents

1. [Overview & Workflow](#1-overview--workflow)
2. [File Structure](#2-file-structure)
3. [pinn_train.py — Deep Dive](#3-pinn_trainpy--deep-dive)
   - 3.1 [CrowdPINN Model](#31-crowdpinn-model)
   - 3.2 [Polynomial Flux Law](#32-polynomial-flux-law)
   - 3.3 [Loss Function](#33-loss-function)
   - 3.4 [Data Loading & Normalisation](#34-data-loading--normalisation)
   - 3.5 [Training Loop — Adam Phase](#35-training-loop--adam-phase)
   - 3.6 [Training Loop — L-BFGS Phase](#36-training-loop--l-bfgs-phase)
   - 3.7 [Saving Results to Disk](#37-saving-results-to-disk)
4. [pinn_plot.py — Deep Dive](#4-pinn_plotpy--deep-dive)
   - 4.1 [Loading the Model](#41-loading-the-model)
   - 4.2 [Loading the Data](#42-loading-the-data)
   - 4.3 [Snapshot Evolution Plots](#43-snapshot-evolution-plots)
   - 4.4 [Animated Evolution](#44-animated-evolution)
   - 4.5 [Fundamental Diagram](#45-fundamental-diagram)
   - 4.6 [Plot Toggle Flags](#46-plot-toggle-flags)
5. [How the Two Files Communicate](#5-how-the-two-files-communicate)
6. [Customisation Guide](#6-customisation-guide)
7. [Dependencies](#7-dependencies)

---

## 1. Overview & Workflow

This project implements a **Physics-Informed Neural Network (PINN)** for 2D crowd dynamics. The system learns three output fields simultaneously:

| Symbol   | Meaning                          |
|----------|----------------------------------|
| `ρ`      | Crowd density at point (t, x, y) |
| `φ_x`    | x-component of the pedestrian flux |
| `φ_y`    | y-component of the pedestrian flux |

The network is trained on agent-based simulation data and simultaneously penalised for violating the **conservation of mass** PDE:

```
∂ρ/∂t + ∂φ_x/∂x + ∂φ_y/∂y = 0
```

Additionally, the model discovers a **polynomial fundamental diagram** — the macroscopic law that relates density to flux — as a set of 6 learnable scalar parameters (`mu_x[0..2]`, `mu_y[0..2]`).

### Two-step workflow

```
[ pinn_train.py ]               [ pinn_plot.py ]
      │                                │
      ▼                                ▼
Load CSV data             Load pinn_results/model.pt
Normalise inputs          Load pinn_results/data.npz
Train (Adam + L-BFGS)     Reconstruct model in eval mode
Save model + data  ──────►  Generate snapshots / animation / diagram
```

You only need to re-run `pinn_train.py` when you change the model or the data. For all visual changes, `pinn_plot.py` is sufficient.

---

## 2. File Structure

```
project/
├── pinn_train.py          ← Training script (slow, run once)
├── pinn_plot.py           ← Plotting script (fast, run anytime)
│
├── pinn_results/          ← Created automatically by pinn_train.py
│   ├── model.pt           ← Saved weights + architecture info
│   └── data.npz           ← Saved preprocessed tensors
│
├── frames/                ← Created by plot_evolution()
│   ├── snapshot_000.png
│   ├── snapshot_001.png
│   └── ...
│
├── crowd_evolution_high_contrast.mp4   ← Created by animate_evolution()
└── fundamental_diagram.png             ← Created by plot_fundamental_diagram()
```

---

## 3. `pinn_train.py`

### 3.1 CrowdPINN Model

```python
class CrowdPINN(nn.Module):
    def __init__(self, layers=[3, 128, 128, 128, 3]):
```

The neural network has:

- **Input layer**: 3 neurons — the coordinates `(t, x, y)`
- **Hidden layers**: three fully-connected layers of 128 neurons each, activated with `tanh`
- **Output layer**: 3 neurons — `(ρ, φ_x, φ_y)`

`tanh` is chosen deliberately over `ReLU` because PINNs need to differentiate the network output with respect to the inputs. `tanh` is infinitely differentiable and produces smooth predictions, which makes automatic differentiation via `torch.autograd.grad` stable and accurate. `ReLU` has a zero second derivative almost everywhere, which would kill the PDE loss.

The `layers` list is stored as an instance attribute (`self.layers`) so that the full architecture can be reconstructed from a checkpoint without hard-coding the shape elsewhere.

**Physics parameters:**

```python
self.mu_x = nn.Parameter(torch.randn(3) * 0.1)
self.mu_y = nn.Parameter(torch.randn(3) * 0.1)
```

These are **trainable scalars** (not network weights). They represent the coefficients of the polynomial fundamental diagram (see §3.2). They are initialised with small random values and updated by the same optimiser as the network weights.

---

### 3.2 Polynomial Flux Law

```python
def get_poly_flux(self, rho, rho_max=1.0):
```

This method evaluates the **macroscopic constitutive law** — the relationship between density and flux encoded as a 4th-degree polynomial. The polynomial has the form:

```
φ_x(ρ) = μ₁ρ + μ₂ρ² + μ₃ρ³ + μ₄ρ⁴
φ_y(ρ) = μ₅ρ + μ₆ρ² + μ₇ρ³ + μ₈ρ⁴
```

The 4th coefficient (`mu4`) is **not a free parameter** — it is computed analytically from the others to enforce the **boundary condition `φ(ρ_max) = 0`**, meaning no net flux at maximum density (fully jammed crowd):

```python
mu4_x = -(mu_x[0]*rho_max + mu_x[1]*rho_max² + mu_x[2]*rho_max³) / rho_max⁴
```

This constraint reduces the free parameter count from 8 to 6 and injects a physically meaningful prior into the learned law.

---

### 3.3 Loss Function

```python
def compute_loss(model, t, x, y, rho_true, phix_true, phiy_true, ...):
```

The total loss is a weighted sum of three terms:

#### Term 1 — Data Loss (`loss_data`)

```
L_data = mean[ (ρ_pred - ρ_true)² + (φ_x_pred - φ_x_true)² + (φ_y_pred - φ_y_true)² ]
```

A standard mean-squared-error between the network output and the ground-truth simulation values. This drives the network to fit the data.

#### Term 2 — Polynomial Consistency Loss (`loss_poly`)

```
L_poly = mean[ (φ_x_pred - φ_x_poly)² + (φ_y_pred - φ_y_poly)² ]
```

This forces the network's flux outputs to be consistent with the polynomial law. Without this term, the NN could fit the data without learning a meaningful parametric law. This loss bridges the black-box NN with the interpretable polynomial model.

#### Term 3 — PDE Loss (`loss_pde`)

```python
drho_dt  = grad(rho_pred,  t, ...)
dphix_dx = grad(phix_pred, x, ...)
dphiy_dy = grad(phiy_pred, y, ...)

pde_residual = drho_dt + dphix_dx + dphiy_dy
loss_pde = mean(pde_residual²)
```

This is the **physics constraint**. The gradients are computed using PyTorch's automatic differentiation (not finite differences). `create_graph=True` is necessary so that these gradients can themselves be differentiated when `loss.backward()` is called during training — this is what makes PINNs computationally expensive compared to standard supervised learning.

The total loss is:

```
L_total = w_data · L_data + w_pde · L_pde + w_poly · L_poly
```

The default weights are all `1.0`. You can tune them to prioritise data fidelity vs. physical consistency.

---

### 3.4 Data Loading & Normalisation

```python
def load_and_prep_data(filepath, device):
```

Reads the CSV file (columns: `t, x, y, ρ, φ_x, φ_y`) and applies two normalisation steps:

**Spatial/temporal normalisation — min-max scaling to [0, 1]:**
```python
t = (t - t.min()) / (t.max() - t.min())
```
Applied to `t`, `x`, and `y`. This is critical for PINN training because automatic differentiation involves multiplying gradients through the network depth. If the input coordinates span large ranges (e.g. seconds or metres), gradient signals become numerically unstable. Mapping to [0, 1] keeps gradients well-conditioned.

**Density normalisation — also min-max to [0, 1]:**
```python
rho = (rho - rho.min()) / (rho.max() - rho.min())
```

Note: the flux components (`phix`, `phiy`) are **not** normalised. This is intentional — they are used directly in the data loss and their scale is already compatible with the normalised density.

---

### 3.5 Training Loop — Adam Phase

```python
for epoch in range(epochs_adam):  # default: 3500
```

**Balanced mini-batching:**

The dataset is split into `occupied_indices` (ρ > 0.001) and `empty_indices` (ρ ≤ 0.001). Each batch is forced to be 50% occupied and 50% empty:

```python
n_occ = min(batch_size // 2, len(occupied_indices))
n_emp = batch_size - n_occ
idx   = torch.cat([idx_occ, idx_emp])
```

Without this balancing, the model would be dominated by empty-space samples (which are typically much more numerous), causing it to over-fit the trivial ρ=0 region and under-fit the crowd dynamics. Balanced sampling ensures the network sees enough occupied-space gradients at every step.

Batch size is 10,000. On each step: compute loss → backpropagate → Adam update.

---

### 3.6 Training Loop — L-BFGS Phase

```python
optimizer_lbfgs = torch.optim.LBFGS(model.parameters(), max_iter=epochs_lbfgs, ...)
```

After Adam has brought the model to a good basin, L-BFGS is used for **fine-tuning**. L-BFGS is a quasi-Newton second-order method that converges much faster than first-order methods near a minimum, at the cost of needing the full gradient of a **fixed** batch.

This is why a single static balanced batch is created before the L-BFGS phase and reused across all its internal iterations (unlike Adam which uses a fresh random batch each step):

```python
idx_lbfgs = torch.cat([idx_occ_lbfgs, idx_emp_lbfgs])

def closure():
    optimizer_lbfgs.zero_grad()
    loss, _, _, _ = compute_loss(model, t[idx_lbfgs], ...)
    loss.backward()
    return loss

optimizer_lbfgs.step(closure)
```

The `closure` pattern is mandatory for L-BFGS in PyTorch: the optimiser may call `closure()` multiple times per `.step()` call (to perform line search), so the loss must be re-computed and re-differentiated each time it is called.

---

### 3.7 Saving Results to Disk

```python
def save_results(model, t, x, y, rho, phix, phiy, save_dir=SAVE_DIR):
```

Two files are written to `pinn_results/`:

**`model.pt`** — a PyTorch checkpoint dictionary:
```python
{
    "model_state_dict": model.state_dict(),   # all weights + biases + mu_x, mu_y
    "layers":           model.layers,          # architecture (e.g. [3,128,128,128,3])
    "mu_x":             ...,                   # learned flux params (numpy, for quick inspection)
    "mu_y":             ...,
}
```

Saving `layers` inside the checkpoint is the key design decision: it makes `pinn_plot.py` fully self-contained and independent of any hard-coded architecture constant.

**`data.npz`** — a numpy archive:
```python
np.savez(DATA_PATH, t=..., x=..., y=..., rho=..., phix=..., phiy=...)
```

All tensors are detached from the computation graph and moved to CPU before saving (`.detach().cpu().numpy()`). This ensures the file is portable regardless of whether training was done on GPU or CPU.

---

## 4. `pinn_plot.py`

### 4.1 Loading the Model

```python
def load_model(model_path=MODEL_PATH, device=None):
```

1. Calls `torch.load(..., map_location=device)` — the `map_location` argument ensures that a model trained on GPU can be loaded on a CPU-only machine without errors.
2. Reads `layers` from the checkpoint to reconstruct the exact same architecture.
3. Calls `model.load_state_dict(...)` to restore all weights.
4. Calls `model.eval()` — this disables dropout and batch-norm training behaviour (not used here, but good practice). More importantly, it signals that we will not call `.backward()`, which allows PyTorch to skip storing intermediate values for the backward pass and saves memory.

---

### 4.2 Loading the Data

```python
def load_data(data_path=DATA_PATH, device=None):
```

Loads the `.npz` archive and converts each array back to a PyTorch `float32` tensor on the target device. The data is already normalised (min-max scaled) from the training phase — no preprocessing is needed here.

---

### 4.3 Snapshot Evolution Plots

```python
def plot_evolution(model, t, x, y, rho_true, phix_true, phiy_true,
                   num_frames=10, out_dir="frames"):
```

Generates `num_frames` static PNG images, one per time slice. For each frame:

1. **Time slicing**: selects all data points where `|t - target_t| < 0.02`. The tolerance of 0.02 (in normalised time units) is chosen to capture enough points for a visually dense scatter plot. If your data is sparse in time, increasing this value will include more points per frame at the cost of temporal resolution.

2. **Inference**: calls `model(t_s, x_s, y_s)` inside `torch.no_grad()` to suppress autograd tracking — no gradients are needed during visualisation, so this saves both memory and computation time.

3. **Three-panel plot**:
   - **Left**: True density (ground truth from simulation)
   - **Centre**: PINN predicted density
   - **Right**: Absolute error `|ρ_true - ρ_pred|` on a `magma` colormap

Both density panels share the same `vmin=0, vmax=1` range for consistent colour comparison across frames.

Each figure is saved and immediately closed (`plt.close()`) to prevent memory accumulation across many frames.

---

### 4.4 Animated Evolution

```python
def animate_evolution(model, t, x, y, rho_true,
                      filename="crowd_evolution_high_contrast.mp4",
                      num_frames=50, fps=4, time_window=0.02):
```

Builds an MP4 (or GIF fallback) using `matplotlib.animation.FuncAnimation`.

**High-contrast normalisation:**

```python
norm_density = colors.PowerNorm(gamma=0.5, vmin=0, vmax=1)
```

A `PowerNorm` with `gamma=0.5` applies a square-root colour mapping, which stretches low-density values visually. Without this, small but real crowd clusters at low density would appear nearly invisible against the background.

**Colorbars created once:**

The dummy scatter objects `sc1, sc2, sc3` are created before the animation loop so that `fig.colorbar(...)` can be called once. If colorbars were created inside `update()`, they would stack up and the figure would become cluttered and progressively slower.

**`update(frame)` function:**

This is the per-frame callback called by `FuncAnimation`. It:
1. Clears all three axes (`ax.clear()`)
2. Selects the time slice
3. Runs a no-grad forward pass
4. Computes relative error: `|ρ_true - ρ_pred| / (ρ_true + 0.01) × 100`, capped at 100%
5. Redraws all three scatter plots

**Video saving:**

The script first tries `ffmpeg` (higher quality, smaller file). If not installed, it falls back to `pillow` and saves a GIF.

---

### 4.5 Fundamental Diagram

```python
def plot_fundamental_diagram(model, t, x, y, rho_true, phix_true, phiy_true,
                             out_path="fundamental_diagram.png"):
```

Visualises the learned macroscopic law — the relationship between density and flux magnitude.

Three series are overlaid:

| Series | Colour | What it shows |
|--------|--------|---------------|
| True Data | Blue scatter | Raw (ρ, ‖φ‖) pairs from the simulation |
| PINN Prediction | Red scatter | (ρ_pred, ‖φ_pred‖) output by the network |
| Polynomial Law | Black line | The analytical polynomial `φ(ρ)` evaluated on a fine `ρ ∈ [0, 1]` grid |

The polynomial line is computed on 500 evenly-spaced density values using `get_poly_flux()`, not by running a forward pass of the neural network. This gives the clean, smooth curve of the discovered fundamental diagram independent of any spatial or temporal noise.

A well-trained model will show the black line threading through the cloud of blue points, confirming that the learned polynomial law captures the data's macroscopic structure.

---

### 4.6 Plot Toggle Flags

At the bottom of `pinn_plot.py` there is a dedicated configuration block:

```python
RUN_SNAPSHOTS   = True
RUN_ANIMATION   = True
RUN_FUNDAMENTAL = True
```

Set any flag to `False` to skip that plot entirely. Below the flags, all function parameters are also exposed as named constants (e.g. `SNAPSHOT_NUM_FRAMES`, `ANIM_FPS`, `ANIM_TIME_WINDOW`) so you can tune plots without scrolling into function bodies.

---

## 5. How the Two Files Communicate

The only bridge between the two scripts is the `pinn_results/` directory:

```
pinn_train.py  ──writes──►  pinn_results/model.pt
                            pinn_results/data.npz
                                    │
pinn_plot.py   ◄──reads──────────────┘
```

**`model.pt`** contains a Python dictionary serialised with `torch.save`. It stores:
- `model_state_dict` — all learnable parameters (weights, biases, `mu_x`, `mu_y`) as tensors
- `layers` — the list `[3, 128, 128, 128, 3]` describing the network shape
- `mu_x`, `mu_y` — also stored as numpy arrays for quick human inspection (e.g. `print(checkpoint["mu_x"])`)

**`data.npz`** is a NumPy binary archive (essentially a zip of `.npy` files). It stores the six normalised arrays: `t, x, y, rho, phix, phiy`. Loading this instead of re-reading and re-normalising the original CSV guarantees that `pinn_plot.py` operates on exactly the same coordinate system as the trained model.

> **Important**: never change the normalisation in `pinn_train.py` after saving, without re-saving the data. If the model was trained with coordinates scaled to [0, 1] and you later feed it un-scaled coordinates, the predictions will be meaningless.

---

## 6. Customisation Guide

### Changing the network size

Edit only in `pinn_train.py`:
```python
model = CrowdPINN(layers=[3, 256, 256, 256, 256, 3])
```
The new architecture is automatically stored in `model.pt` and picked up by `pinn_plot.py` — no changes needed there.

### Changing loss weights

```python
loss, ... = compute_loss(model, ..., weights=(1.0, 0.5, 2.0))
#                                              data  pde  poly
```

Increase `w_pde` to force stronger physics compliance at the cost of data fit. Increase `w_poly` if you see the polynomial law diverging from the scatter data.

### Changing snapshot count or time window

Only in `pinn_plot.py`:
```python
SNAPSHOT_NUM_FRAMES = 20    # more frames
ANIM_TIME_WINDOW    = 0.05  # wider time slice per frame
```

### Changing colormaps or normalisations

In `pinn_plot.py`, the colormaps and norms are set directly in `animate_evolution()` and `plot_evolution()`:
```python
norm_density = colors.PowerNorm(gamma=0.3)   # more contrast
cmap='inferno'                                 # different colormap
```

### Running on GPU

No changes needed. Both scripts auto-detect CUDA:
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

## 7. Dependencies

| Package            | Purpose                             |
|--------------------|-------------------------------------|
| `torch`            | Neural network, autograd, optimisers |
| `numpy`            | Array operations, `.npz` I/O        |
| `pandas`           | CSV loading                          |
| `matplotlib`       | All plots and animations             |
| `ffmpeg`  | MP4 video encoding        |

Install Python dependencies with:
```bash
pip install torch numpy pandas matplotlib
```

For MP4 output, install `ffmpeg` via your system package manager:
```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS (Homebrew)
brew install ffmpeg

# Windows (Chocolatey)
choco install ffmpeg
```

If `ffmpeg` is not available, the animation falls back to a GIF automatically.
