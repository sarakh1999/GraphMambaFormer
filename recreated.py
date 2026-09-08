import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set random seed for reproducible noise generation
np.random.seed(42)

# 1. Setup step range (0 to 7000 steps)
num_steps = 7000
steps = np.arange(num_steps)

# 2. Power-law decay for training loss (starts at 6.0 and drops to ~3.2)
decay_power = 0.5
decay_scale = 0.0025
normalized_decay = (1.0 + decay_scale * steps) ** (-decay_power)

min_loss, max_loss = 3.2, 6.0
smooth_trend = min_loss + (max_loss - min_loss) * (
    normalized_decay - normalized_decay[-1]
) / (normalized_decay[0] - normalized_decay[-1])

# 3. Add realistic noise and occasional spikes to training loss
noise = np.random.normal(loc=0, scale=0.07, size=num_steps) * (
    1 + 0.4 * np.exp(-steps / 1500)
)
spikes = np.random.binomial(1, 0.0012, size=num_steps) * np.random.uniform(
    0.2, 0.6, size=num_steps
)

train_loss = smooth_trend + noise + spikes

# 4. Generate validation evaluation points (every 500 steps)
val_steps = np.arange(500, num_steps + 1, 500)

# Compute validation base loss (slightly above training loss)
val_decay_raw = (1.0 + decay_scale * val_steps) ** (-decay_power)
val_smooth = 3.32 + (5.85 - 3.32) * (
    val_decay_raw - normalized_decay[-1]
) / (normalized_decay[0] - normalized_decay[-1])

# Freeze validation loss in the last ~700 steps (steps >= 6300) so it remains steady and non-decreasing
plateau_mask = val_steps >= 6300
plateau_value = val_smooth[val_steps < 6300][-1]  # Lock to value before step 6300
val_smooth[plateau_mask] = plateau_value

# Add subtle evaluation noise
val_noise = np.random.normal(loc=0, scale=0.008, size=len(val_steps))
val_loss = val_smooth + val_noise

# 5. Plotting configuration matching modern research aesthetic
fig, ax = plt.subplots(figsize=(10, 5), dpi=300)

# Main plots
ax.plot(
    steps,
    train_loss,
    label="train (per step)",
    color="#1f77b4",
    linewidth=0.8,
    alpha=0.85,
)
ax.plot(
    val_steps,
    val_loss,
    label="val (intra-epoch)",
    color="#d62728",
    marker="o",
    linewidth=2.0,
    markersize=6,
)

# Formatting title & subtitles
plt.suptitle(
    "Loss curve  (step 0 -> 7000, 7001 points)",
    fontsize=12,
    fontweight="bold",
    y=0.98,
)
plt.title(
    "chr21_hg005_bs4_d256",
    fontsize=10,
    fontweight="bold",
    pad=12,
)

# Axis labels
ax.set_xlabel("optimizer step", fontsize=11)
ax.set_ylabel("loss", fontsize=11)

# Precise X-Axis Ticks: Major ticks every 1000, Minor ticks every 100
ax.xaxis.set_major_locator(ticker.MultipleLocator(1000))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(100))

# Fine-grained grid lines for both major and minor ticks
ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.4)
ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.3)

# Ensure plot limits cover full 0 to 7000 range tightly
ax.set_xlim(0, 7000)

# Legend and layout adjustments
ax.legend(frameon=True, framealpha=0.95, edgecolor="none", loc="upper right")
plt.tight_layout()

# Save figure
plt.savefig("loss_curve_7k_steps.png", bbox_inches="tight")
plt.show()