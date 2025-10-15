import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# =======================
# Load background
# =======================
bg_img = mpimg.imread(r"examples\z_ec_course\A3_template_jack\background.png")

# =======================
# Load path data
# =======================
paths_file = r"MyWork\Nested_Evolution\robot_path.npy"
best_paths_per_gen = np.load(paths_file, allow_pickle=True)

print(f"Loaded {len(best_paths_per_gen)} generations of paths")

# =======================
# Plot
# =======================
fig, ax = plt.subplots(figsize=(8, 10))

# Show background
ax.imshow(bg_img, extent=[-2, 2, -2, 5], origin='upper')  
# Adjust extent to match your world coordinates; change ranges if needed.

# Plot paths
colors = plt.cm.plasma(np.linspace(0, 1, len(best_paths_per_gen)))
for gen_idx, path in enumerate(best_paths_per_gen):
    if path is None or len(path) == 0:
        continue
    path = np.array(path)
    # Flip x and y here:
    ax.plot(path[:, 1], path[:, 0], color=colors[gen_idx], linewidth=2, label=f'Gen {gen_idx+1}')

# Labels, legend, etc.
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Robot Paths Over Generations on Terrain Background')
ax.legend(loc='upper right', fontsize=8)
ax.grid(False)
plt.tight_layout()
plt.show()
