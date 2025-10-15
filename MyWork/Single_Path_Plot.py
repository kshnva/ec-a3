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
path_file = r"MyWork\Nested_Evolution\robot_path1.npy"
path = np.load(path_file, allow_pickle=True)

if path is None or len(path) == 0:
    raise ValueError("Path is empty!")

path = np.array(path)  # ensure it's a numpy array
print(f"Loaded path with {len(path)} points")

# =======================
# Plot
# =======================
fig, ax = plt.subplots(figsize=(8, 10))

# Show background
ax.imshow(bg_img, extent=[-2, 2, -2, 5], origin='upper')  
# Adjust extent to match your world coordinates; change ranges if needed.

# Plot path (use only X and Y)
ax.plot(path[:, 0], path[:, 1], color='royalblue', linewidth=2, label='Robot Path')

# Labels, legend, etc.
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('Robot Path on Terrain Background')
ax.legend(loc='upper right', fontsize=10)
ax.grid(False)
plt.tight_layout()
plt.show()
