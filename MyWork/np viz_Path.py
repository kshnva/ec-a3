import numpy as np
import matplotlib.pyplot as plt

# =======================
# Load saved paths
# =======================
paths_file = r"MyWork\Nested_Evolution\robot_path1.npy"

# MUST set allow_pickle=True because it's an object array
best_paths_per_gen = np.load(paths_file, allow_pickle=True)
best_paths_per_gen=best_paths_per_gen[:,1]

print(f"Loaded {len(best_paths_per_gen)} generations of best paths")

# Inspect contents safely
for i, path in enumerate(best_paths_per_gen):
    if path is None:
        print(f"Generation {i+1}: No path (non-learner body)")
    else:
        print(f"Generation {i+1}: Type={type(path)}, Length={len(path)}")

# =======================
# 2D XY plot of paths per generation
# =======================
plt.figure(figsize=(10, 8))

colors = plt.cm.viridis(np.linspace(0, 1, len(best_paths_per_gen)))

for gen_idx, path in enumerate(best_paths_per_gen):
    if path is None or len(path) == 0:
        continue  # skip missing paths
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], color=colors[gen_idx], label=f'Gen {gen_idx+1}')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Best Robot Paths Per Generation (XY Plane)')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()
