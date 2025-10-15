import numpy as np
import matplotlib.pyplot as plt

# =======================
# Load fitness data
# =======================
fitness_file = r"MyWork\Nested_Evolution\best_robot_fitness_POP25_GEN41.fitness_all.npy"
fitness_all = np.load(fitness_file, allow_pickle=True)

print(f"Loaded fitness data with shape: {fitness_all.shape}")

# =======================
# Compute statistics per generation (ignore very low values)
# =======================
gens = np.arange(len(fitness_all))

mean_fitness = []
min_fitness = []
max_fitness = []
low_value_counts = []

for i, f in enumerate(fitness_all):
    low_count = np.sum(f < -100)
    low_value_counts.append(low_count)

    f_valid = f[f > -100]  # ignore all fitness values less than -100
    if len(f_valid) > 0:
        mean_fitness.append(np.mean(f_valid))
        min_fitness.append(np.min(f_valid))
        max_fitness.append(np.max(f_valid))
    else:
        mean_fitness.append(np.nan)
        min_fitness.append(np.nan)
        max_fitness.append(np.nan)

# Convert lists to numpy arrays
mean_fitness = np.array(mean_fitness)
min_fitness = np.array(min_fitness)
max_fitness = np.array(max_fitness)
low_value_counts = np.array(low_value_counts)

# =======================
# Print ignored value stats
# =======================
print("\nCount of very low (< -100) fitness values per generation:")
for gen, count in zip(gens, low_value_counts):
    print(f"Generation {gen:2d}: {count} ignored")

# =======================
# Create single figure
# =======================
plt.figure(figsize=(10, 5))
plt.plot(gens, mean_fitness, color='royalblue', label='Mean Fitness', linewidth=2)
plt.fill_between(gens, min_fitness, max_fitness, color='royalblue', alpha=0.2, label='Range (min–max)')
plt.ylabel('Fitness')
plt.xlabel('Generation')
plt.title('Fitness Evolution Across Generations')
plt.ylim(-8, -4.5)
plt.xlim(0, len(gens)-1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
