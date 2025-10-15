"""Plot script for data files with _shania suffix."""

# Standard libraries
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.cm import viridis
import mujoco as mj

# Local libraries
try:
    from ariel.simulation.environments import OlympicArena
    from ariel.utils.renderers import single_frame_renderer
    ARIEL_AVAILABLE = True
except ImportError:
    ARIEL_AVAILABLE = False
    print("Warning: ariel library not available, 3D path visualization will be limited")

# --- DATA SETUP ---
CWD = Path.cwd()
DATA_DIR = CWD / "ImprovementRuns" / "MultiTerrain"
OUTPUT_DIR = CWD / "Plots_Shania"
OUTPUT_DIR.mkdir(exist_ok=True)

# Target position (from the original script)
TARGET_POSITION = [5.0, 0.0, 0.5]
TARGET = np.array(TARGET_POSITION, dtype=np.float64)

# Terrain spawns (from the original script)
SPAWN_FLAT = np.array([-2.40, 0.0, 0.10], dtype=np.float64)
SPAWN_RUGGED = np.array([-0.90, 0.0, 0.12], dtype=np.float64)
SPAWN_HILL = np.array([2.65, 0.0, 0.12], dtype=np.float64)
TERRAIN_SPAWNS = (
    ("flat", SPAWN_FLAT),
    ("rugged", SPAWN_RUGGED),
    ("hill", SPAWN_HILL),
)

def load_data_files():
    """Load all data files with _shania suffix."""
    # Find the files
    generation_data_file = next(DATA_DIR.glob("generation_data_*_shania.json"), None)
    best_genome_file = next(DATA_DIR.glob("best_genome_*_shania.json"), None)
    
    if not generation_data_file:
        raise FileNotFoundError("Could not find generation data file with _shania suffix")
    if not best_genome_file:
        print("Warning: Could not find best genome file with _shania suffix")
    
    # Load generation data
    with open(generation_data_file, 'r') as f:
        generation_data = json.load(f)
    
    # Convert generation keys back to integers (they're stored as strings in JSON)
    generation_data = {int(k): v for k, v in generation_data.items()}
    
    # Load best genome if available
    best_genome = None
    if best_genome_file:
        with open(best_genome_file, 'r') as f:
            best_genome = json.load(f)
    
    return generation_data, best_genome

def plot_fitness_progress(generation_data):
    """Plot fitness progress over generations."""
    generations = sorted([int(gen) for gen in generation_data.keys()])
    
    # Extract fitness values
    fitness_history = [generation_data[gen]["best_fitness"] for gen in generations]
    
    # Extract per-terrain fitness components
    terrain_history = {}
    for terrain_name, _ in TERRAIN_SPAWNS:
        terrain_history[terrain_name] = []
        for gen in generations:
            components = generation_data[gen]["fitness_components"]
            terrain_history[terrain_name].append(components.get(terrain_name, 0))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(generations, fitness_history, marker="o", linestyle="-", linewidth=2, 
             label="Total (best-of-gen)")
    
    for name, series in terrain_history.items():
        plt.plot(generations, series, linestyle="--", marker=".", label=name)
    
    plt.xlabel("Generation", fontsize=12)
    plt.ylabel("Fitness (sum of projected progress)", fontsize=12)
    plt.title("Evolution Progress (CMA-ES) — Brain Only, 3 Terrains", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Save the plot
    plot_path = OUTPUT_DIR / "fitness_progress_shania.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=200)
    print(f"Saved fitness progress plot: {plot_path}")
    plt.close()

def plot_robot_paths_2d(generation_data):
    """Create 2D visualizations of robot paths across generations and terrains."""
    generations = sorted([int(gen) for gen in generation_data.keys()])
    
    # Select which generations to plot - first, middle, and last generations
    if len(generations) >= 5:
        generations_to_plot = [
            generations[0],  # First
            generations[len(generations) // 4],  # 25%
            generations[len(generations) // 2],  # Middle
            generations[3 * len(generations) // 4],  # 75%
            generations[-1]  # Last
        ]
    else:
        generations_to_plot = generations
    
    # Create subplots for each terrain
    fig, axes = plt.subplots(1, len(TERRAIN_SPAWNS), figsize=(15, 5))
    if len(TERRAIN_SPAWNS) == 1:
        axes = [axes]
    
    colors = viridis(np.linspace(0, 1, len(generations_to_plot)))
    
    for terrain_idx, (terrain_name, spawn_pos) in enumerate(TERRAIN_SPAWNS):
        ax = axes[terrain_idx]
        
        for gen_idx, gen in enumerate(generations_to_plot):
            if gen in generation_data and generation_data[gen]["robot_paths"] is not None:
                paths_data = generation_data[gen]["robot_paths"]
                if terrain_name in paths_data and paths_data[terrain_name] is not None:
                    path = paths_data[terrain_name]
                    path_array = np.array(path)
                    fitness = generation_data[gen]["fitness_components"].get(terrain_name, 0)
                    ax.plot(path_array[:, 0], path_array[:, 1], 
                           color=colors[gen_idx], linewidth=2, alpha=0.8,
                           label=f'Gen {gen} (fit={fitness:.2f})')
                    
                    # Mark start and end points
                    ax.scatter(path_array[0, 0], path_array[0, 1], 
                              color=colors[gen_idx], marker='o', s=100, alpha=0.8)
                    ax.scatter(path_array[-1, 0], path_array[-1, 1], 
                              color=colors[gen_idx], marker='s', s=100, alpha=0.8)
        
        # Mark spawn position and target
        ax.scatter(spawn_pos[0], spawn_pos[1], color='red', marker='*', s=200, 
                  label='Spawn', zorder=10)
        ax.scatter(TARGET[0], TARGET[1], color='gold', marker='*', s=200, 
                  label='Target', zorder=10)
        
        ax.set_xlabel('X Position', fontsize=11)
        ax.set_ylabel('Y Position', fontsize=11)
        ax.set_title(f'{terrain_name.capitalize()} Terrain', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=8)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    path_plot_file = OUTPUT_DIR / "robot_paths_2d_shania.png"
    plt.savefig(path_plot_file, dpi=200, bbox_inches='tight')
    print(f"Saved 2D robot path plot: {path_plot_file}")
    plt.close()

# def plot_robot_path_3d_visualization(generation_data, terrain_name="flat"):
#     """
#     Create a 3D visualization of the best robot path using ariel library.
#     This function is similar to the show_xpos_history function in A3_plot_function.py.
#     """
#     if not ARIEL_AVAILABLE:
#         print("Skipping 3D visualization - ariel library not available")
#         return
    
#     # Find best generation and its path
#     best_gen = max(generation_data.keys(), key=lambda g: generation_data[g]["best_fitness"])
    
#     # Get path data for the specific terrain
#     if generation_data[best_gen]["robot_paths"] is None or terrain_name not in generation_data[best_gen]["robot_paths"]:
#         print(f"No path data available for {terrain_name} terrain in generation {best_gen}")
#         return
    
#     history = generation_data[best_gen]["robot_paths"][terrain_name]
    
#     # Find the appropriate spawn position
#     spawn_position = next((spawn for name, spawn in TERRAIN_SPAWNS if name == terrain_name), SPAWN_FLAT)
    
#     # Create output directory
#     terrain_dir = OUTPUT_DIR / terrain_name
#     terrain_dir.mkdir(exist_ok=True)
    
#     # Initialize world to get the background
#     mj.set_mjcb_control(None)
#     world = OlympicArena(load_precompiled=False)
    
#     # Add objects to the world
#     start_sphere = r"""
#     <mujoco>
#         <worldbody>
#             <geom name="green_sphere"
#             size=".1"
#             rgba="0 1 0 1"/>
#         </worldbody>
#     </mujoco>
#     """
#     end_sphere = r"""
#     <mujoco>
#         <worldbody>
#             <geom name="red_sphere"
#             size=".1"
#             rgba="1 0 0 1"/>
#         </worldbody>
#     </mujoco>
#     """
#     target_box = r"""
#     <mujoco>
#         <worldbody>
#             <geom name="magenta_box"
#                 size=".1 .1 .1"
#                 type="box"
#                 rgba="1 0 1 0.75"/>
#         </worldbody>
#     </mujoco>
#     """
#     spawn_box = r"""
#     <mujoco>
#         <worldbody>
#             <geom name="gray_box"
#             size=".1 .1 .1"
#             type="box"
#             rgba="0.5 0.5 0.5 0.5"/>
#         </worldbody>
#     </mujoco>
#     """
#     # Convert list of [x,y,z] positions to numpy array
#     pos_data = np.array(history)
    
#     # Starting point of robot
#     adjustment = np.array((0, 0, TARGET_POSITION[2] + 1))
#     world.spawn(
#         mj.MjSpec.from_string(start_sphere),
#         position=pos_data[0] + (adjustment * 1.5),
#         correct_collision_with_floor=False,
#     )

#     # End point of robot
#     world.spawn(
#         mj.MjSpec.from_string(end_sphere),
#         position=pos_data[-1] + (adjustment * 2),
#         correct_collision_with_floor=False,
#     )

#     # Target position
#     world.spawn(
#         mj.MjSpec.from_string(target_box),
#         position=TARGET_POSITION + adjustment,
#         correct_collision_with_floor=False,
#     )

#     # Spawn position of robot
#     world.spawn(
#         mj.MjSpec.from_string(spawn_box),
#         position=spawn_position,
#         correct_collision_with_floor=False,
#     )

#     # Draw the path of the robot
#     smooth = np.linspace(0, 1, len(pos_data))
#     inv_smooth = 1 - smooth
#     smooth_rise = np.linspace(1.25, 1.95, len(pos_data))
#     for i in range(1, len(pos_data)):
#         # Get the two points to draw the distance between
#         pos_i = pos_data[i]
#         pos_j = pos_data[i - 1]

#         # Size of the box to represent the distance
#         distance = pos_i - pos_j
#         minimum_size = 0.05
#         geom_size = np.array([
#             max(abs(distance[0]) / 2, minimum_size),
#             max(abs(distance[1]) / 2, minimum_size),
#             max(abs(distance[2]) / 2, minimum_size),
#         ])
#         geom_size_str: str = f"{geom_size[0]} {geom_size[1]} {geom_size[2]}"

#         # Position the box in the middle of the two points
#         half_way_point = (pos_i + pos_j) / 2
#         geom_pos_str = (
#             f"{half_way_point[0]} {half_way_point[1]} {half_way_point[2]}"
#         )

#         # Smooth color transition from green to red
#         geom_rgba = f"{smooth[i]} {inv_smooth[i]} 0 0.75"
#         path_box = rf"""
#         <mujoco>
#             <worldbody>
#                 <geom name="yellow_sphere"
#                     type="box"
#                     pos="{geom_pos_str}"
#                     size="{geom_size_str}"
#                     rgba="{geom_rgba}"
#                 />
#             </worldbody>
#         </mujoco>
#         """
#         world.spawn(
#             mj.MjSpec.from_string(path_box),
#             position=(adjustment * smooth_rise[i]),
#             correct_collision_with_floor=False,
#         )

#     # Setup the plot
#     _, ax = plt.subplots(figsize=(10, 8))

#     # Add legend to the plot
#     plt.rc("legend", fontsize="small")
#     red_patch = mpatches.Patch(color="red", label="End Position")
#     gray_patch = mpatches.Patch(color="gray", label="Spawn Position")
#     green_patch = mpatches.Patch(color="green", label="Start Position")
#     magenta_patch = mpatches.Patch(color="magenta", label="Target Position")
#     yellow_patch = mpatches.Patch(color="yellow", label="Robot Path")
#     ax.legend(
#         handles=[
#             green_patch,
#             red_patch,
#             magenta_patch,
#             gray_patch,
#             yellow_patch,
#         ],
#         loc="upper left",
#         bbox_to_anchor=(1.05, 1),
#     )

#     # Add labels and title
#     ax.set_xlabel("Y Position")
#     ax.set_ylabel("X Position")
#     ax.get_xaxis().set_ticks([])
#     ax.get_yaxis().set_ticks([])

#     # Title
#     plt.title(f"Best Robot Path in {terrain_name.capitalize()} Terrain (Generation {best_gen})")

#     # Render the background image
#     model = world.spec.compile()
#     data = mj.MjData(model)
#     bg_path = str(terrain_dir / "background.png")
#     single_frame_renderer(
#         model,
#         data,
#         save_path=bg_path,
#         save=True,
#         width=200,
#         height=600,
#         cam_fovy=8,
#         cam_pos=[2.1, 0, 50],
#         cam_quat=[-0.7071, 0, 0, 0.7071],
#     )

#     # Setup background image
#     img = plt.imread(bg_path)
#     ax.imshow(img)

#     # Save the figure
#     fig_path = terrain_dir / f"robot_path_{terrain_name}_shania.png"
#     plt.savefig(fig_path, bbox_inches="tight", dpi=300)
#     print(f"Saved 3D robot path visualization for {terrain_name} terrain: {fig_path}")
#     plt.close()

def plot_robot_path_3d_visualization(generation_data, terrain_name="flat"):
    """
    Create a 3D visualization of the best robot path using ariel library.
    This function is similar to the show_xpos_history function in A3_plot_function.py.
    """
    if not ARIEL_AVAILABLE:
        print("Skipping 3D visualization - ariel library not available")
        return
    
    # Find best generation and its path
    best_gen = max(generation_data.keys(), key=lambda g: generation_data[g]["best_fitness"])
    
    # Get path data for the specific terrain
    if generation_data[best_gen]["robot_paths"] is None or terrain_name not in generation_data[best_gen]["robot_paths"]:
        print(f"No path data available for {terrain_name} terrain in generation {best_gen}")
        return
    
    history = generation_data[best_gen]["robot_paths"][terrain_name]
    
    # Reduce path density significantly to avoid memory issues - sample every N points
    path_sample_rate = max(1, len(history) // 25)  # Limit to ~25 path segments
    history = history[::path_sample_rate]
    
    # Find the appropriate spawn position
    spawn_position = next((spawn for name, spawn in TERRAIN_SPAWNS if name == terrain_name), SPAWN_FLAT)
    
    # Create output directory
    terrain_dir = OUTPUT_DIR / terrain_name
    terrain_dir.mkdir(exist_ok=True)
    
    # Initialize world - we'll reduce the number of objects instead of trying to increase limits
    mj.set_mjcb_control(None)
    world = OlympicArena(load_precompiled=False)
    
    # Add marker objects (fewer, more efficient)
    start_sphere = r"""
    <mujoco>
        <worldbody>
            <geom name="green_sphere"
            size=".15"
            rgba="0 1 0 1"/>
        </worldbody>
    </mujoco>
    """
    end_sphere = r"""
    <mujoco>
        <worldbody>
            <geom name="red_sphere"
            size=".15"
            rgba="1 0 0 1"/>
        </worldbody>
    </mujoco>
    """
    target_box = r"""
    <mujoco>
        <worldbody>
            <geom name="magenta_box"
                size=".15 .15 .15"
                type="box"
                rgba="1 0 1 0.75"/>
        </worldbody>
    </mujoco>
    """
    spawn_box = r"""
    <mujoco>
        <worldbody>
            <geom name="gray_box"
            size=".15 .15 .15"
            type="box"
            rgba="0.5 0.5 0.5 0.5"/>
        </worldbody>
    </mujoco>
    """
    
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Starting point of robot
    adjustment = np.array((0, 0, TARGET_POSITION[2] + 1))
    world.spawn(
        mj.MjSpec.from_string(start_sphere),
        position=pos_data[0] + (adjustment * 1.5),
        correct_collision_with_floor=False,
    )

    # End point of robot
    world.spawn(
        mj.MjSpec.from_string(end_sphere),
        position=pos_data[-1] + (adjustment * 2),
        correct_collision_with_floor=False,
    )

    # Target position
    world.spawn(
        mj.MjSpec.from_string(target_box),
        position=TARGET_POSITION + adjustment,
        correct_collision_with_floor=False,
    )

    # Spawn position of robot
    world.spawn(
        mj.MjSpec.from_string(spawn_box),
        position=spawn_position,
        correct_collision_with_floor=False,
    )

    # Draw a simplified path (every Nth point to reduce memory usage)
    smooth = np.linspace(0, 1, len(pos_data))
    inv_smooth = 1 - smooth
    smooth_rise = np.linspace(1.25, 1.95, len(pos_data))
    
    # Only draw path segments for every few points to reduce memory significantly
    step_size = max(1, len(pos_data) // 10)  # Limit to ~10 path segments total
    
    for i in range(step_size, len(pos_data), step_size):
        # Get the two points to draw the distance between
        pos_i = pos_data[i]
        pos_j = pos_data[i - step_size]

        # Size of the box to represent the distance (larger for visibility)
        distance = pos_i - pos_j
        minimum_size = 0.08
        geom_size = np.array([
            max(abs(distance[0]) / 2, minimum_size),
            max(abs(distance[1]) / 2, minimum_size),
            max(abs(distance[2]) / 2, minimum_size),
        ])
        geom_size_str: str = f"{geom_size[0]} {geom_size[1]} {geom_size[2]}"

        # Position the box in the middle of the two points
        half_way_point = (pos_i + pos_j) / 2
        geom_pos_str = (
            f"{half_way_point[0]} {half_way_point[1]} {half_way_point[2]}"
        )

        # Smooth color transition from green to red
        geom_rgba = f"{smooth[i]} {inv_smooth[i]} 0 0.75"
        path_box = rf"""
        <mujoco>
            <worldbody>
                <geom name="path_segment_{i}"
                    type="box"
                    pos="{geom_pos_str}"
                    size="{geom_size_str}"
                    rgba="{geom_rgba}"
                />
            </worldbody>
        </mujoco>
        """
        world.spawn(
            mj.MjSpec.from_string(path_box),
            position=(adjustment * smooth_rise[i]),
            correct_collision_with_floor=False,
        )

    # Setup the plot
    _, ax = plt.subplots(figsize=(10, 8))

    # Add legend to the plot
    plt.rc("legend", fontsize="small")
    red_patch = mpatches.Patch(color="red", label="End Position")
    gray_patch = mpatches.Patch(color="gray", label="Spawn Position")
    green_patch = mpatches.Patch(color="green", label="Start Position")
    magenta_patch = mpatches.Patch(color="magenta", label="Target Position")
    yellow_patch = mpatches.Patch(color="yellow", label="Robot Path")
    ax.legend(
        handles=[
            green_patch,
            red_patch,
            magenta_patch,
            gray_patch,
            yellow_patch,
        ],
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
    )

    # Add labels and title
    ax.set_xlabel("Y Position")
    ax.set_ylabel("X Position")
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    # Title
    plt.title(f"Best Robot Path in {terrain_name.capitalize()} Terrain (Generation {best_gen})")

    try:
        # Render the background image
        model = world.spec.compile()
        data = mj.MjData(model)
        bg_path = str(terrain_dir / "background.png")
        single_frame_renderer(
            model,
            data,
            save_path=bg_path,
            save=True,
            width=200,
            height=600,
            cam_fovy=8,
            cam_pos=[2.1, 0, 50],
            cam_quat=[-0.7071, 0, 0, 0.7071],
        )

        # Setup background image
        img = plt.imread(bg_path)
        ax.imshow(img)
        
        print(f"Successfully rendered 3D visualization for {terrain_name} terrain")
        
    except Exception as e:
        print(f"Warning: Could not render 3D background for {terrain_name} terrain: {e}")
        print("Saving plot without 3D background...")

    # Save the figure
    fig_path = terrain_dir / f"robot_path_{terrain_name}_shania.png"
    plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"Saved 3D robot path visualization for {terrain_name} terrain: {fig_path}")
    plt.close()

def extract_network_metrics(generation_data):
    """Extract network analysis metrics from the generative hypothesis data."""
    generations = sorted([int(gen) for gen in generation_data.keys() if generation_data[gen].get("network_analysis") is not None])
    
    if len(generations) < 2:
        print("Insufficient network analysis data for plotting")
        return None, None
    
    # Extract metrics
    metrics = {
        'fitness': [],
        'pattern_reuse': [],
        'modularity': [],
        'symmetry': [],
        'efficiency': [],
        'sparsity': []
    }
    
    for gen in generations:
        analysis = generation_data[gen]["network_analysis"]
        if analysis is None:
            continue
            
        metrics['fitness'].append(generation_data[gen]["best_fitness"])
        
        if "pattern_reuse" in analysis:
            metrics['pattern_reuse'].append(analysis["pattern_reuse"].get("total_repeated_patterns", 0))
        else:
            metrics['pattern_reuse'].append(0)
            
        if "functional_modularity" in analysis:
            metrics['modularity'].append(analysis["functional_modularity"].get("modularity_score", 0))
        else:
            metrics['modularity'].append(0)
            
        if "structural_regularity" in analysis:
            metrics['symmetry'].append(analysis["structural_regularity"].get("recurrent_symmetry", 0))
        else:
            metrics['symmetry'].append(0)
            
        if "efficiency_metrics" in analysis:
            metrics['efficiency'].append(analysis["efficiency_metrics"].get("connection_efficiency", 0))
        else:
            metrics['efficiency'].append(0)
            
        if "weight_statistics" in analysis:
            metrics['sparsity'].append(analysis["weight_statistics"].get("sparsity", 0))
        else:
            metrics['sparsity'].append(0)
            
    return generations, metrics

def plot_modularity_evolution(generations, metrics):
    """Plot evolution of modularity over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(generations, metrics['modularity'], 'bo-', alpha=0.7, linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Modularity Score', fontsize=12)
    plt.title('Functional Modularity Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(generations) > 2:
        z = np.polyfit(generations, metrics['modularity'], 1)
        p = np.poly1d(z)
        plt.plot(generations, p(generations), "r--", alpha=0.8, 
                 label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        plt.legend()
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "modularity_evolution_shania.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved modularity evolution plot: {plot_path}")
    plt.close()

def plot_efficiency_fitness(generations, metrics):
    """Plot efficiency and fitness evolution as dual-axis line plot."""
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    # Create primary axis for efficiency
    color1 = '#2E86AB'  # Blue
    ax1.set_xlabel('Generation', fontsize=13)
    ax1.set_ylabel('Connection Efficiency', color=color1, fontsize=13)
    line1 = ax1.plot(generations, metrics['efficiency'], 'o-', color=color1, 
                     linewidth=2.5, markersize=6, label='Connection Efficiency')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # Create secondary axis for fitness
    ax2 = ax1.twinx()
    color2 = '#F18F01'  # Orange
    ax2.set_ylabel('Fitness', color=color2, fontsize=13)
    line2 = ax2.plot(generations, metrics['fitness'], 's-', color=color2, 
                     linewidth=2.5, markersize=6, label='Fitness')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Styling improvements
    ax1.set_title('Evolution of Connection Efficiency and Fitness', fontsize=15, pad=20)
    
    # Add trend lines if we have enough data points
    if len(generations) > 2:
        # Efficiency trend
        z1 = np.polyfit(generations, metrics['efficiency'], 1)
        p1 = np.poly1d(z1)
        ax1.plot(generations, p1(generations), "--", alpha=0.6, color=color1, linewidth=1.5)
        
        # Fitness trend
        z2 = np.polyfit(generations, metrics['fitness'], 1)
        p2 = np.poly1d(z2)
        ax2.plot(generations, p2(generations), "--", alpha=0.6, color=color2, linewidth=1.5)
        
        # Add correlation coefficient and trends in a nice info box
        corr_coef = np.corrcoef(metrics['efficiency'], metrics['fitness'])[0, 1]
        info_text = (f'Correlation: {corr_coef:.3f}\n'
                    f'Efficiency trend: {z1[0]:+.4f}/gen\n'
                    f'Fitness trend: {z2[0]:+.4f}/gen')
        
        ax1.text(0.02, 0.98, info_text, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor='gray'))
    
    # Add combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', 
              bbox_to_anchor=(0.98, 0.85), frameon=True, fancybox=True, shadow=True)
    
    # Improve layout
    fig.tight_layout()
    
    # Add some padding around the data
    ax1.margins(x=0.02)
    ax2.margins(x=0.02)
    
    # Save with high quality
    plot_path = OUTPUT_DIR / "efficiency_vs_fitness_shania.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"Saved efficiency vs fitness line plot: {plot_path}")
    plt.close()

def plot_sparsity_evolution(generations, metrics):
    """Plot evolution of network sparsity over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(generations, metrics['sparsity'], 'ro-', alpha=0.7, linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Network Sparsity', fontsize=12)
    plt.title('Network Sparsity Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(generations) > 2:
        z = np.polyfit(generations, metrics['sparsity'], 1)
        p = np.poly1d(z)
        plt.plot(generations, p(generations), "k--", alpha=0.8, 
                 label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        plt.legend()
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "sparsity_evolution_shania.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved sparsity evolution plot: {plot_path}")
    plt.close()

def plot_symmetry_evolution(generations, metrics):
    """Plot evolution of structural symmetry over generations."""
    plt.figure(figsize=(10, 6))
    plt.plot(generations, metrics['symmetry'], 'go-', alpha=0.7, linewidth=2)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Recurrent Symmetry', fontsize=12)
    plt.title('Structural Symmetry Evolution', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add trend line
    if len(generations) > 2:
        z = np.polyfit(generations, metrics['symmetry'], 1)
        p = np.poly1d(z)
        plt.plot(generations, p(generations), "k--", alpha=0.8, 
                 label=f'Trend: {z[0]:.4f}x + {z[1]:.4f}')
        plt.legend()
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "symmetry_evolution_shania.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved symmetry evolution plot: {plot_path}")
    plt.close()

def plot_correlation_matrix(generations, metrics):
    """Plot correlation matrix of network metrics."""
    import matplotlib.cm as cm
    
    plt.figure(figsize=(10, 8))
    
    metric_names = ['Fitness', 'Pattern Reuse', 'Modularity', 'Symmetry', 'Efficiency', 'Sparsity']
    metric_values = np.array([metrics['fitness'], metrics['pattern_reuse'], metrics['modularity'], 
                             metrics['symmetry'], metrics['efficiency'], metrics['sparsity']])
    
    if metric_values.shape[1] > 2:  # Need at least 3 data points for meaningful correlation
        correlation_matrix = np.corrcoef(metric_values)
        correlation_matrix = np.nan_to_num(correlation_matrix)  # Replace NaN with 0
        
        im = plt.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
        plt.colorbar(im, shrink=0.8, label='Correlation Coefficient')
        
        plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha='right')
        plt.yticks(range(len(metric_names)), metric_names)
        
        # Add correlation values to heatmap
        for i in range(len(metric_names)):
            for j in range(len(metric_names)):
                plt.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                        ha="center", va="center", 
                        color="black" if abs(correlation_matrix[i, j]) < 0.5 else "white",
                        fontsize=9)
        
        plt.title('Neural Network Metric Correlations', fontsize=14)
    else:
        plt.text(0.5, 0.5, 'Insufficient data\nfor correlation analysis', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14)
        plt.title('Metric Correlations (Need more data)')
    
    plt.tight_layout()
    plot_path = OUTPUT_DIR / "metric_correlation_matrix_shania.png"
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    print(f"Saved metric correlation matrix: {plot_path}")
    plt.close()

def plot_network_analysis(generation_data):
    """Plot all network analysis metrics as separate figures."""
    generations, metrics = extract_network_metrics(generation_data)
    
    if generations is None or len(generations) < 2:
        print("Insufficient data for network analysis plots")
        return
    
    # Create a folder for network analysis plots
    network_dir = OUTPUT_DIR / "network_analysis"
    network_dir.mkdir(exist_ok=True)
    
    # Create individual plots
    plot_modularity_evolution(generations, metrics)
    plot_efficiency_fitness(generations, metrics)
    plot_sparsity_evolution(generations, metrics)
    plot_symmetry_evolution(generations, metrics)
    plot_correlation_matrix(generations, metrics)

def main():
    """Main function to generate all plots."""
    print("Loading data files...")
    generation_data, best_genome = load_data_files()
    print(f"Loaded data for {len(generation_data)} generations")
    
    print("\nGenerating fitness progress plot...")
    plot_fitness_progress(generation_data)
    
    print("\nGenerating 2D robot path plots...")
    plot_robot_paths_2d(generation_data)
    
    print("\nGenerating 3D robot path visualizations...")
    for terrain_name, _ in TERRAIN_SPAWNS:
        plot_robot_path_3d_visualization(generation_data, terrain_name)
    
    print("\nGenerating network analysis plots...")
    plot_network_analysis(generation_data)
    
    print("\nAll plots saved to:", OUTPUT_DIR)

if __name__ == "__main__":
    main()