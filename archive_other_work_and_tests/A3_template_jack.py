"""EC A3 Template Code (Jack)."""

# Standard library
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from dataclasses import dataclass
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
import numpy.typing as npt
from mujoco import viewer
from rich.progress import track

# EvoTorch for advanced evolutionary algorithms
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES
from evotorch.logging import StdOutLogger, PandasLogger

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    CoreModule,
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import (
    single_frame_renderer,
    tracking_video_renderer,
    video_renderer,
)
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal[
    "launcher",
    "video",
    "simple",
    "tracking",
    "no_control",
    "frame",
]
type Vector = npt.NDArray[np.float64]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

# === ROBOT OLYMPICS CONFIGURATION === #
# Environment parameters
SPAWN_POS = [0, 0, 0.1]  # Starting position [x, y, z]
TARGET_POSITION = [5, 0, 0.5]  # Target position [x, y, z]

# Robot morphology parameters  
NUM_OF_MODULES = 25  # Number of modules in robot body
GENOTYPE_SIZE = 64  # Size of each gene array

# Evolution parameters - Optimized for CMA-ES
POPULATION_SIZE = 24  # CMA-ES works well with moderate population sizes (4+floor(3*ln(n)))
MAX_GENERATIONS = 150  # More generations for CMA-ES convergence
ELITE_COUNT = 6  # Keep more good solutions
TOURNAMENT_SIZE = 4  # Stronger selection pressure
CROSSOVER_PROBABILITY = 0.8  # More genetic mixing

# Simulation parameters
SIMULATION_DURATION = 45  # Longer simulation for better evaluation
FINAL_EVAL_DURATION = 90  # Much longer final evaluation

# Mutation parameters - More aggressive early, refined later
GENOTYPE_MUTATION_RATE = 0.2  # Higher initial mutation
GENOTYPE_MUTATION_SCALE = 0.15  # Larger mutations
CONTROLLER_MUTATION_RATE = 0.15
CONTROLLER_MUTATION_SCALE = 0.12


# >>> CHANGED: Enhanced fitness function for Robot Olympics
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    """
    Multi-objective fitness function that evaluates:
    1. Distance to target (primary objective)
    2. Movement efficiency (distance traveled)
    3. Stability (vertical oscillations)
    4. Forward progress (x-direction movement)
    """
    if not history or len(history) < 2:
        return -1000.0  # Penalty for no movement
    
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    x_start, y_start, z_start = history[0]
    
    # Primary objective: distance to target (minimize)
    target_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    
    # Secondary objectives
    # 1. Forward progress in x-direction
    x_progress = xc - x_start
    
    # 2. Stability - penalize excessive vertical oscillations
    z_vals = [pos[2] for pos in history]
    z_stability = -np.var(z_vals) if len(z_vals) > 1 else 0
    
    # 3. Movement efficiency - total distance traveled
    total_distance = 0.0
    for i in range(1, len(history)):
        dx = history[i][0] - history[i-1][0]
        dy = history[i][1] - history[i-1][1] 
        dz = history[i][2] - history[i-1][2]
        total_distance += np.sqrt(dx**2 + dy**2 + dz**2)
    
    # Enhanced fitness with better reward shaping
    base_fitness = (
        -target_distance * 15.0 +      # Stronger penalty for distance
        x_progress * 3.0 +             # Higher reward for forward movement
        z_stability * 1.0 +            # Better stability reward
        min(total_distance, 8.0) * 0.2  # Encourage movement but cap it
    )
    
    # Progressive bonuses for getting closer to target
    if target_distance < 2.0:
        base_fitness += 20.0 * (2.0 - target_distance)  # Bonus for < 2 units
    if target_distance < 1.0:
        base_fitness += 40.0 * (1.0 - target_distance)  # Big bonus for < 1 unit
    if target_distance < 0.5:
        base_fitness += 80.0 * (0.5 - target_distance)  # Huge bonus for < 0.5 units
    if target_distance < 0.2:
        base_fitness += 200.0 * (0.2 - target_distance)  # Massive bonus for very close
    
    # Additional rewards for final position quality
    final_z_penalty = abs(zc - zt) * 5.0  # Penalize wrong height
    final_y_penalty = abs(yc - yt) * 5.0  # Penalize lateral deviation
    
    fitness = base_fitness - final_z_penalty - final_y_penalty
    
    # Survival bonus - reward robots that don't fall over
    if zc > 0.05:  # Robot is still upright
        fitness += 5.0
    
    return fitness
## >>> CHANGED: Enhanced mutation functions with adaptive rates
def mutate_genotype(
    genotype: list[np.ndarray], 
    mutation_rate: float = 0.2, 
    mutation_scale: float = 0.15,
    generation: int = 0,
    max_generations: int = 100
) -> list[np.ndarray]:
    """
    Enhanced adaptive mutation with fine-tuning in later generations.
    """
    progress = generation / max_generations
    
    # Adaptive mutation - high exploration early, fine-tuning later
    if progress < 0.5:
        # Early phase: higher mutation for exploration
        adaptive_rate = mutation_rate * (1.0 - 0.3 * progress)
        adaptive_scale = mutation_scale * (1.0 - 0.2 * progress)
    else:
        # Later phase: fine-tuning with smaller mutations
        late_progress = (progress - 0.5) * 2.0
        adaptive_rate = mutation_rate * 0.7 * (1.0 - 0.5 * late_progress)
        adaptive_scale = mutation_scale * 0.8 * (1.0 - 0.6 * late_progress)
    
    new_genotype = []
    for arr in genotype:
        # Apply mutation mask
        mask = RNG.random(arr.shape) < adaptive_rate
        
        # Different noise strategies based on generation
        if progress < 0.3:
            # Early: bold exploration
            noise = RNG.normal(0, adaptive_scale * 1.5, size=arr.shape)
        elif progress < 0.7:
            # Mid: balanced exploration/exploitation
            gaussian_noise = RNG.normal(0, adaptive_scale, size=arr.shape)
            uniform_noise = RNG.uniform(-adaptive_scale, adaptive_scale, size=arr.shape)
            noise_type = RNG.random(arr.shape) < 0.7
            noise = np.where(noise_type, gaussian_noise, uniform_noise)
        else:
            # Late: fine-tuning
            noise = RNG.normal(0, adaptive_scale * 0.5, size=arr.shape)
        
        new_arr = arr + mask * noise
        # Clip to reasonable bounds
        new_arr = np.clip(new_arr, 0.0, 1.0)
        new_genotype.append(new_arr.astype(arr.dtype))
    
    return new_genotype

def mutate_controller_weights(
    weights: tuple[Vector, ...], 
    mutation_rate: float = 0.1, 
    mutation_scale: float = 0.1,
    generation: int = 0,
    max_generations: int = 100
) -> tuple[Vector, ...]:
    """
    Adaptive mutation for neural network weights.
    """
    # Adaptive parameters
    adaptive_rate = mutation_rate * (1.0 - 0.5 * generation / max_generations)
    adaptive_scale = mutation_scale * (1.0 - 0.3 * generation / max_generations)
    
    new_weights = []
    for w in weights:
        mask = RNG.random(w.shape) < adaptive_rate
        
        # Layer-specific mutation scaling
        if len(w.shape) == 2:  # Weight matrices
            noise = RNG.normal(0, adaptive_scale / np.sqrt(w.shape[0]), size=w.shape)
        else:  # Bias vectors
            noise = RNG.normal(0, adaptive_scale, size=w.shape)
        
        new_w = w + mask * noise
        # Clip weights to prevent explosion
        new_w = np.clip(new_w, -5.0, 5.0)
        new_weights.append(new_w)
    
    return tuple(new_weights)

def crossover_genotypes(parent1: list[np.ndarray], parent2: list[np.ndarray], generation: int = 0, max_generations: int = 100) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Adaptive crossover between two genotypes.
    """
    progress = generation / max_generations
    child1, child2 = [], []
    
    for arr1, arr2 in zip(parent1, parent2):
        if progress < 0.3:
            # Early: more uniform mixing
            mask = RNG.random(arr1.shape) < 0.5
        elif progress < 0.7:
            # Mid: blend crossover for smoother combination
            alpha = RNG.random(arr1.shape)
            child1_arr = alpha * arr1 + (1 - alpha) * arr2
            child2_arr = alpha * arr2 + (1 - alpha) * arr1
            child1.append(child1_arr.astype(arr1.dtype))
            child2.append(child2_arr.astype(arr1.dtype))
            continue
        else:
            # Late: more conservative crossover
            mask = RNG.random(arr1.shape) < 0.3  # Less mixing
        
        if 'mask' in locals():
            child1_arr = np.where(mask, arr1, arr2)
            child2_arr = np.where(mask, arr2, arr1)
            child1.append(child1_arr.astype(arr1.dtype))
            child2.append(child2_arr.astype(arr1.dtype))
    
    return child1, child2


def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.savefig(DATA / "robot_path.png")


def create_robot_body(
    genotype: list[np.ndarray] | None = None,
    *,
    save_graph: bool = True,
) -> CoreModule:
    # Create random genotype if None is provided
    if genotype is None:
        type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        genotype = [
            type_p_genes,
            conn_p_genes,
            rot_p_genes,
        ]

    # Decode the genotype into probability matrices
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    # Decode the high-probability graph
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph[Any] = hpd.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    # Save the graph to a file
    if save_graph is True:
        save_graph_as_json(
            robot_graph,
            DATA / "robot_graph.json",
        )

    # Print all nodes
    return construct_mjspec_from_graph(robot_graph)


def quick_spawn(
    robot: CoreModule,
) -> tuple[mj.MjModel, mj.MjData, OlympicArena]:
    mj.set_mjcb_control(None)
    world = OlympicArena()
    temp_robot_xml = robot.spec.to_xml()
    temp_robot = mj.MjSpec.from_string(temp_robot_xml)
    world.spawn(
        temp_robot,
        position=SPAWN_POS,
    )
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    return (cast("mj.MjModel", model), data, world)


class NN:
    def __init__(self, robot: CoreModule) -> None:
        _, data, _ = quick_spawn(robot)

        # Use fixed maximum sizes to avoid dimension mismatches
        self.max_input_size = 100  # Large enough for any reasonable robot
        self.hidden_size = 16  # Increased capacity for better control
        self.max_output_size = 50   # Large enough for any reasonable robot
        
        # Store actual sizes for this robot
        self.actual_qpos_size = len(data.qpos.copy())
        self.actual_qvel_size = len(data.qvel.copy())
        self.actual_output_size = len(data.ctrl)
        
        # Use fixed sizes for network architecture
        self.input_size = self.max_input_size
        self.output_size = self.max_output_size
        
        # Add bias terms
        self.use_bias = True
        
        # Initialize weights
        self.weights = None
        
        # Memory for temporal context
        self.prev_inputs = None
        self.output_momentum = 0.1  # Add momentum to outputs

        # Clear cache
        del data

    def random_controller(self) -> None:
        """Initialize neural network weights with fixed dimensions."""
        # Xavier initialization for better gradient flow
        w1_scale = np.sqrt(2.0 / (self.max_input_size + self.hidden_size))
        w2_scale = np.sqrt(2.0 / (self.hidden_size + self.hidden_size))
        w3_scale = np.sqrt(2.0 / (self.hidden_size + self.max_output_size))
        
        w1 = RNG.normal(0.0, w1_scale, size=(self.max_input_size, self.hidden_size))
        w2 = RNG.normal(0.0, w2_scale, size=(self.hidden_size, self.hidden_size))
        w3 = RNG.normal(0.0, w3_scale, size=(self.hidden_size, self.max_output_size))
        
        # Add biases if enabled
        if self.use_bias:
            b1 = RNG.normal(0.0, 0.1, size=self.hidden_size)
            b2 = RNG.normal(0.0, 0.1, size=self.hidden_size)
            b3 = RNG.normal(0.0, 0.1, size=self.max_output_size)
            self.weights = (w1, w2, w3, b1, b2, b3)
        else:
            self.weights = (w1, w2, w3)

    def set_controller_weights(self, weights: tuple[Vector, ...]) -> None:
        """Set the neural network weights - now with fixed dimensions."""
        expected_length = 6 if self.use_bias else 3
        
        if len(weights) != expected_length:
            self.random_controller()
            return
            
        # Check if weight dimensions match fixed network structure
        try:
            if self.use_bias and len(weights) == 6:
                w1, w2, w3, b1, b2, b3 = weights
                if (w1.shape == (self.max_input_size, self.hidden_size) and
                    w2.shape == (self.hidden_size, self.hidden_size) and
                    w3.shape == (self.hidden_size, self.max_output_size) and
                    b1.shape == (self.hidden_size,) and
                    b2.shape == (self.hidden_size,) and
                    b3.shape == (self.max_output_size,)):
                    self.weights = weights
                else:
                    self.random_controller()
            else:
                w1, w2, w3 = weights[:3]
                if (w1.shape == (self.max_input_size, self.hidden_size) and
                    w2.shape == (self.hidden_size, self.hidden_size) and
                    w3.shape == (self.hidden_size, self.max_output_size)):
                    self.weights = weights
                else:
                    self.random_controller()
                    
        except (ValueError, IndexError, AttributeError):
            self.random_controller()

    def forward(self, model: mj.MjModel, data: mj.MjData) -> npt.NDArray[np.float64]:
        """Forward pass with fixed-size network and input padding."""
        if self.weights is None:
            self.random_controller()
            
        # Get current robot state
        pos_inputs = data.qpos
        vel_inputs = data.qvel
        
        # Enhanced input processing
        # Normalize positions and velocities more carefully
        pos_inputs = np.tanh(pos_inputs * 0.5)  # Keep positions in reasonable range
        vel_inputs = np.tanh(vel_inputs * 0.2)  # More conservative velocity scaling
        
        # Combine position and velocity inputs
        current_inputs = np.concatenate([pos_inputs, vel_inputs])
        
        # Add temporal context if available
        if self.prev_inputs is not None and len(self.prev_inputs) == len(current_inputs):
            # Include input derivatives for temporal awareness
            input_diff = current_inputs - self.prev_inputs
            enhanced_inputs = np.concatenate([current_inputs, input_diff * 0.1])
        else:
            enhanced_inputs = current_inputs
            
        self.prev_inputs = current_inputs.copy()
        
        # Pad inputs to fixed size
        inputs = np.zeros(self.max_input_size)
        actual_size = min(len(enhanced_inputs), self.max_input_size)
        inputs[:actual_size] = enhanced_inputs[:actual_size]
        
        # Forward pass through network
        if self.use_bias and len(self.weights) == 6:
            w1, w2, w3, b1, b2, b3 = self.weights
            
            # Layer 1: input -> hidden
            layer1 = np.tanh(np.dot(inputs, w1) + b1)
            
            # Layer 2: hidden -> hidden  
            layer2 = np.tanh(np.dot(layer1, w2) + b2)
            
            # Layer 3: hidden -> output
            full_outputs = np.tanh(np.dot(layer2, w3) + b3)
        else:
            # Fallback to basic network without bias
            w1, w2, w3 = self.weights[:3]
            
            layer1 = np.tanh(np.dot(inputs, w1))
            layer2 = np.tanh(np.dot(layer1, w2))
            full_outputs = np.tanh(np.dot(layer2, w3))
        
        # Extract actual outputs for this robot
        outputs = full_outputs[:self.actual_output_size]
        
        # Scale outputs to appropriate range for joint control
        return outputs * (np.pi / 2)


def run(
    model: mj.MjModel,
    data: mj.MjData,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "tracking":
            # This records a video of the simulation
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)

            # Render with video recorder
            tracking_video_renderer(
                model,
                data,
                duration=duration,
                video_recorder=video_recorder,
            )
        case "launcher":
            # This opens a liver viewer of the simulation
            viewer.launch(
                model=model,
                data=data,
            )
        case "no_control":
            # If mj.set_mjcb_control(None), you can control the limbs manually.
            mj.set_mjcb_control(None)
            viewer.launch(
                model=model,
                data=data,
            )


def evaluate(
    robot: CoreModule,
    nn: NN,
    *,
    plot_and_record: bool = False,
    duration: int = 30,
    max_attempts: int = 3,
) -> float:
    """
    Evaluate a robot's performance with error handling and multiple attempts.
    """
    best_fitness = -np.inf
    
    for attempt in range(max_attempts):
        try:
            # Define what to track
            mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
            name_to_bind = "core"
            tracker = Tracker(
                mujoco_obj_to_find=mujoco_type_to_find,
                name_to_bind=name_to_bind,
            )

            # Create the controller
            controller = Controller(
                controller_callback_function=nn.forward,
                tracker=tracker,
            )

            # Create the robot in the world
            model, data, world = quick_spawn(robot)

            # Pass the model and data to the tracker
            controller.tracker.setup(world.spec, data)

            # Set the control callback function
            mj.set_mjcb_control(controller.set_control)

            # Run simulation
            if plot_and_record:
                # Extended duration for final evaluation
                run(model, data, duration=duration * 2, mode="video")
                show_xpos_history(tracker.history["xpos"][0])
            else:
                run(model, data, duration=duration, mode="simple")

            # Calculate fitness
            if tracker.history and "xpos" in tracker.history and tracker.history["xpos"]:
                current_fitness = fitness_function(tracker.history["xpos"][0])
                best_fitness = max(best_fitness, current_fitness)
                
                # If we got a reasonable result, break early
                if current_fitness > -100:
                    break
            else:
                pass
                # console.log(f"Warning: No tracking data available (attempt {attempt + 1})")
                
        except Exception as e:
            # console.log(f"Evaluation error (attempt {attempt + 1}): {str(e)}")
            if attempt == max_attempts - 1:
                # Last attempt failed, return penalty
                return -1000.0
            continue
    
    return best_fitness if best_fitness > -np.inf else -1000.0


# >>> CHANGED: Enhanced evolutionary algorithm with better strategies
def main() -> None:
    """Entry point - Robot Olympics Evolution."""
    # Use configuration parameters
    POP_SIZE = POPULATION_SIZE
    GENERATIONS = MAX_GENERATIONS
    ELITE_SIZE = ELITE_COUNT
    TOURNAMENT_SIZE_LOCAL = TOURNAMENT_SIZE
    CROSSOVER_RATE = CROSSOVER_PROBABILITY
    
    # console.log(f"Starting Robot Olympics Evolution: {POP_SIZE} individuals, {GENERATIONS} generations")
    
    # Initialize population
    population = []
    for i in range(POP_SIZE):
        # Random genotype
        type_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        conn_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        rot_p_genes = RNG.random(GENOTYPE_SIZE).astype(np.float32)
        genotype = [type_p_genes, conn_p_genes, rot_p_genes]
        
        # Random controller
        robot = create_robot_body(genotype, save_graph=False)
        nn = NN(robot)
        nn.random_controller()
        weights = tuple(np.copy(w) for w in nn.weights)
        population.append((genotype, weights, None))
        
        # if i % 5 == 0:
        #     console.log(f"Initialized individual {i+1}/{POP_SIZE}")

    # Evolution tracking
    best_fitness = -np.inf
    best_individual = None
    fitness_history = []
    avg_fitness_history = []
    diversity_history = []
    
    for gen in track(range(GENERATIONS), description="Evolving robots..."):
        # Evaluate population
        new_population = []
        fitnesses = []
        
        for idx, (genotype, weights, _) in enumerate(population):
            try:
                robot = create_robot_body(genotype, save_graph=False)
                nn = NN(robot)
                nn.set_controller_weights(weights)
                
                # Evaluate fitness
                fitness = evaluate(robot, nn, plot_and_record=False, duration=SIMULATION_DURATION)
                fitnesses.append(fitness)
                new_population.append((genotype, nn.weights, fitness))
                
            except Exception as e:
                # console.log(f"Error evaluating individual {idx}: {str(e)}")
                # Add penalty fitness for failed individuals
                fitnesses.append(-1000.0)
                # Keep the old weights or generate new random ones
                try:
                    robot = create_robot_body(genotype, save_graph=False)
                    nn = NN(robot)
                    nn.random_controller()
                    new_population.append((genotype, nn.weights, -1000.0))
                except:
                    # If even that fails, skip this individual
                    new_population.append((genotype, weights, -1000.0))
            
            # Track best individual
            if fitness > best_fitness:
                best_fitness = fitness
                best_individual = (genotype, nn.weights)
        
        # Statistics
        avg_fitness = np.mean(fitnesses)
        fitness_history.append(np.max(fitnesses))
        avg_fitness_history.append(avg_fitness)
        
        # Calculate diversity (genetic diversity in first genotype array)
        genotype_diversity = np.mean([
            np.std([ind[0][0] for ind in new_population]),
        ])
        diversity_history.append(genotype_diversity)
        
        # console.log(f"Gen {gen}: Best={best_fitness:.3f}, Avg={avg_fitness:.3f}, Diversity={genotype_diversity:.4f}")
        
        # Selection and reproduction
        sorted_pop = sorted(new_population, key=lambda x: x[2], reverse=True)
        
        # Elitism - keep best individuals
        survivors = sorted_pop[:ELITE_SIZE]
        
        # Generate offspring
        while len(survivors) < POP_SIZE:
            # Tournament selection for parents
            parent1 = tournament_selection(sorted_pop, TOURNAMENT_SIZE_LOCAL, gen, GENERATIONS)
            parent2 = tournament_selection(sorted_pop, TOURNAMENT_SIZE_LOCAL, gen, GENERATIONS)
            
            # Crossover
            if RNG.random() < CROSSOVER_RATE:
                child1_genotype, child2_genotype = crossover_genotypes(parent1[0], parent2[0], gen, GENERATIONS)
                # For simplicity, don't crossover controller weights, just mutate
                child1_weights = parent1[1]
                child2_weights = parent2[1]
            else:
                child1_genotype, child1_weights = parent1[0], parent1[1]
                child2_genotype, child2_weights = parent2[0], parent2[1]
            
            # Mutation
            child1_genotype = mutate_genotype(child1_genotype, generation=gen, max_generations=GENERATIONS)
            child1_weights = mutate_controller_weights(child1_weights, generation=gen, max_generations=GENERATIONS)
            
            survivors.append((child1_genotype, child1_weights, None))
            
            if len(survivors) < POP_SIZE:
                child2_genotype = mutate_genotype(child2_genotype, generation=gen, max_generations=GENERATIONS)
                child2_weights = mutate_controller_weights(child2_weights, generation=gen, max_generations=GENERATIONS)
                survivors.append((child2_genotype, child2_weights, None))
        
        population = survivors[:POP_SIZE]
    
    # Final evaluation and visualization
    # console.log("Evolution complete! Evaluating best individual...")
    
    if best_individual is not None:
        robot = create_robot_body(best_individual[0], save_graph=True)
        nn = NN(robot)
        nn.set_controller_weights(best_individual[1])
        fitness_final = evaluate(robot, nn, plot_and_record=True, duration=FINAL_EVAL_DURATION)
        
        console.log(f"🏆 Final best fitness: {fitness_final:.3f}")
        
        # Save best individual data
        best_data = {
            'fitness': fitness_final,
            'genotype': [arr.tolist() for arr in best_individual[0]],
            'target_position': TARGET_POSITION,
            'spawn_position': SPAWN_POS
        }
        
        with open(DATA / "best_individual.json", 'w') as f:
            json.dump(best_data, f, indent=2)
            
        console.log(f"📊 Results saved to {DATA / 'best_individual.json'}")
        console.log(f"📈 Evolution analysis saved to {DATA / 'evolution_analysis.png'}")
    else:
        console.log("❌ No best individual found.")
    
    # Enhanced plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Fitness evolution
    ax1.plot(fitness_history, 'b-', label='Best Fitness', linewidth=2)
    ax1.plot(avg_fitness_history, 'r--', label='Average Fitness', alpha=0.7)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Fitness')
    ax1.set_title('Fitness Evolution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Diversity
    ax2.plot(diversity_history, 'g-', linewidth=2)
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Genetic Diversity')
    ax2.set_title('Population Diversity')
    ax2.grid(True, alpha=0.3)
    
    # Fitness distribution (final generation)
    final_fitnesses = [ind[2] for ind in sorted_pop if ind[2] is not None]
    ax3.hist(final_fitnesses, bins=10, alpha=0.7, color='purple')
    ax3.set_xlabel('Fitness')
    ax3.set_ylabel('Count')
    ax3.set_title('Final Generation Fitness Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Improvement rate
    improvement = np.diff(fitness_history)
    ax4.plot(improvement, 'orange', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.set_xlabel('Generation')
    ax4.set_ylabel('Fitness Improvement')
    ax4.set_title('Generation-to-Generation Improvement')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(DATA / "evolution_analysis.png", dpi=300, bbox_inches='tight')
    # console.log(f"Evolution analysis saved to {DATA / 'evolution_analysis.png'}")

def tournament_selection(population: list, tournament_size: int, generation: int = 0, max_generations: int = 100) -> tuple:
    """Enhanced tournament selection with adaptive pressure."""
    # Increase selection pressure over time
    progress = generation / max_generations
    effective_tournament_size = max(2, int(tournament_size * (1.0 + progress)))
    effective_tournament_size = min(effective_tournament_size, len(population))
    
    tournament = RNG.choice(len(population), size=effective_tournament_size, replace=False)
    tournament_individuals = [population[i] for i in tournament]
    
    # Return individual with best fitness
    return max(tournament_individuals, key=lambda x: x[2] if x[2] is not None else -np.inf)


# >>> EVOTORCH INTEGRATION <<<
class RobotOlympicsProblem(Problem):
    """EvoTorch Problem definition for Robot Olympics optimization."""
    
    def __init__(self):
        # Calculate total parameter space size
        genotype_params = GENOTYPE_SIZE * 3  # 3 genotype arrays
        
        # Neural network parameters (fixed size)
        nn_params = (
            100 * 16 +      # w1: input to hidden
            16 * 16 +       # w2: hidden to hidden  
            16 * 50 +       # w3: hidden to output
            16 +            # b1: hidden bias
            16 +            # b2: hidden bias
            50              # b3: output bias
        )
        
        total_params = genotype_params + nn_params
        console.log(f"EvoTorch Problem: {total_params} parameters")
        
        # Define the fitness function that EvoTorch expects
        def fitness_function(solution_tensor):
            """Fitness function that EvoTorch calls for each solution."""
            try:
                return self._evaluate_single_solution(solution_tensor)
            except Exception as e:
                return -1000.0
        
        super().__init__(
            objective_sense="max",  # Maximize fitness
            solution_length=total_params,
            initial_bounds=(-2.0, 2.0),  # Wider initial bounds for CMA-ES exploration
            dtype=torch.float32,
            device="cpu",  # Use CPU for compatibility
            objective_func=fitness_function  # Provide the fitness function directly
        )
    
    def _evaluate_single_solution(self, solution: torch.Tensor) -> float:
        """Evaluate a single solution with robust error handling."""
        try:
            solution_np = solution.detach().cpu().numpy()
            
            # Validate solution
            if not np.isfinite(solution_np).all():
                return -1000.0
            
            # Split solution into genotype and neural network parameters
            genotype_params = GENOTYPE_SIZE * 3
            
            # Extract genotype
            genotype_flat = solution_np[:genotype_params]
            genotype = [
                genotype_flat[0:GENOTYPE_SIZE].astype(np.float32),
                genotype_flat[GENOTYPE_SIZE:2*GENOTYPE_SIZE].astype(np.float32), 
                genotype_flat[2*GENOTYPE_SIZE:3*GENOTYPE_SIZE].astype(np.float32)
            ]
            
            # Apply sigmoid to keep genotype in [0,1] range with better scaling for CMA-ES
            for i in range(len(genotype)):
                # Use tanh for smoother gradients and better CMA-ES performance
                genotype[i] = 0.5 * (np.tanh(genotype[i] * 0.5) + 1.0)
                # Ensure no NaN values
                if not np.isfinite(genotype[i]).all():
                    genotype[i] = RNG.random(genotype[i].shape).astype(np.float32)
            
            # Extract neural network parameters
            nn_params = solution_np[genotype_params:]
            
            # Create robot and neural network
            robot = create_robot_body(genotype, save_graph=False)
            if robot is None:
                return -1000.0
                
            nn = NN(robot)
            
            # Set neural network weights from parameters
            weights = self._params_to_weights(nn_params, nn)
            nn.set_controller_weights(weights)
            
            # Evaluate fitness with shorter duration for efficiency
            fitness = evaluate(robot, nn, plot_and_record=False, duration=min(SIMULATION_DURATION, 30))
            
            # Ensure fitness is finite
            if not np.isfinite(fitness):
                return -1000.0
                
            return float(fitness)
            
        except Exception as e:
            # Return penalty for any evaluation failure
            return -1000.0
    
    def _params_to_weights(self, params: np.ndarray, nn: NN) -> tuple:
        """Convert flat parameter array to neural network weights with bounds checking."""
        try:
            # Ensure parameters are finite
            params = np.nan_to_num(params, nan=0.0, posinf=1.0, neginf=-1.0)
            
            idx = 0
            
            # Extract weight matrices and biases with bounds checking
            w1_size = 100 * 16
            if idx + w1_size > len(params):
                # Not enough parameters, use random initialization
                return self._create_random_weights()
            w1 = params[idx:idx+w1_size].reshape(100, 16)
            idx += w1_size
            
            w2_size = 16 * 16  
            if idx + w2_size > len(params):
                return self._create_random_weights()
            w2 = params[idx:idx+w2_size].reshape(16, 16)
            idx += w2_size
            
            w3_size = 16 * 50
            if idx + w3_size > len(params):
                return self._create_random_weights()
            w3 = params[idx:idx+w3_size].reshape(16, 50)
            idx += w3_size
            
            if idx + 16 > len(params):
                return self._create_random_weights()
            b1 = params[idx:idx+16]
            idx += 16
            
            if idx + 16 > len(params):
                return self._create_random_weights()
            b2 = params[idx:idx+16] 
            idx += 16
            
            if idx + 50 > len(params):
                return self._create_random_weights()
            b3 = params[idx:idx+50]
            
            # Clip weights to reasonable bounds
            w1 = np.clip(w1, -5.0, 5.0)
            w2 = np.clip(w2, -5.0, 5.0)
            w3 = np.clip(w3, -5.0, 5.0)
            b1 = np.clip(b1, -2.0, 2.0)
            b2 = np.clip(b2, -2.0, 2.0)
            b3 = np.clip(b3, -2.0, 2.0)
            
            return (w1, w2, w3, b1, b2, b3)
            
        except Exception as e:
            return self._create_random_weights()
    
    def _create_random_weights(self) -> tuple:
        """Create random neural network weights as fallback."""
        w1 = RNG.normal(0.0, 0.1, size=(100, 16))
        w2 = RNG.normal(0.0, 0.1, size=(16, 16))
        w3 = RNG.normal(0.0, 0.1, size=(16, 50))
        b1 = RNG.normal(0.0, 0.1, size=16)
        b2 = RNG.normal(0.0, 0.1, size=16)
        b3 = RNG.normal(0.0, 0.1, size=50)
        return (w1, w2, w3, b1, b2, b3)


def run_evotorch_optimization():
    """Run CMA-ES based optimization via EvoTorch."""
    console.log("🚀 Starting CMA-ES Robot Olympics Optimization...")
    
    # Create the optimization problem
    problem = RobotOlympicsProblem()
    
    # Choose algorithm - CMA-ES is excellent for continuous optimization
    # CMA-ES (Covariance Matrix Adaptation Evolution Strategy) - State-of-the-art for continuous optimization
    algorithm = CMAES(
        problem,
        popsize=POPULATION_SIZE,
        stdev_init=0.3,  # Higher initial exploration for complex robot optimization
        # center_init will be automatically set to zeros by default
    )
    
    # Alternative algorithms you can try:
    
    # Option 2: SNES (Separable Natural Evolution Strategies) - Fast and effective
    # algorithm = SNES(
    #     problem,
    #     popsize=POPULATION_SIZE,
    #     stdev_init=0.1,
    # )
    
    # Option 3: CEM (Cross Entropy Method) - Good for high-dimensional problems
    # algorithm = CEM(
    #     problem,
    #     popsize=POPULATION_SIZE,
    #     parentsize=POPULATION_SIZE // 4,  # Elite size
    #     stdev_init=0.1,
    # )
    
    # Option 4: PGPE (Policy Gradients with Parameter Exploration)
    # algorithm = PGPE(
    #     problem,
    #     popsize=POPULATION_SIZE,
    #     stdev_init=0.1,
    # )
    
    # Option 5: Genetic Algorithm
    # algorithm = GeneticAlgorithm(
    #     problem,
    #     popsize=POPULATION_SIZE,
    #     elite_ratio=0.2,
    #     mutation_stdev=0.1,
    # )
    
    # Set up logging 
    try:
        logger = StdOutLogger(algorithm, interval=10)  # Print every 10 generations to reduce noise
        pandas_logger = PandasLogger(algorithm)  # For detailed analysis
    except Exception as e:
        console.log(f"Warning: Could not set up loggers: {e}")
        logger = None
        pandas_logger = None
    
    console.log(f"Algorithm: {type(algorithm).__name__}")
    console.log(f"Population size: {POPULATION_SIZE}")
    console.log(f"Generations: {MAX_GENERATIONS}")
    
    # Evolution loop with better error handling
    best_fitness_seen = -np.inf
    for generation in track(range(MAX_GENERATIONS), description="EvoTorch Evolution"):
        try:
            algorithm.step()
            
            # Get current best fitness with error handling
            if hasattr(algorithm, 'status') and algorithm.status is not None:
                current_best = algorithm.status.get('best_eval', None)
                if current_best is not None and np.isfinite(current_best):
                    best_fitness_seen = max(best_fitness_seen, float(current_best))
                    # Occasionally report progress
                    if generation % 20 == 0 or generation < 5:
                        pass  # Keep evolution quiet
                        # console.log(f"Generation {generation}: Best = {current_best:.3f}")
        except Exception as e:
            console.log(f"Error in generation {generation}: {e}")
            # Try to continue evolution
            continue
    
    # Get final results with error handling
    try:
        if hasattr(algorithm, 'status') and algorithm.status is not None:
            best_solution = algorithm.status.get('best', None)
            best_fitness = algorithm.status.get('best_eval', None)
            
            if best_solution is None or best_fitness is None:
                console.log("Warning: No valid best solution found, using best seen fitness")
                best_fitness = best_fitness_seen
                best_solution = None
        else:
            console.log("Warning: Algorithm status not available")
            best_fitness = best_fitness_seen
            best_solution = None
    except Exception as e:
        console.log(f"Error accessing final results: {e}")
        best_fitness = best_fitness_seen
        best_solution = None
    
    console.log(f"🏆 CMA-ES Best Fitness: {best_fitness:.3f}")
    
    # Evaluate and visualize best solution
    if best_solution is not None:
        console.log("Evaluating best solution with visualization...")
        problem_instance = RobotOlympicsProblem()
        
        try:
            # Convert best solution and evaluate with recording
            solution_np = best_solution.detach().cpu().numpy()
            genotype_params = GENOTYPE_SIZE * 3
            
            # Extract genotype
            genotype_flat = solution_np[:genotype_params]
            genotype = [
                genotype_flat[0:GENOTYPE_SIZE].astype(np.float32),
                genotype_flat[GENOTYPE_SIZE:2*GENOTYPE_SIZE].astype(np.float32), 
                genotype_flat[2*GENOTYPE_SIZE:3*GENOTYPE_SIZE].astype(np.float32)
            ]
            
            # Apply sigmoid to keep genotype in [0,1] range
            for i in range(len(genotype)):
                genotype[i] = 0.5 * (np.tanh(genotype[i] * 0.5) + 1.0)
            
            # Extract and set neural network parameters
            nn_params = solution_np[genotype_params:]
            robot = create_robot_body(genotype, save_graph=True)
            nn = NN(robot)
            weights = problem_instance._params_to_weights(nn_params, nn)
            nn.set_controller_weights(weights)
            
            # Final evaluation with recording
            final_fitness = evaluate(robot, nn, plot_and_record=True, duration=FINAL_EVAL_DURATION)
            console.log(f"📊 Final recorded fitness: {final_fitness:.3f}")
            
            # Save results
            results = {
                'algorithm': type(algorithm).__name__,
                'best_fitness': float(best_fitness),
                'final_fitness': float(final_fitness),
                'generations': MAX_GENERATIONS,
                'population_size': POPULATION_SIZE,
                'best_solution': solution_np.tolist(),
                'genotype': [arr.tolist() for arr in genotype]
            }
            
        except Exception as e:
            console.log(f"Error evaluating best solution: {e}")
            console.log("Creating a random solution for demonstration...")
            
            # Fallback: create a random solution
            test_genotype = [
                RNG.random(GENOTYPE_SIZE).astype(np.float32),
                RNG.random(GENOTYPE_SIZE).astype(np.float32),
                RNG.random(GENOTYPE_SIZE).astype(np.float32)
            ]
            robot = create_robot_body(test_genotype, save_graph=True)
            nn = NN(robot)
            nn.random_controller()
            final_fitness = evaluate(robot, nn, plot_and_record=True, duration=FINAL_EVAL_DURATION)
            
            results = {
                'algorithm': type(algorithm).__name__,
                'best_fitness': float(best_fitness),
                'final_fitness': float(final_fitness),
                'generations': MAX_GENERATIONS,
                'population_size': POPULATION_SIZE,
                'note': 'Fallback random solution due to evaluation error'
            }
    else:
        console.log("No best solution available, creating random solution for demonstration...")
        
        # Fallback: create a random solution
        test_genotype = [
            RNG.random(GENOTYPE_SIZE).astype(np.float32),
            RNG.random(GENOTYPE_SIZE).astype(np.float32), 
            RNG.random(GENOTYPE_SIZE).astype(np.float32)
        ]
        robot = create_robot_body(test_genotype, save_graph=True)
        nn = NN(robot)
        nn.random_controller()
        final_fitness = evaluate(robot, nn, plot_and_record=True, duration=FINAL_EVAL_DURATION)
        
        results = {
            'algorithm': type(algorithm).__name__,
            'best_fitness': float(best_fitness),
            'final_fitness': float(final_fitness),
            'generations': MAX_GENERATIONS,
            'population_size': POPULATION_SIZE,
            'note': 'Random solution due to no best solution available'
        }
    
    with open(DATA / "evotorch_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save pandas logger data for analysis
    if pandas_logger is not None and hasattr(pandas_logger, 'to_dataframe'):
        try:
            df = pandas_logger.to_dataframe()
            df.to_csv(DATA / "evotorch_history.csv", index=False)
            
            # Plot evolution history
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.plot(df['iter'], df['best'])
            plt.title('Best Fitness Evolution')
            plt.xlabel('Generation')
            plt.ylabel('Best Fitness')
            plt.grid(True)
            
            plt.subplot(2, 2, 2)
            plt.plot(df['iter'], df['mean'])
            plt.title('Mean Fitness Evolution')
            plt.xlabel('Generation') 
            plt.ylabel('Mean Fitness')
            plt.grid(True)
            
            if 'std' in df.columns:
                plt.subplot(2, 2, 3)
                plt.plot(df['iter'], df['std'])
                plt.title('Fitness Standard Deviation')
                plt.xlabel('Generation')
                plt.ylabel('Std Dev')
                plt.grid(True)
            
            plt.subplot(2, 2, 4)
            plt.plot(df['iter'], df['best'], label='Best')
            plt.plot(df['iter'], df['mean'], label='Mean')
            plt.title('Fitness Comparison')
            plt.xlabel('Generation')
            plt.ylabel('Fitness')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(DATA / "evotorch_evolution.png", dpi=300, bbox_inches='tight')
            
        except Exception as e:
            console.log(f"Warning: Could not save evolution history: {e}")
        
    console.log(f"📈 Results saved to {DATA / 'evotorch_results.json'}")
    console.log(f"📊 Evolution history saved to {DATA / 'evotorch_history.csv'}")
    
    return best_fitness


if __name__ == "__main__":
    # Choose optimization method
    USE_EVOTORCH = True  # Set to False to use original algorithm
    
    # Test basic functionality first
    console.log("🧪 Testing basic robot creation...")
    try:
        # Test random robot creation
        test_genotype = [
            RNG.random(GENOTYPE_SIZE).astype(np.float32),
            RNG.random(GENOTYPE_SIZE).astype(np.float32),
            RNG.random(GENOTYPE_SIZE).astype(np.float32)
        ]
        test_robot = create_robot_body(test_genotype, save_graph=False)
        test_nn = NN(test_robot)
        test_nn.random_controller()
        test_fitness = evaluate(test_robot, test_nn, plot_and_record=False, duration=10)
        console.log(f"✅ Basic test successful! Fitness: {test_fitness:.3f}")
    except Exception as e:
        console.log(f"❌ Basic test failed: {str(e)}")
        console.log("Falling back to original algorithm...")
        USE_EVOTORCH = False
    
    if USE_EVOTORCH:
        console.log("🤖 Starting CMA-ES Robot Olympics Evolution...")
        console.log(f"Configuration:")
        console.log(f"  Algorithm: CMA-ES (Covariance Matrix Adaptation)")
        console.log(f"  Population: {POPULATION_SIZE} individuals")
        console.log(f"  Generations: {MAX_GENERATIONS}")
        console.log(f"  Target: {TARGET_POSITION}")
        console.log(f"  Robot modules: {NUM_OF_MODULES}")
        console.log(f"  Simulation duration: {SIMULATION_DURATION}s")
        
        best_fitness = run_evotorch_optimization()
        console.log(f"🏆 CMA-ES Final Best Fitness: {best_fitness:.3f}")
    else:
        console.log("🤖 Starting Original Robot Olympics Evolution...")
        main()
    
    console.log("✅ Robot Olympics Evolution Complete!")
