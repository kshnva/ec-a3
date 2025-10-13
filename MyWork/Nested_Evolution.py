# =======================
# Nested EA + CPG + Body Evolution for Arbitrary Robot Body
# =======================

# Standard library
from pathlib import Path
from typing import Literal
import copy
import time

# Third-party
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES
import mujoco as mj
from mujoco import viewer
from networkx import DiGraph

# Ariel imports
import os
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import save_graph_as_json
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder


# =======================
# Configuration
# =======================
SEED = 43
RNG = np.random.default_rng(SEED)

# Robot setup
NUM_OF_MODULES = 30
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]

# Evolution parameters
BODY_GENOTYPE_SIZE = 64 * 3
POP_SIZE_BODY = 504     # CMA-ES population for body evolution
POP_SIZE_CPG = 15       # CMA-ES population for CPG evolution
GENERATIONS_BODY = 1          # outer loop generations
GENERATIONS_CPG_INNER = 20     # inner loop generations
INITIAL_BODY_RANGE = 1.0
STDEV_INIT = 0.7

# CPG bounds (freqs, amps, phases)
CPG_BOUNDS = (-5.0, 5.0)

# Durations
INNER_DURATION = 15       # during fitness eval
OUTER_DURATION = 20       # during final body evolution eval
VISUALIZE_DURATION = 15   # when launching viewer
FILTER_DURATION = 3       # short random test for "learner" filtering

# Threshold for learner check (XY-plane)
DISPLACEMENT_THRESHOLD = 0.15  # meters
Max_Bricks = 4

# Paths
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

json_filename = (
    f"best_robot_body_P{POP_SIZE_BODY}_C{POP_SIZE_CPG}"
    f"_G{GENERATIONS_BODY}_CG{GENERATIONS_CPG_INNER}.json"
)
custom_json_path = Path(r"D:\Evolutionary Computing GitClone Ariel\ariel\MyWork\Nested_Evolution") / json_filename
custom_json_path.parent.mkdir(parents=True, exist_ok=True)

# Type Aliases
ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]


# =======================
# Fitness function
# =======================
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    cartesian_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    return -cartesian_distance


# =======================
# CPG Controller
# =======================
def cpg_controller(model: mj.MjModel, data: mj.MjData, cpg_params: np.ndarray) -> np.ndarray:
    num_actuators = model.nu
    t = data.time
    freqs = cpg_params[:num_actuators]
    amps = cpg_params[num_actuators:2*num_actuators]
    phases = cpg_params[2*num_actuators:3*num_actuators]
    outputs = amps * np.sin(freqs * t + phases)
    return outputs


class CPGController(Controller):
    def __init__(self, cpg_params, tracker=None):
        super().__init__(controller_callback_function=None, tracker=tracker)
        self.cpg_params = cpg_params

    def set_control(self, model, data, *args, **kwargs):
        data.ctrl[:] = cpg_controller(model, data, self.cpg_params)
        if self.tracker is not None:
            self.tracker.update(data)


# =======================
# Simulation / Experiment
# =======================
def experiment(robot_graph: DiGraph, cpg_params: np.ndarray, duration: int = 15, mode: ViewerTypes = "simple") -> float:
    mj.set_mjcb_control(None)
    world = OlympicArena()

    # Build robot from graph
    fresh_core = construct_mjspec_from_graph(robot_graph)
    spawn_pos = np.array(SPAWN_POS)
    world.spawn(fresh_core.spec, position=spawn_pos)

    model = world.spec.compile()
    data = mj.MjData(model)

    mj.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mj.mj_forward(model, data)

    # Tracker + Controller
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)
    ctrl = CPGController(cpg_params=cpg_params, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

    # Run simulation
    if mode == "simple":
        simple_runner(model, data, duration=duration)
    elif mode == "frame":
        from ariel.utils.renderers import single_frame_renderer
        save_path = str(DATA / "robot.png")
        single_frame_renderer(model, data, save=True, save_path=save_path)
    elif mode == "video":
        path_to_video_folder = str(DATA / "videos")
        video_recorder = VideoRecorder(output_folder=path_to_video_folder)
        from ariel.utils.renderers import video_renderer
        video_renderer(model, data, duration=duration, video_recorder=video_recorder)
    elif mode == "launcher":
        viewer.launch(model=model, data=data)

    return fitness_function(tracker.history["xpos"][0])


# =======================
# Non-learner Filter (XY displacement)
# =======================
def is_learner(robot_graph: DiGraph) -> bool:
    mj.set_mjcb_control(None)
    world = OlympicArena()

    fresh_core = construct_mjspec_from_graph(robot_graph)
    spawn_pos = np.array(SPAWN_POS)
    world.spawn(fresh_core.spec, position=spawn_pos)

    model = world.spec.compile()
    data = mj.MjData(model)

    mj.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mj.mj_forward(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    num_actuators = model.nu
    steps = max(1, int(FILTER_DURATION / 0.01))
    for step in range(steps):
        data.ctrl[:] = RNG.uniform(-1, 1, size=num_actuators)
        mj.mj_step(model, data)
        tracker.update(data)

    start = tracker.history["xpos"][0][0]
    end = tracker.history["xpos"][0][-1]
    x0, y0, _ = start
    x1, y1, _ = end
    displacement_xy = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)

    if displacement_xy < DISPLACEMENT_THRESHOLD:
        print(f"[FILTER] Non-learner detected (XY displacement={displacement_xy:.4f} < {DISPLACEMENT_THRESHOLD}), skipping CPG evolution")
        return False

    return True


# =======================
# Body + CPG Evaluation
# =======================
def evaluate_body(body_vector: torch.Tensor) -> float:
    start_body = time.time()

    # Convert vector -> genotype
    body_vector_np = np.array(body_vector, dtype=np.float32)
    type_p_genes = body_vector_np[:64]
    conn_p_genes = body_vector_np[64:128]
    rot_p_genes = body_vector_np[128:]
    genotype = [type_p_genes, conn_p_genes, rot_p_genes]

    # Decode to robot graph
    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )

    # reject robots with too many BRICK modules
    num_bricks = sum(1 for _, data in robot_graph.nodes(data=True) if data.get("type") == "BRICK")
    if num_bricks > Max_Bricks:
        print(f"[FILTER] Robot has {num_bricks} BRICK modules, skipping...")
        return -1e6

    if not is_learner(robot_graph):
        print("[INFO] Skipping non-learner body")
        return -1e6

    fresh_core = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()
    spawn_pos = np.array(SPAWN_POS)
    world.spawn(fresh_core.spec, position=spawn_pos)
    model = world.spec.compile()
    num_actuators = model.nu
    cpg_genotype_size = num_actuators * 3

    def evaluate_cpg(cpg_tensor: torch.Tensor) -> float:
        cpg_params = np.array(cpg_tensor, dtype=np.float32)
        return experiment(robot_graph, cpg_params, duration=INNER_DURATION, mode="simple")

    problem_cpg = Problem(
        "max",
        evaluate_cpg,
        solution_length=cpg_genotype_size,
        dtype=torch.float32,
        initial_bounds=CPG_BOUNDS,
    )
    searcher_cpg = CMAES(problem_cpg, popsize=POP_SIZE_CPG, stdev_init=STDEV_INIT)

    for inner_gen in range(GENERATIONS_CPG_INNER):
        inner_start = time.time()
        searcher_cpg.step()
        elapsed_inner = time.time() - inner_start
        best_inner_fit = searcher_cpg.status["best_eval"]
        print(f"   [CPG Gen {inner_gen+1}/{GENERATIONS_CPG_INNER}] Best fitness: {best_inner_fit:.4f} | Runtime: {elapsed_inner:.2f}s")

    body_elapsed = time.time() - start_body
    print(f"   [Body Eval Done] Runtime for this body: {body_elapsed:.2f}s")

    return searcher_cpg.status["best_eval"]


# =======================
# Main
# =======================
def main():
    total_start = time.time()

    problem_body = Problem(
        "max",
        evaluate_body,
        solution_length=BODY_GENOTYPE_SIZE,
        dtype=torch.float32,
        initial_bounds=(-INITIAL_BODY_RANGE, INITIAL_BODY_RANGE),
    )
    searcher_body = CMAES(problem_body, popsize=POP_SIZE_BODY, stdev_init=STDEV_INIT)
    fitness_history_body = []

    # Track the best valid robot
    global_best_fit = -np.inf
    global_best_vector = None

    for gen in range(GENERATIONS_BODY):
        gen_start = time.time()
        print(f"\n[Body Evolution] Generation {gen+1}/{GENERATIONS_BODY}")

        searcher_body.step()

        # Check each candidate in the current CMA-ES population
        for candidate in searcher_body.population:
            fitness = evaluate_body(candidate)
            if fitness > global_best_fit:
                global_best_fit = fitness
                global_best_vector = candidate

        fitness_history_body.append(global_best_fit)

        gen_elapsed = time.time() - gen_start
        print(f"   [Body Gen {gen+1}] Best valid fitness so far: {global_best_fit:.4f} | Runtime: {gen_elapsed:.2f}s")

    # Save fitness history
    fitness_file_path = custom_json_path.with_suffix(".fitness.txt")
    np.savetxt(fitness_file_path, fitness_history_body)
    print(f"\nFinal best valid fitness: {global_best_fit:.4f}")

    # Decode the best valid body
    best_body_vector = np.array(global_best_vector, dtype=np.float32)
    type_p_genes = best_body_vector[:64]
    conn_p_genes = best_body_vector[64:128]
    rot_p_genes = best_body_vector[128:]
    genotype = [type_p_genes, conn_p_genes, rot_p_genes]

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )

    try:
        save_graph_as_json(robot_graph, custom_json_path)
        print(f"[INFO] Saved best robot JSON to {custom_json_path}")
    except Exception as e:
        print(f"[WARN] Could not save robot graph JSON: {e}")

    # Construct and spawn the best robot
    fresh_core = construct_mjspec_from_graph(robot_graph)
    world = OlympicArena()
    spawn_pos = np.array(SPAWN_POS)
    world.spawn(fresh_core.spec, position=spawn_pos)
    model = world.spec.compile()
    num_actuators = model.nu
    cpg_genotype_size = num_actuators * 3

    # Evolve CPG for the best robot
    def evaluate_cpg(cpg_tensor: torch.Tensor) -> float:
        cpg_params = np.array(cpg_tensor, dtype=np.float32)
        return experiment(robot_graph, cpg_params, duration=OUTER_DURATION, mode="simple")

    problem_cpg = Problem(
        "max",
        evaluate_cpg,
        solution_length=cpg_genotype_size,
        dtype=torch.float32,
        initial_bounds=CPG_BOUNDS,
    )
    searcher_cpg = CMAES(problem_cpg, popsize=POP_SIZE_CPG, stdev_init=STDEV_INIT)
    for inner_gen in range(GENERATIONS_CPG_INNER):
        inner_start = time.time()
        searcher_cpg.step()
        elapsed_inner = time.time() - inner_start
        best_inner_fit = searcher_cpg.status["best_eval"]
        print(f"   [Final CPG Gen {inner_gen+1}/{GENERATIONS_CPG_INNER}] Best fitness: {best_inner_fit:.4f} | Runtime: {elapsed_inner:.2f}s")

    best_cpg = np.array(searcher_cpg.status["best"].values, dtype=np.float32)

    print("\nLaunching MuJoCo visualizer for best robot...")
    experiment(robot_graph, best_cpg, duration=VISUALIZE_DURATION, mode="launcher")

    total_elapsed = time.time() - total_start
    print(f"\n[INFO] Total runtime: {total_elapsed:.2f}s")



if __name__ == "__main__":
    main()
