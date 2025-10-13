# =======================
# EA + CPG Evolution for Arbitrary Robot Body
# =======================

# Standard library
from pathlib import Path
from typing import Literal
import copy

# Third-party
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES
import matplotlib.pyplot as plt
import mujoco as mj
from mujoco import viewer

# Ariel imports
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from networkx import DiGraph

# Type Aliases
ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# =======================
# Random generator
# =======================
SEED = 42
RNG = np.random.default_rng(SEED)

# =======================
# Paths and Data
# =======================
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

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
    
    # Build a fresh spec from graph
    fresh_core = construct_mjspec_from_graph(robot_graph)
    
    # Spawn at fixed height z=0.5
    spawn_pos = np.array([SPAWN_POS[0], SPAWN_POS[1], SPAWN_POS[2]])
    world.spawn(fresh_core.spec, spawn_position=spawn_pos)
    
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    # Zero positions (except z), velocities, and accelerations
    data.qpos[:] = 0
    data.qpos[2] = spawn_pos[2]
    data.qvel[:] = 0
    data.qacc[:] = 0
    mj.mj_forward(model, data)

    # Tracker and controller
    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)
    
    ctrl = CPGController(cpg_params=cpg_params, tracker=tracker)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    
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
    
    fitness = fitness_function(tracker.history["xpos"][0])
    print(f"[INFO] Fitness: {fitness:.4f}")
    return fitness

# =======================
# Main
# =======================
def main():
    # === Body encoding
    genotype_size = 64
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    genotype = [type_p_genes, conn_p_genes, rot_p_genes]

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)

    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph = hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2])

    # === CPG CMA-ES setup ===
    num_actuators = NUM_OF_MODULES
    cpg_genotype_size = num_actuators * 3
    INITIAL_WEIGHT_RANGE = 1.0
    POPULATION_SIZE = 6
    GENERATIONS = 100

    def evaluate(cpg_params_tensor: torch.Tensor) -> float:
        cpg_params = cpg_params_tensor.numpy().astype(np.float32)
        return experiment(robot_graph, cpg_params, duration=10, mode="simple")

    problem = Problem(
        "max",
        evaluate,
        solution_length=cpg_genotype_size,
        dtype=torch.float32,
        initial_bounds=(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE),
    )

    searcher = CMAES(
        problem,
        popsize=POPULATION_SIZE,
        stdev_init=0.5,
    )

    fitness_history = []
    for gen in range(GENERATIONS):
        searcher.step()
        best_fit = searcher.status["best_eval"]
        fitness_history.append(best_fit)
        print(f"Gen {gen} | Best fitness: {best_fit:.4f}")

    best_cpg = searcher.status["best"].values
    print("\nBest CPG genome:", best_cpg)

    # =======================
    # Plot fitness
    # =======================
    plt.figure(figsize=(8,5))
    plt.plot(fitness_history, marker='o', linestyle='-', color='b')
    plt.xlabel("Generation")
    plt.ylabel("Best Fitness")
    plt.title("CPG Evolution Progress")
    plt.grid(True)
    plt.show()

    # =======================
    # Launch MuJoCo visualizer for best genome
    # =======================
    print("Launching MuJoCo visualizer for the best CPG genome...")
    experiment(robot_graph, best_cpg, duration=30, mode="launcher")


if __name__ == "__main__":
    main()
