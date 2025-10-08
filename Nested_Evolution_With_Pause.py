# =======================
# Nested EA + CPG + Body Evolution with Checkpointing
# =======================

# Standard library
from pathlib import Path
from typing import Literal
import copy
import pickle

# Third-party
import numpy as np
import torch
from evotorch import Problem
from evotorch.algorithms import CMAES
import mujoco as mj
from mujoco import viewer
from networkx import DiGraph

# Ariel imports
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
SEED = 42
RNG = np.random.default_rng(SEED)

NUM_OF_MODULES = 30
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]

BODY_GENOTYPE_SIZE = 64 * 3
POP_SIZE_BODY = 20
POP_SIZE_CPG = 10
GENERATIONS_BODY = 1          # outer loop generations
GENERATIONS_CPG_INNER = 15     # inner loop generations
INITIAL_BODY_RANGE = 1.0
STDEV_INIT = 0.7

CPG_BOUNDS = (-5.0, 5.0)

INNER_DURATION = 10
OUTER_DURATION = 30
VISUALIZE_DURATION = 30
FILTER_DURATION = 3
DISPLACEMENT_THRESHOLD = 0.07

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

CHECKPOINT_FILE = DATA / "checkpoint.pkl"

ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# =======================
# Fitness function
# =======================
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    return -np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)

# =======================
# CPG Controller
# =======================
def cpg_controller(model: mj.MjModel, data: mj.MjData, cpg_params: np.ndarray) -> np.ndarray:
    num_actuators = model.nu
    t = data.time
    freqs = cpg_params[:num_actuators]
    amps = cpg_params[num_actuators:2*num_actuators]
    phases = cpg_params[2*num_actuators:3*num_actuators]
    return amps * np.sin(freqs * t + phases)

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
    fresh_core = construct_mjspec_from_graph(robot_graph)
    world.spawn(fresh_core.spec, spawn_position=np.array(SPAWN_POS))

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    data.qpos[:] = 0
    data.qpos[2] = SPAWN_POS[2]
    data.qvel[:] = 0
    data.qacc[:] = 0
    mj.mj_forward(model, data)

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
        from ariel.utils.renderers import video_renderer
        video_recorder = VideoRecorder(output_folder=str(DATA / "videos"))
        video_renderer(model, data, duration=duration, video_recorder=video_recorder)
    elif mode == "launcher":
        viewer.launch(model=model, data=data)

    return fitness_function(tracker.history["xpos"][0])

# =======================
# Non-learner filter
# =======================
def is_learner(robot_graph: DiGraph) -> bool:
    mj.set_mjcb_control(None)
    world = OlympicArena()
    fresh_core = construct_mjspec_from_graph(robot_graph)
    world.spawn(fresh_core.spec, spawn_position=np.array(SPAWN_POS))

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    data.qpos[:] = 0
    data.qpos[2] = SPAWN_POS[2]
    data.qvel[:] = 0
    data.qacc[:] = 0
    mj.mj_forward(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    num_actuators = model.nu
    steps = max(1, int(FILTER_DURATION / 0.01))
    for _ in range(steps):
        data.ctrl[:] = RNG.uniform(-1, 1, size=num_actuators)
        mj.mj_step(model, data)
        tracker.update(data)

    start = tracker.history["xpos"][0][0]
    end = tracker.history["xpos"][0][-1]
    displacement_xy = np.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)

    if displacement_xy < DISPLACEMENT_THRESHOLD:
        print(f"[FILTER] Non-learner detected (XY displacement={displacement_xy:.4f}), skipping")
        return False
    return True

# =======================
# Body + CPG evaluation
# =======================
def evaluate_body(body_vector: torch.Tensor) -> float:
    body_vector_np = np.array(body_vector, dtype=np.float32)
    genotype = [body_vector_np[:64], body_vector_np[64:128], body_vector_np[128:]]

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph: DiGraph = hpd.probability_matrices_to_graph(*p_matrices)

    if not is_learner(robot_graph):
        return -1e6

    num_actuators = NUM_OF_MODULES
    cpg_genotype_size = num_actuators * 3

    def evaluate_cpg(cpg_tensor: torch.Tensor) -> float:
        return experiment(robot_graph, np.array(cpg_tensor, dtype=np.float32), duration=INNER_DURATION, mode="simple")

    problem_cpg = Problem(
        "max",
        evaluate_cpg,
        solution_length=cpg_genotype_size,
        dtype=torch.float32,
        initial_bounds=CPG_BOUNDS,
    )
    searcher_cpg = CMAES(problem_cpg, popsize=POP_SIZE_CPG, stdev_init=STDEV_INIT)
    for _ in range(GENERATIONS_CPG_INNER):
        searcher_cpg.step()

    return searcher_cpg.status["best_eval"]

# =======================
# Checkpointing
# =======================
def save_checkpoint(gen, searcher_body, fitness_history_body, best_body=None, best_cpg=None):
    checkpoint = {
        "gen": gen,
        "searcher_body": searcher_body,
        "fitness_history": fitness_history_body,
        "best_body": best_body,
        "best_cpg": best_cpg
    }
    with open(CHECKPOINT_FILE, "wb") as f:
        pickle.dump(checkpoint, f)
    print(f"[Checkpoint] Saved generation {gen}")

def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, "rb") as f:
            checkpoint = pickle.load(f)
        print(f"[Checkpoint] Loaded generation {checkpoint['gen']}")
        return checkpoint
    else:
        return None

# =======================
# Main
# =======================
def main():
    checkpoint = load_checkpoint()
    if checkpoint:
        start_gen = checkpoint["gen"]
        searcher_body = checkpoint["searcher_body"]
        fitness_history_body = checkpoint["fitness_history"]
        best_body = checkpoint.get("best_body", None)
        best_cpg = checkpoint.get("best_cpg", None)
    else:
        start_gen = 0
        fitness_history_body = []
        best_body = None
        best_cpg = None

        problem_body = Problem(
            "max",
            evaluate_body,
            solution_length=BODY_GENOTYPE_SIZE,
            dtype=torch.float32,
            initial_bounds=(-INITIAL_BODY_RANGE, INITIAL_BODY_RANGE),
        )
        searcher_body = CMAES(problem_body, popsize=POP_SIZE_BODY, stdev_init=STDEV_INIT)

    for gen in range(start_gen, GENERATIONS_BODY):
        searcher_body.step()
        best_fit = searcher_body.status["best_eval"]
        fitness_history_body.append(best_fit)
        print(f"Body Gen {gen} | Best fitness: {best_fit:.4f}")

        # Save intermediate checkpoint
        save_checkpoint(gen + 1, searcher_body, fitness_history_body, best_body, best_cpg)

    # Save best body genome for final visualization
    best_body = np.array(searcher_body.status["best"].values, dtype=np.float32)
    save_checkpoint(GENERATIONS_BODY, searcher_body, fitness_history_body, best_body, best_cpg)

    print("\nBest body saved. To visualize, use the 'visualize_best.py' script with best_body.pkl")

if __name__ == "__main__":
    main()
