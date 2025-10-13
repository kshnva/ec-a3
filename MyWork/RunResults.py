# =======================
# Replay & Visualize Best Robot from Checkpoint (deterministic)
# =======================

from pathlib import Path
import pickle
import numpy as np
import mujoco as mj
from mujoco import viewer
import sys
import random
import torch

# --- Patch for old checkpoints ---
# If the checkpoint references __main__.evaluate_body (from your evolution script),
# pickle.load will fail here unless we provide a placeholder.
def evaluate_body(*args, **kwargs):
    raise RuntimeError("evaluate_body from checkpoint is not usable in RunResults.")
sys.modules["__main__"].evaluate_body = evaluate_body

# Ariel imports (after the patch)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.renderers import single_frame_renderer, video_renderer

# =======================
# Config (tweakable)
# =======================
NUM_OF_MODULES = 30
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]
VISUALIZE_DURATION = 30

# Path to the checkpoint you produced previously
CHECKPOINT_FILE = Path(
    r"D:\Evolutionary Computing GitClone Ariel\ariel\MyWork\Nested_Evolution_With_Pause\checkpoint.pkl"
)

# Where to store the JSON & render outputs for this script
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(parents=True, exist_ok=True)

# Seed used to ensure deterministic decoding + simulation reproducibility
SEED = 42

# =======================
# Helpers
# =======================
def to_numpy(x):
    """Convert many possible numeric container types into a numpy float32 array."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    # evotorch ReadOnlyTensor or other objects: try to coerce via np.array
    try:
        import torch as _torch
        if isinstance(x, _torch.Tensor):
            return x.detach().cpu().astype(_torch.float32).numpy()
    except Exception:
        pass
    try:
        return np.array(x, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Cannot convert object to numpy: {type(x)} : {e}")

def set_global_seed(seed: int):
    """Seed common RNGs so decoding + construction are reproducible."""
    np.random.seed(seed)
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# =======================
# Fitness
# =======================
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    return -np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)

# =======================
# Default deterministic controller (used if no best_cpg stored)
# =======================
def nn_controller(model: mj.MjModel, data: mj.MjData):
    """Deterministic oscillatory controller (no random state)."""
    num_actuators = model.nu
    t = data.time
    # uses only numpy operations and t => deterministic given same simulation
    return np.sin(2.0 * np.pi * 0.5 * t + np.arange(num_actuators))

# =======================
# Experiment runner
# =======================
def experiment(robot, controller, duration=15, mode="launcher"):
    mj.set_mjcb_control(None)
    world = OlympicArena()
    world.spawn(robot.spec, spawn_position=np.array(SPAWN_POS))

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)

    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))

    if mode == "simple":
        simple_runner(model, data, duration=duration)
    elif mode == "frame":
        save_path = str(DATA / "robot.png")
        single_frame_renderer(model, data, save=True, save_path=save_path)
    elif mode == "video":
        video_recorder = VideoRecorder(output_folder=str(DATA / "videos"))
        video_renderer(model, data, duration=duration, video_recorder=video_recorder)
    elif mode == "launcher":
        viewer.launch(model=model, data=data)

# =======================
# Main
# =======================
def main():
    if not CHECKPOINT_FILE.exists():
        print(f"[ERROR] No checkpoint found at: {CHECKPOINT_FILE}")
        return

    # enforce deterministic seeds BEFORE decoding the genotype
    set_global_seed(SEED)

    with open(CHECKPOINT_FILE, "rb") as f:
        checkpoint = pickle.load(f)

    # 1) Get best_body from checkpoint, or try to extract from a pickled searcher
    best_body = checkpoint.get("best_body", None)
    best_cpg = checkpoint.get("best_cpg", None)

    if best_body is None and "searcher_body" in checkpoint:
        # Try to extract the best genome from the stored CMAES/searcher object
        searcher_body = checkpoint["searcher_body"]
        try:
            sb_best = searcher_body.status["best"].values
            best_body = to_numpy(sb_best)
            print("[INFO] Extracted best_body from stored searcher_body object.")
        except Exception as e:
            print("[WARN] Could not extract best_body from searcher_body:", e)
            best_body = None

    # Final conversion to numpy
    best_body = to_numpy(best_body)
    best_cpg = to_numpy(best_cpg)

    if best_body is None:
        print("[ERROR] No best body genome found in checkpoint.")
        return

    print("[INFO] Loaded best body from checkpoint (shape: {})".format(best_body.shape))

    # decode genotype -> probability matrices -> robot graph
    nde_genotype = [
        best_body[:64],
        best_body[64:128],
        best_body[128:],
    ]

    # Set seeds again right before decoding (some internals may use global RNG)
    set_global_seed(SEED)

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
    p_matrices = nde.forward(nde_genotype)
    hpd = HighProbabilityDecoder(NUM_OF_MODULES)
    robot_graph = hpd.probability_matrices_to_graph(*p_matrices)

    # Save JSON graph for sharing / replay
    json_path = DATA / "best_robot_graph.json"
    try:
        save_graph_as_json(robot_graph, json_path)
        print(f"[INFO] Saved robot graph JSON to {json_path}")
    except Exception as e:
        # save_graph_as_json might not be present depending on ariel version; ignore gracefully
        print(f"[WARN] Could not save robot graph JSON: {e}")

    # Build MuJoCo robot spec
    robot = construct_mjspec_from_graph(robot_graph)

    # Make tracker and controller
    tracker = Tracker(mj.mjtObj.mjOBJ_GEOM, "core")

    if best_cpg is not None:
        class BestCPGController(Controller):
            def __init__(self, cpg_params, tracker):
                super().__init__(controller_callback_function=None, tracker=tracker)
                self.cpg_params = cpg_params

            def set_control(self, model, data, *args, **kwargs):
                num_actuators = model.nu
                t = data.time
                freqs = self.cpg_params[:num_actuators]
                amps = self.cpg_params[num_actuators:2*num_actuators]
                phases = self.cpg_params[2*num_actuators:3*num_actuators]
                data.ctrl[:] = amps * np.sin(freqs * t + phases)
                if self.tracker is not None:
                    self.tracker.update(data)

        ctrl = BestCPGController(best_cpg, tracker)
        print("[INFO] Using saved best CPG controller.")
    else:
        ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)


    # Run visualization/simulation (deterministic because of fixed seeds)
    experiment(robot=robot, controller=ctrl, duration=VISUALIZE_DURATION, mode="launcher")

    # Report fitness
    try:
        history = tracker.history["xpos"][0]
        fitness = fitness_function(history)
        print(f"[RESULT] Fitness of best robot: {fitness:.6f}")
    except Exception as e:
        print("[WARN] Could not compute fitness from tracker:", e)

if __name__ == "__main__":
    main()
