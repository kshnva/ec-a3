"""Robot Olympics Competition template code (minimally modified for Step_Curry)."""

# Standard library
from pathlib import Path
from typing import Any
import json

import mujoco as mj
import numpy as np

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    load_graph_from_json,
)
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import video_renderer
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
TARGET_POSITION = [5, 0, 0.5]
TIME_OUT = 500

# Local scripts
#from A3_plot_function import show_xpos_history

# --- IMPORT YOUR ELEMENTS --- #
from pathlib import Path
import numpy as np

ROBOT_JSON = Path("Robot Result/Step_Curry.json")
GENOME_JSON = Path("Robot Result/Step_Curry_Brain.json")

# --- NN CONTROLLER SETUP --- #
HIDDEN_SIZE = 16
MAX_VELOCITY = 0.05
MAX_ANGLE = np.pi / 2

def tanh(x): return np.tanh(x)
def normalize_qpos(qpos): return np.clip(qpos / np.pi, -1.0, 1.0)
def normalize_xpos(xpos_flat): return np.clip(xpos_flat / 10.0, -1.0, 1.0)
def normalize_xmat(xmat_flat): return np.clip(xmat_flat, -1.0, 1.0)

def assemble_inputs_from_data(data, model):
    NQ = model.nq
    qpos = np.array(data.qpos[:NQ], dtype=np.float64)
    qpos_norm = normalize_qpos(qpos)
    xpos = np.array(data.xpos[0], dtype=np.float64)
    xmat = np.array(data.xmat[0].reshape(-1), dtype=np.float64)
    features = np.concatenate([qpos_norm, normalize_xpos(xpos), normalize_xmat(xmat)]).astype(np.float32)
    return features

def velocity_to_target(prev_ctrl, raw_outputs):
    velocity = raw_outputs * MAX_VELOCITY
    new_ctrl = prev_ctrl + velocity
    return np.clip(new_ctrl, -MAX_ANGLE, MAX_ANGLE)

def decode_genome(genome, input_size, output_size):
    g = np.array(genome, dtype=np.float64)
    idx = 0
    W_in = g[idx: idx + input_size * HIDDEN_SIZE].reshape(input_size, HIDDEN_SIZE)
    idx += input_size * HIDDEN_SIZE
    W_rec = g[idx: idx + HIDDEN_SIZE * HIDDEN_SIZE].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
    idx += HIDDEN_SIZE * HIDDEN_SIZE
    W_out = g[idx: idx + HIDDEN_SIZE * output_size].reshape(HIDDEN_SIZE, output_size)
    return W_in, W_rec, W_out

# --- DEFINE YOUR CONTROLLER FUNCTION --- #
def YOUR_CONTROLLER(model, data, genome=None, hidden=None, ctrl_state=None):
    """Neural controller from Step_Curry genome."""
    if not hasattr(YOUR_CONTROLLER, "initialized"):
        # one-time setup
        with open(GENOME_JSON, "r") as f:
            genome_data = json.load(f)
        genome = np.array(genome_data["genome"], dtype=np.float64)

        NQ = model.nq
        NU = model.nu
        INPUT_SIZE = NQ + 3 + 9 + 2  # qpos + xpos + xmat + time features
        OUTPUT_SIZE = NU
        W_in, W_rec, W_out = decode_genome(genome, INPUT_SIZE, OUTPUT_SIZE)
        YOUR_CONTROLLER.W_in, YOUR_CONTROLLER.W_rec, YOUR_CONTROLLER.W_out = W_in, W_rec, W_out
        YOUR_CONTROLLER.h = np.zeros(HIDDEN_SIZE, dtype=np.float64)
        YOUR_CONTROLLER.ctrl_state = np.zeros(NU, dtype=np.float64)
        YOUR_CONTROLLER.initialized = True

    # recurrent NN step
    W_in, W_rec, W_out = YOUR_CONTROLLER.W_in, YOUR_CONTROLLER.W_rec, YOUR_CONTROLLER.W_out
    h, ctrl_state = YOUR_CONTROLLER.h, YOUR_CONTROLLER.ctrl_state

    base_feats = assemble_inputs_from_data(data, model)
    t = getattr(data, "time", 0.0)
    phase = 2 * np.pi * (t / 5.0)
    time_feats = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
    inputs = np.concatenate([base_feats, time_feats]).astype(np.float64)

    h = tanh(np.dot(inputs, W_in) + np.dot(h, W_rec))
    raw_outputs = tanh(np.dot(h, W_out))
    ctrl_state = velocity_to_target(ctrl_state, raw_outputs)

    # apply control
    YOUR_CONTROLLER.h = h
    YOUR_CONTROLLER.ctrl_state = ctrl_state

    # Return the control output to Ariel’s Controller wrapper
    return ctrl_state


# --- FITNESS FUNCTION --- #
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    return -np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)

# --- EXPERIMENT FUNCTION (unchanged except robot/controller injection) --- #
def experiment(robot: Any, controller: Controller, duration: int = TIME_OUT) -> None:
    mj.set_mjcb_control(None)
    world = OlympicArena(load_precompiled=False)
    world.spawn(robot.spec, position=SPAWN_POS, correct_collision_with_floor=True)
    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    controller.tracker.setup(world.spec, data)

    mj.set_mjcb_control(lambda m, d: controller.set_control(m, d))
    path_to_video_folder = str(DATA / "videos")
    video_recorder = VideoRecorder(output_folder=path_to_video_folder)
    video_renderer(model, data, duration=duration, video_recorder=video_recorder)

# --- MAIN --- #
def main() -> None:
    path_to_graph = ROBOT_JSON
    robot_graph = load_graph_from_json(path_to_graph)
    core = construct_mjspec_from_graph(robot_graph)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

    ctrl = Controller(controller_callback_function=YOUR_CONTROLLER, tracker=tracker)
    experiment(robot=core, controller=ctrl)

    #show_xpos_history(tracker.history["xpos"][0])
    fitness = fitness_function(tracker.history["xpos"][0])
    console.log(f"Fitness of generated robot: {fitness}")

if __name__ == "__main__":
    main()
