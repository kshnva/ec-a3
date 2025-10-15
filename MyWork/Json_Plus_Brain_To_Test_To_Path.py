import numpy as np
import torch
import json
from pathlib import Path
import mujoco
from mujoco import mj_step, viewer

# ariel imports
from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces
import networkx as nx

# -----------------------
# USER SETTINGS
# -----------------------
ROBOT_JSON = Path("Robot Result/Step_Curry.json")
GENOME_JSON = Path("Robot Result/Step_Curry_Brain.json")
SPAWN_POSITION = [-0.8, 0, 0.1]
HIDDEN_SIZE = 16
MAX_VELOCITY = 0.05
MAX_ANGLE = np.pi / 2
TARGET_POSITION = np.array([5.0, 0.0, 0.5])  # target for fitness
PATH_SAVE_FILE = Path("MyWork/Nested_Evolution/robot_path1.npy")

# -----------------------
# Helper functions
# -----------------------
def load_graph_from_json(path: Path) -> nx.DiGraph:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    graph = nx.DiGraph()
    for node in data["nodes"]:
        graph.add_node(node["id"], type=node["type"], rotation=node.get("rotation", "DEG_0"))
    for edge in data["edges"]:
        graph.add_edge(edge["source"], edge["target"], face=edge["face"])
    return graph

def construct_core_from_graph(graph: nx.DiGraph):
    modules = {}
    for node_id, attrs in graph.nodes(data=True):
        node_type = attrs["type"]
        rotation = attrs.get("rotation", "DEG_0")
        if node_type == "CORE":
            modules[node_id] = CoreModule(index=node_id)
        elif node_type == "BRICK":
            modules[node_id] = BrickModule(index=node_id)
        elif node_type == "HINGE":
            modules[node_id] = HingeModule(index=node_id)
        else:
            modules[node_id] = None

        if node_type in ["HINGE", "BRICK"] and modules[node_id] is not None:
            try:
                deg = int(rotation.replace("DEG_", ""))
                modules[node_id].rotate(deg)
            except Exception:
                pass

    # attach modules
    for src, tgt, edge_attrs in graph.edges(data=True):
        face = edge_attrs["face"]
        src_module = modules.get(src)
        tgt_module = modules.get(tgt)
        if src_module is None or tgt_module is None:
            continue
        try:
            src_module.sites[getattr(ModuleFaces, face)].attach_body(
                body=tgt_module.body,
                prefix=f"{tgt_module.__class__.__name__.lower()}_{tgt}"
            )
        except Exception:
            pass

    core_nodes = [nid for nid, a in graph.nodes(data=True) if a["type"] == "CORE"]
    if len(core_nodes) != 1:
        raise ValueError("Expected exactly one CORE module")
    return modules[core_nodes[0]], modules

# -----------------------
# NN helper functions
# -----------------------
def tanh(x):
    return np.tanh(x)

def normalize_qpos(qpos):
    return np.clip(qpos / np.pi, -1.0, 1.0)

def normalize_xpos(xpos_flat):
    return np.clip(xpos_flat / 10.0, -1.0, 1.0)

def normalize_xmat(xmat_flat):
    return np.clip(xmat_flat, -1.0, 1.0)

def assemble_inputs_from_data(data, model):
    NQ = model.nq
    NBODY = model.nbody
    qpos = np.array(data.qpos[:NQ], dtype=np.float64)
    qpos_norm = normalize_qpos(qpos)

    try:
        xpos = np.array(data.xpos[0], dtype=np.float64)
    except Exception:
        xpos = np.zeros(3, dtype=np.float64)
    xpos_norm = normalize_xpos(xpos)

    try:
        xmat = np.array(data.xmat[0].reshape(-1), dtype=np.float64)
    except Exception:
        xmat = np.zeros(9, dtype=np.float64)
    xmat_norm = normalize_xmat(xmat)

    features = np.concatenate([qpos_norm, xpos_norm, xmat_norm]).astype(np.float32)
    return features  # length NQ+3+9

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

# -----------------------
# FITNESS SIMULATION WITH PATH TRACKING
# -----------------------
def run_simulation(genome, steps=500, return_path=False):
    robot_graph = load_graph_from_json(ROBOT_JSON)
    core, _ = construct_core_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, position=np.array(SPAWN_POSITION))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Get body ID of the CORE module
    core_body_id = 0

    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)

    NQ = model.nq
    NBODY = model.nbody
    NU = model.nu
    BASE_INPUT_SIZE = NQ + 3 + 9
    INPUT_SIZE = BASE_INPUT_SIZE + 2
    OUTPUT_SIZE = NU

    W_in, W_rec, W_out = decode_genome(genome, INPUT_SIZE, OUTPUT_SIZE)
    h = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    ctrl_state = np.zeros(NU, dtype=np.float64)

    history = []

    for t in range(steps):
        base_feats = assemble_inputs_from_data(data, model)
        phase = 2 * np.pi * (t / steps)
        time_feats = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
        inputs = np.concatenate([base_feats, time_feats]).astype(np.float64)

        h = tanh(np.dot(inputs, W_in) + np.dot(h, W_rec))
        raw_outputs = tanh(np.dot(h, W_out))
        ctrl_state = velocity_to_target(ctrl_state, raw_outputs)
        napply = min(len(data.ctrl), len(ctrl_state))
        data.ctrl[:napply] = ctrl_state[:napply]

        mj_step(model, data)

        # track core position
        try:
            xpos = np.array(data.xpos[core_body_id], dtype=np.float64)
        except Exception:
            xpos = np.zeros(3, dtype=np.float64)
        history.append(xpos.copy())

    final_pos = np.array(history[-1])
    fitness = -np.linalg.norm(TARGET_POSITION - final_pos)

    if return_path:
        return fitness, np.array(history)
    else:
        return fitness

# -----------------------
# VIEWER RUNNER
# -----------------------
def run_viewer(genome):
    mujoco.set_mjcb_control(None)
    robot_graph = load_graph_from_json(ROBOT_JSON)
    core, _ = construct_core_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, position=np.array(SPAWN_POSITION))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    core_body_id = 0

    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)

    NQ = model.nq
    NBODY = model.nbody
    NU = model.nu
    BASE_INPUT_SIZE = NQ + 3 + 9
    INPUT_SIZE = BASE_INPUT_SIZE + 2
    OUTPUT_SIZE = NU

    W_in, W_rec, W_out = decode_genome(genome, INPUT_SIZE, OUTPUT_SIZE)
    h = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    ctrl_state = np.zeros(NU, dtype=np.float64)

    def control_callback(model_cb, data_cb):
        nonlocal h, ctrl_state
        base_feats = assemble_inputs_from_data(data_cb, model_cb)
        t = getattr(data_cb, "time", 0.0)
        phase = 2 * np.pi * (t / 5.0)
        time_feats = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
        inputs = np.concatenate([base_feats, time_feats]).astype(np.float64)

        h = tanh(np.dot(inputs, W_in) + np.dot(h, W_rec))
        raw_outputs = tanh(np.dot(h, W_out))
        ctrl_state = velocity_to_target(ctrl_state, raw_outputs)
        napply = min(len(data_cb.ctrl), len(ctrl_state))
        data_cb.ctrl[:napply] = ctrl_state[:napply]

    mujoco.set_mjcb_control(control_callback)
    viewer.launch(model=model, data=data)

# -----------------------
# MAIN
# -----------------------
if __name__ == "__main__":
    # Load genome
    with open(GENOME_JSON, "r") as f:
        genome_data = json.load(f)
    genome = np.array(genome_data["genome"], dtype=np.float32)

    # Evaluate fitness and track path
    fitness, path = run_simulation(genome, steps=500, return_path=True)
    print(f"Fitness of loaded genome: {fitness:.4f}")

    # Save path for visualization
    np.save(PATH_SAVE_FILE, path)
    print(f"Saved robot path to {PATH_SAVE_FILE}")

    # Launch viewer for visualization
    run_viewer(genome)
