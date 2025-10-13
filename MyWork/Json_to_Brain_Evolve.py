"""
EvoTorch + MuJoCo Robot Example
Extended inputs: hinge qpos + all bodies' xpos + all bodies' xmat (rotation) + time features
Fitness: negative final Euclidean distance to TARGET_POSITION (so we maximize -distance)

Adjust JSON_PATH, SIMULATION_STEPS, POPULATION_SIZE, etc. as needed.
"""

# Third-party libraries
import numpy as np
import torch
import json
from pathlib import Path
from evotorch import Problem
from evotorch.algorithms import CMAES
import matplotlib.pyplot as plt
import networkx as nx
import mujoco
from mujoco import mj_step, viewer

# ariel imports (ensure ariel is on PYTHONPATH)
from ariel.simulation.environments.olympic_arena import OlympicArena
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces

# -----------------------
# Settings (user-changeable)
# -----------------------
JSON_PATH = Path(r"D:\Evolutionary Computing GitClone Ariel\ariel\MyWork\Nested_Evolution\best_robot_body_P100_C20_G45_CG20.json")
POPULATION_SIZE = 70
GENERATIONS = 200
SIMULATION_STEPS = 40000
INITIAL_WEIGHT_RANGE = 1.0
HIDDEN_SIZE = 16
MAX_VELOCITY = 0.05
MAX_ANGLE = np.pi / 2
TIME_INPUTS = 2
SPAWN_POSITION = [-0.8, 0, 0.1]
TARGET_POSITION = [5.0, 0.0, 0.5]  # Target to reach
  # replace with actual core body index

# -----------------------
# Helpers: JSON graph loader and robot constructor
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
                # ignore rotation parse errors
                pass
    # attach bodies according to edges
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
            # if site doesn't exist or attach fails, skip gracefully
            pass
    core_nodes = [nid for nid, attrs in graph.nodes(data=True) if attrs["type"] == "CORE"]
    if len(core_nodes) != 1:
        raise ValueError("Expected exactly one CORE module")
    return modules[core_nodes[0]], modules

# load graph now
robot_graph = load_graph_from_json(JSON_PATH)

# -----------------------
# Create a temporary model to determine sizes (nq, nbody, nu)
# -----------------------
def build_sample_model():
    core, modules = construct_core_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, position=np.array(SPAWN_POSITION))
    model = world.spec.compile()
    return model, core, modules

_sample_model, _sample_core, _sample_modules = build_sample_model()

# read dimensions from sample model
NQ = int(_sample_model.nq)         # number of generalized coordinates
NBODY = int(_sample_model.nbody)   # number of bodies
NU = int(_sample_model.nu)         # number of actuators (controls)

# Now compute base input size:
# - joint angles: NQ
# - all body positions: NBODY * 3
# - all body rotation matrices (xmat): NBODY * 9
# (if you want to add quaternions or velocities, add them here)
BASE_INPUT_SIZE = NQ + 3 +  9
OUTPUT_SIZE = NU
INPUT_SIZE = BASE_INPUT_SIZE + TIME_INPUTS
GENOME_SIZE = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE

print(f"Computed dims: NQ={NQ}, NBODY={NBODY}, NU={NU}")
print(f"BASE_INPUT_SIZE={BASE_INPUT_SIZE}, INPUT_SIZE={INPUT_SIZE}, GENOME_SIZE={GENOME_SIZE}")

# -----------------------
# Decode genome into weight matrices
# -----------------------
def decode_genome(genome):
    # convert to NumPy array if it's a tensor
    if not isinstance(genome, np.ndarray):
        genome = genome.detach().cpu().numpy()
    g = genome.astype(np.float64)
    
    idx = 0
    W_in = g[idx: idx + INPUT_SIZE*HIDDEN_SIZE].reshape(INPUT_SIZE, HIDDEN_SIZE)
    idx += INPUT_SIZE*HIDDEN_SIZE
    W_rec = g[idx: idx + HIDDEN_SIZE*HIDDEN_SIZE].reshape(HIDDEN_SIZE, HIDDEN_SIZE)
    idx += HIDDEN_SIZE*HIDDEN_SIZE
    W_out = g[idx: idx + HIDDEN_SIZE*OUTPUT_SIZE].reshape(HIDDEN_SIZE, OUTPUT_SIZE)
    return W_in, W_rec, W_out

# -----------------------
# Activations and normalizations
# -----------------------
def tanh(x):
    return np.tanh(x)

def normalize_qpos(qpos: np.ndarray) -> np.ndarray:
    # qpos are angles/positions; for angles normalise by pi to [-1,1]
    return np.clip(qpos / np.pi, -1.0, 1.0)

def normalize_xpos(xpos_flat: np.ndarray) -> np.ndarray:
    # xpos are world coordinates (meters). scale by a reasonable scene scale (e.g., 10m)
    # clamp to [-1,1]
    return np.clip(xpos_flat / 10.0, -1.0, 1.0)

def normalize_xmat(xmat_flat: np.ndarray) -> np.ndarray:
    # rotation matrices entries are bounded roughly in [-1,1] already
    return np.clip(xmat_flat, -1.0, 1.0)

def assemble_inputs_from_data(data, model):
    """
    Collects features:
    - qpos (NQ)
    - data.xpos flattened (NBODY*3)
    - data.xmat flattened (NBODY*9)
    - time sin/cos (2)
    Returns concatenated float32 1D array of length INPUT_SIZE
    """
    # qpos may be bigger than NQ if mocap etc. We take first NQ entries.
    qpos = np.array(data.qpos[:NQ], dtype=np.float64)
    qpos_norm = normalize_qpos(qpos)

    # xpos: body positions; attempt to use data.xpos attribute
    try:
        core_body_id = 0  # adjust if core body index differs
        xpos = np.array(data.xpos[core_body_id], dtype=np.float64)  # 3 values
    except Exception:
        xpos = np.zeros(3, dtype=np.float64)
    xpos_norm = normalize_xpos(xpos)


    # xmat: body rotation matrices (3x3) flattened
    try:
        core_body_id = 0
        xmat = np.array(data.xmat[core_body_id].reshape(-1), dtype=np.float64)  # should be nbody*9
    except Exception:
        # fallback zeros
        xmat = np.zeros(NBODY * 9, dtype=np.float64)

    xmat_norm = normalize_xmat(xmat)

    # time inputs (sin/cos)
    # use data.time if available, else use step counter fallback (handled by caller)
    # caller will pass time via data.time in visualization; in run_simulation we use step count
    # here include placeholders - caller should append sin/cos after this function
    features = np.concatenate([qpos_norm, xpos_norm, xmat_norm]).astype(np.float32)
    return features  # length BASE_INPUT_SIZE

def velocity_to_target(prev_ctrl: np.ndarray, raw_outputs: np.ndarray) -> np.ndarray:
    velocity = raw_outputs * MAX_VELOCITY
    new_ctrl = prev_ctrl + velocity
    return np.clip(new_ctrl, -MAX_ANGLE, MAX_ANGLE)

# -----------------------
# Fitness function on history (final core position)
# -----------------------
def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    cartesian_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    return -cartesian_distance

# -----------------------
# Simulation (fresh robot each run)
# -----------------------
def run_simulation(genome: torch.Tensor, steps: int = 500) -> float:
    # build fresh robot
    core, _ = construct_core_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, position=np.array(SPAWN_POSITION))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # reset & forward
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)

    # pick a body to track for final position: prefer 'core' geom/body name
    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    # fallback to first body location if no 'core' geom found
    use_tracker = None
    if len(to_track) > 0:
        use_tracker = to_track[0]
    else:
        # fallback to body 0 position from data.xpos
        use_tracker = None

    # decode controller weights
    W_in, W_rec, W_out = decode_genome(genome)
    h = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    ctrl_state = np.zeros(OUTPUT_SIZE, dtype=np.float64)

    history = []

    for t in range(steps):
        # assemble state features
        base_feats = assemble_inputs_from_data(data, model)  # length BASE_INPUT_SIZE
        phase = 2 * np.pi * (t / steps)
        time_feats = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32)
        inputs = np.concatenate([base_feats, time_feats]).astype(np.float64)  # shape (INPUT_SIZE,)

        # forward pass (recurrent)
        h = tanh(np.dot(inputs, W_in) + np.dot(h, W_rec))
        raw_outputs = tanh(np.dot(h, W_out))
        ctrl_state = velocity_to_target(ctrl_state, raw_outputs)
        # apply controls (MuJoCo expects length nu)
        # If OUTPUT_SIZE < model.nu, MuJoCo will ignore extras; if >, we slice
        napply = min(len(data.ctrl), len(ctrl_state))
        data.ctrl[:napply] = ctrl_state[:napply]

        # step simulator
        mj_step(model, data)

        # track core position: try to use the bound geom if available
        if use_tracker is not None:
            xpos = tuple(use_tracker.xpos.tolist())
        else:
            # fallback: take the model body 0 pos from data.xpos
            try:
                xpos = tuple(np.array(data.xpos[:3]).tolist())  # first 3 numbers
            except Exception:
                xpos = (0.0, 0.0, 0.0)
        history.append(xpos)

    # compute fitness based on final position
    return fitness_function(history)

# -----------------------
# Evaluate wrapper for EvoTorch
# -----------------------
def evaluate(genome: torch.Tensor) -> float:
    return run_simulation(genome, steps=SIMULATION_STEPS)

# -----------------------
# EvoTorch problem & CMA-ES
# -----------------------
problem = Problem(
    "max",                      # we maximize negative distance
    evaluate,
    solution_length=int(GENOME_SIZE),
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
    best_fit = float(searcher.status["best_eval"])
    fitness_history.append(best_fit)
    print(f"Gen {gen} | Best Fitness: {best_fit:.4f}")

best_genome = searcher.status["best"].values.numpy()  # <-- convert to NumPy

print("\nBest genome found (first 10 values):", best_genome[:10])

# -----------------------
# Save best genome as JSON
# -----------------------
save_dir = Path(r"D:\Evolutionary")
save_dir.mkdir(parents=True, exist_ok=True)

genome_dict = {
    "genome": best_genome.tolist(),
    "POPULATION_SIZE": POPULATION_SIZE,
    "GENERATIONS": GENERATIONS
}

json_filename = f"best_genome_POP{POPULATION_SIZE}_GEN{GENERATIONS}.json"
json_path = save_dir / json_filename

with open(json_path, "w") as f:
    json.dump(genome_dict, f, indent=4)

print(f"Saved best genome as JSON to: {json_path}")

# -----------------------
# Plot fitness progress
# -----------------------
plt.figure(figsize=(8, 5))
plt.plot(fitness_history, marker="o", linestyle="-")
plt.xlabel("Generation")
plt.ylabel("Best Fitness (final -distance)")
plt.title("Evolution Progress (CMA-ES)")
plt.grid(True)
plt.show()

# -----------------------
# Visualize best genome in viewer
# -----------------------
def run_best_genome(best_genome):
    mujoco.set_mjcb_control(None)
    core, _ = construct_core_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, position=np.array(SPAWN_POSITION))
    model = world.spec.compile()
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)

    W_in, W_rec, W_out = decode_genome(best_genome)
    h = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    ctrl_state = np.zeros(OUTPUT_SIZE, dtype=np.float64)

    def control_callback(model_cb, data_cb):
        nonlocal h, ctrl_state
        # assemble features using the same function
        base_feats = assemble_inputs_from_data(data_cb, model_cb)
        t = data_cb.time if hasattr(data_cb, "time") else 0.0
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

if __name__ == "__main__":
    # Run viewer for the best genome found after evolutionary run
    run_best_genome(best_genome)
