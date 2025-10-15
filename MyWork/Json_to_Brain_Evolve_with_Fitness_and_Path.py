"""
EvoTorch + MuJoCo Robot Example
Tracks fitness of all individuals and path of best individual per generation
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
# Settings
# -----------------------
JSON_PATH = Path(r"D:\Evolutionary Computing GitClone Ariel\ariel\MyWork\Nested_Evolution\Step_Curry.json")
SAVE_DIR = Path(r"D:\Evolutionary")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

POPULATION_SIZE = 25
GENERATIONS = 41
SIMULATION_STEPS = 20000
INITIAL_WEIGHT_RANGE = 1.0
HIDDEN_SIZE = 16
MAX_VELOCITY = 0.05
MAX_ANGLE = np.pi / 2
TIME_INPUTS = 2
SPAWN_POSITION = [-0.8, 0, 0.1]
TARGET_POSITION = [5.0, 0.0, 0.5]

# -----------------------
# Helpers
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
    core_nodes = [nid for nid, attrs in graph.nodes(data=True) if attrs["type"] == "CORE"]
    if len(core_nodes) != 1:
        raise ValueError("Expected exactly one CORE module")
    return modules[core_nodes[0]], modules

robot_graph = load_graph_from_json(JSON_PATH)

# -----------------------
# Sample model for dimensions
# -----------------------
def build_sample_model():
    core, modules = construct_core_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, position=np.array(SPAWN_POSITION))
    model = world.spec.compile()
    return model, core, modules

_sample_model, _sample_core, _sample_modules = build_sample_model()
NQ = int(_sample_model.nq)
NBODY = int(_sample_model.nbody)
NU = int(_sample_model.nu)
BASE_INPUT_SIZE = NQ + 3 + 9
OUTPUT_SIZE = NU
INPUT_SIZE = BASE_INPUT_SIZE + TIME_INPUTS
GENOME_SIZE = INPUT_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * HIDDEN_SIZE + HIDDEN_SIZE * OUTPUT_SIZE

print(f"Computed dims: NQ={NQ}, NBODY={NBODY}, NU={NU}")
print(f"BASE_INPUT_SIZE={BASE_INPUT_SIZE}, INPUT_SIZE={INPUT_SIZE}, GENOME_SIZE={GENOME_SIZE}")

# -----------------------
# Genome decoding
# -----------------------
def decode_genome(genome):
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

def tanh(x): return np.tanh(x)

def normalize_qpos(qpos: np.ndarray) -> np.ndarray: return np.clip(qpos / np.pi, -1.0, 1.0)
def normalize_xpos(xpos_flat: np.ndarray) -> np.ndarray: return np.clip(xpos_flat / 10.0, -1.0, 1.0)
def normalize_xmat(xmat_flat: np.ndarray) -> np.ndarray: return np.clip(xmat_flat, -1.0, 1.0)

def assemble_inputs_from_data(data, model):
    qpos = np.array(data.qpos[:NQ], dtype=np.float64)
    qpos_norm = normalize_qpos(qpos)
    try:
        core_body_id = 0
        xpos = np.array(data.xpos[core_body_id], dtype=np.float64)
    except Exception:
        xpos = np.zeros(3, dtype=np.float64)
    xpos_norm = normalize_xpos(xpos)
    try:
        core_body_id = 0
        xmat = np.array(data.xmat[core_body_id].reshape(-1), dtype=np.float64)
    except Exception:
        xmat = np.zeros(NBODY * 9, dtype=np.float64)
    xmat_norm = normalize_xmat(xmat)
    return np.concatenate([qpos_norm, xpos_norm, xmat_norm]).astype(np.float32)

def velocity_to_target(prev_ctrl: np.ndarray, raw_outputs: np.ndarray) -> np.ndarray:
    velocity = raw_outputs * MAX_VELOCITY
    new_ctrl = prev_ctrl + velocity
    return np.clip(new_ctrl, -MAX_ANGLE, MAX_ANGLE)

def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]
    return -np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)

# -----------------------
# Simulation
# -----------------------
def run_simulation(genome: torch.Tensor, steps: int = SIMULATION_STEPS):
    core, _ = construct_core_from_graph(robot_graph)
    world = OlympicArena()
    world.spawn(core.spec, position=np.array(SPAWN_POSITION))
    model = world.spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]
    use_tracker = to_track[0] if len(to_track) > 0 else None

    W_in, W_rec, W_out = decode_genome(genome)
    h = np.zeros(HIDDEN_SIZE, dtype=np.float64)
    ctrl_state = np.zeros(OUTPUT_SIZE, dtype=np.float64)
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

        if use_tracker is not None:
            xpos = tuple(use_tracker.xpos.tolist())
        else:
            try:
                xpos = tuple(np.array(data.xpos[:3]).tolist())
            except Exception:
                xpos = (0.0, 0.0, 0.0)
        history.append(xpos)

    return fitness_function(history), np.array(history)

# -----------------------
# EvoTorch problem
# -----------------------
def evaluate(genome: torch.Tensor) -> float:
    fit, _ = run_simulation(genome)
    return fit

problem = Problem(
    "max",
    evaluate,
    solution_length=int(GENOME_SIZE),
    dtype=torch.float32,
    initial_bounds=(-INITIAL_WEIGHT_RANGE, INITIAL_WEIGHT_RANGE),
)

searcher = CMAES(problem, popsize=POPULATION_SIZE, stdev_init=0.5)

# -----------------------
# Evolution loop (save fitness and paths)
# -----------------------
fitness_all = []        # list of arrays: shape (popsize,)
paths_best_per_gen = [] # list of arrays: shape (steps, 3)

for gen in range(GENERATIONS):
    # Step once
    searcher.step()
    # Evaluate all individuals manually to save fitness
    pop_genomes = [ind.values for ind in searcher.population]
    fitness_gen = []
    best_fit = -np.inf
    best_path = None
    for genome in pop_genomes:
        fit, path = run_simulation(genome)
        fitness_gen.append(fit)
        if fit > best_fit:
            best_fit = fit
            best_path = path
    fitness_all.append(np.array(fitness_gen, dtype=np.float32))
    paths_best_per_gen.append(best_path)
    print(f"Gen {gen} | Best Fitness: {best_fit:.4f}")

SAVE_DIR = Path(r"MyWork\Nested_Evolution")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

fitness_file = SAVE_DIR / f"best_robot_fitness_POP{POPULATION_SIZE}_GEN{GENERATIONS}.fitness_all.npy"
paths_file   = SAVE_DIR / f"best_robot_fitness_POP{POPULATION_SIZE}_GEN{GENERATIONS}.paths_best_per_gen.npy"

np.save(fitness_file, fitness_all)
np.save(paths_file, paths_best_per_gen)

print(f"Saved fitness of all individuals to {fitness_file}")
print(f"Saved best paths per generation to {paths_file}")

# -----------------------
# Plot best fitness per generation
# -----------------------
best_fitness_per_gen = [f.max() for f in fitness_all]
plt.figure(figsize=(8, 5))
plt.plot(best_fitness_per_gen, marker="o", linestyle="-")
plt.xlabel("Generation")
plt.ylabel("Best Fitness (final -distance)")
plt.title("Evolution Progress (CMA-ES)")
plt.grid(True)
plt.show()

# -----------------------
# Run best genome in viewer
# -----------------------
best_genome = searcher.status["best"].values.numpy()

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
    run_best_genome(best_genome)
