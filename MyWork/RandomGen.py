# =======================
# Random Body Generator + Filter
# =======================

from pathlib import Path
import numpy as np
from networkx import DiGraph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import save_graph_as_json
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.environments import OlympicArena
from ariel.utils.tracker import Tracker
import mujoco as mj

# =======================
# Configuration
# =======================
SEED = 123
RNG = np.random.default_rng(SEED)
NUM_MODULES = 30
SPAWN_POS = [-0.8, 0, 0.1]
DISPLACEMENT_THRESHOLD = 0
MAX_BRICKS = 6
MIN_Actuators = 16
MIN_BRICKS = 3

# Save to requested folder
SAVE_PATH = Path(r"D:\Evolutionary Computing GitClone Ariel\ariel\MyWork\Nested_Evolution") / "random_robotV5.json"
SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)

# =======================
# Random Body Generator
# =======================
def generate_random_body_vector():
    return RNG.uniform(-1.0, 1.0, size=64*3)

def decode_to_graph(body_vector):
    type_p = body_vector[:64]
    conn_p = body_vector[64:128]
    rot_p = body_vector[128:]
    genotype = [type_p, conn_p, rot_p]

    nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_MODULES)
    p_matrices = nde.forward(genotype)
    hpd = HighProbabilityDecoder(NUM_MODULES)
    robot_graph: DiGraph = hpd.probability_matrices_to_graph(
        p_matrices[0], p_matrices[1], p_matrices[2]
    )
    return robot_graph

# =======================
# Displacement Filter
# =======================
def passes_displacement_test(robot_graph):
    world = OlympicArena()
    fresh_core = construct_mjspec_from_graph(robot_graph)
    world.spawn(fresh_core.spec, position=np.array(SPAWN_POS))

    model = world.spec.compile()
    data = mj.MjData(model)
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)

    tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
    tracker.setup(world.spec, data)

    num_actuators = model.nu
    steps = 10000  # simulate 3 seconds at 0.01s per step
    for _ in range(steps):
        data.ctrl[:] = RNG.uniform(-1, 1, size=num_actuators)
        mj.mj_step(model, data)
        tracker.update(data)

    start = tracker.history["xpos"][0][0]
    end = tracker.history["xpos"][0][-1]
    x0, y0, _ = start
    x1, y1, _ = end
    displacement_xy = np.sqrt((x1 - x0)**2 + (y1 - y0)**2)

    return displacement_xy >= DISPLACEMENT_THRESHOLD

# =======================
# Main Loop
# =======================
while True:
    body_vector = generate_random_body_vector()
    robot_graph = decode_to_graph(body_vector)

    # Reject robots with too many bricks
    num_bricks = sum(1 for _, data in robot_graph.nodes(data=True) if data.get("type") == "BRICK")
    if num_bricks > MAX_BRICKS:
        print(f"[FILTER] Robot has {num_bricks} bricks, skipping...")
        continue

        # Reject robots with too many bricks
    num_bricks = sum(1 for _, data in robot_graph.nodes(data=True) if data.get("type") == "BRICK")
    if num_bricks < MIN_BRICKS:
        print(f"[FILTER] Robot has {num_bricks} bricks, skipping...")
        continue

    num_Actuators = sum(1 for _, data in robot_graph.nodes(data=True) if data.get("type") == "HINGE")
    if num_Actuators < MIN_Actuators:
        print(f"[FILTER] Robot has {num_Actuators} actuators, skipping...")
        continue

    if not passes_displacement_test(robot_graph):
        print("[FILTER] Robot failed displacement test, skipping...")
        continue

    # Save the first valid robot
    save_graph_as_json(robot_graph, SAVE_PATH)
    print(f"[SUCCESS] Saved valid robot JSON to {SAVE_PATH}")
    break
