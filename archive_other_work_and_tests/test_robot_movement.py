"""Test script to check if the robot can move with simple controls."""

import numpy as np
import json
from pathlib import Path
import mujoco
from ariel.simulation.environments.olympic_arena import OlympicArena
import networkx as nx
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.body_phenotypes.robogen_lite.modules.brick import BrickModule
from ariel.body_phenotypes.robogen_lite.modules.hinge import HingeModule
from ariel.body_phenotypes.robogen_lite.config import ModuleFaces

# Load the robot
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

def test_robot_movement():
    print("Testing robot movement...")
    
    # Load robot
    robot_path = Path("Robot_Result/Step_Curry.json")
    robot_graph = load_graph_from_json(robot_path)
    core, modules = construct_core_from_graph(robot_graph)
    
    # Create world
    world = OlympicArena()
    spawn_pos = np.array([-2.4, 0.0, 0.1])
    world.spawn(core.spec, position=spawn_pos)
    
    # Compile model
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    print(f"Robot has {model.nu} actuators (joints)")
    print(f"Robot has {model.nq} degrees of freedom")
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)
    
    # Record initial position
    initial_pos = data.xpos[0].copy()
    print(f"Initial position: {initial_pos}")
    
    # Test with constant control signals
    positions = []
    for test_name, control_values in [
        ("Zero control", np.zeros(model.nu)),
        ("Small positive", np.full(model.nu, 0.1)),
        ("Small negative", np.full(model.nu, -0.1)),
        ("Oscillating", None)  # Will be set in loop
    ]:
        print(f"\nTesting: {test_name}")
        
        # Reset simulation
        mujoco.mj_resetData(model, data)
        data.qvel[:] = 0
        data.qacc[:] = 0
        mujoco.mj_forward(model, data)
        
        start_pos = data.xpos[0].copy()
        
        # Run simulation
        for step in range(1000):
            if test_name == "Oscillating":
                # Sine wave control
                data.ctrl[:] = 0.2 * np.sin(step * 0.1)
            else:
                data.ctrl[:] = control_values
            
            mujoco.mj_step(model, data)
            
            if step % 200 == 0:
                current_pos = data.xpos[0].copy()
                distance = np.linalg.norm(current_pos - start_pos)
                print(f"  Step {step}: pos={current_pos}, distance_moved={distance:.4f}")
        
        final_pos = data.xpos[0].copy()
        total_distance = np.linalg.norm(final_pos - start_pos)
        print(f"  Final distance moved: {total_distance:.4f}")

if __name__ == "__main__":
    test_robot_movement()