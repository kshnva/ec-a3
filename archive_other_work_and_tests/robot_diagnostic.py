"""Enhanced robot diagnostic script."""

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

def detailed_robot_diagnostic():
    print("=== DETAILED ROBOT DIAGNOSTIC ===")
    
    # Load robot
    robot_path = Path("Robot_Result/Step_Curry.json")
    robot_graph = load_graph_from_json(robot_path)
    core, modules = construct_core_from_graph(robot_graph)
    
    print(f"Robot graph has {len(robot_graph.nodes)} nodes and {len(robot_graph.edges)} edges")
    
    # Create world
    world = OlympicArena()
    spawn_pos = np.array([-2.4, 0.0, 0.1])
    print(f"Spawning robot at: {spawn_pos}")
    world.spawn(core.spec, position=spawn_pos)
    
    # Compile model
    model = world.spec.compile()
    data = mujoco.MjData(model)
    
    print(f"\n=== MODEL INFO ===")
    print(f"Bodies: {model.nbody}")
    print(f"Joints: {model.njnt}")
    print(f"Actuators: {model.nu}")
    print(f"Degrees of freedom: {model.nq}")
    print(f"Generalized velocities: {model.nv}")
    
    # Reset and check all positions
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0
    data.qacc[:] = 0
    mujoco.mj_forward(model, data)
    
    print(f"\n=== INITIAL STATE ===")
    print(f"Number of bodies in xpos: {len(data.xpos)}")
    print("All body positions:")
    for i, pos in enumerate(data.xpos):
        print(f"  Body {i}: {pos}")
    
    print(f"\nGeneralized positions (qpos): {data.qpos}")
    print(f"Control inputs (ctrl): {data.ctrl}")
    
    # Find which body is the robot's root
    print(f"\n=== FINDING ROBOT ROOT ===")
    # Look for non-zero positions or positions near spawn
    robot_body_idx = -1
    for i, pos in enumerate(data.xpos):
        distance_from_spawn = np.linalg.norm(pos - spawn_pos)
        if distance_from_spawn < 1.0:  # Within 1 meter of spawn
            print(f"  Body {i} at {pos} is {distance_from_spawn:.3f}m from spawn - likely robot body")
            if robot_body_idx == -1:
                robot_body_idx = i
    
    if robot_body_idx == -1:
        print("  No robot body found near spawn position!")
        return
    
    print(f"Using body {robot_body_idx} as robot root")
    
    # Test movement with stronger controls
    print(f"\n=== MOVEMENT TEST ===")
    initial_pos = data.xpos[robot_body_idx].copy()
    print(f"Initial robot position: {initial_pos}")
    
    # Try much stronger control signals
    for test_name, control_func in [
        ("Strong oscillation", lambda t: 1.0 * np.sin(t * 0.05)),
        ("Random strong", lambda t: np.random.uniform(-1, 1, model.nu)),
        ("Max positive", lambda t: np.ones(model.nu)),
        ("Max negative", lambda t: -np.ones(model.nu))
    ]:
        print(f"\nTesting: {test_name}")
        
        # Reset
        mujoco.mj_resetData(model, data)
        data.qvel[:] = 0
        data.qacc[:] = 0
        mujoco.mj_forward(model, data)
        
        start_pos = data.xpos[robot_body_idx].copy()
        
        for step in range(500):
            if test_name == "Random strong":
                data.ctrl[:] = np.random.uniform(-1, 1, model.nu)
            else:
                data.ctrl[:] = control_func(step)
            
            mujoco.mj_step(model, data)
            
            if step % 100 == 0:
                current_pos = data.xpos[robot_body_idx].copy()
                distance = np.linalg.norm(current_pos - start_pos)
                print(f"  Step {step}: pos={current_pos}, moved={distance:.4f}")
        
        final_pos = data.xpos[robot_body_idx].copy()
        total_distance = np.linalg.norm(final_pos - start_pos)
        print(f"  Total movement: {total_distance:.4f}")

if __name__ == "__main__":
    detailed_robot_diagnostic()