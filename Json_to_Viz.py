"""
Run a robot from a saved JSON graph.

Author:     jmdm / adapted
Date:       2025-10-09
Py Ver:     3.12
OS:         Windows / macOS
Status:     Ready-to-run ✅
"""

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any
import json

# Third-party libraries
import mujoco
from mujoco import viewer
from rich.console import Console
import networkx as nx
from networkx import DiGraph

# Local libraries
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.body_phenotypes.robogen_lite.modules.core import CoreModule
from ariel.simulation.environments import SimpleFlatWorld
from ariel.utils.renderers import single_frame_renderer

if TYPE_CHECKING:
    from networkx import DiGraph

# Console for logging
console = Console()

# Path to your JSON
JSON_PATH = Path(
    r"D:\Evolutionary Computing GitClone Ariel\ariel\MyWork\Nested_Evolution\best_robot_body_P20_C10_G150_CG10.json"
)
DATA = JSON_PATH.parent
SCRIPT_NAME = JSON_PATH.stem


def load_graph_from_custom_json(path: Path) -> DiGraph:
    """Load a graph saved with save_graph_as_json (custom format)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    graph = DiGraph()
    
    # Add nodes with 'type' and 'rotation'
    for node in data["nodes"]:
        node_id = node["id"]
        graph.add_node(node_id, type=node["type"], rotation=node["rotation"])
    
    # Add edges with 'face'
    for edge in data["edges"]:
        src = edge["source"]
        tgt = edge["target"]
        graph.add_edge(src, tgt, face=edge["face"])
    
    return graph


def run(robot: CoreModule, *, with_viewer: bool = False) -> None:
    """Run simulation of the robot."""
    viz_options = mujoco.MjvOption()
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_ACTUATOR] = True
    viz_options.flags[mujoco.mjtVisFlag.mjVIS_BODYBVH] = True

    # MuJoCo world
    world = SimpleFlatWorld()

    # Optional: set transparency for geoms
    for i in range(len(robot.spec.geoms)):
        robot.spec.geoms[i].rgba[-1] = 0.5

    # Spawn robot in world
    world.spawn(robot.spec)

    # Compile model and create data
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Save model XML
    xml_path = DATA / f"{SCRIPT_NAME}.xml"
    with xml_path.open("w", encoding="utf-8") as f:
        f.write(world.spec.to_xml())
    console.log(f"Saved XML to {xml_path}")

    # Log DoFs and actuators
    console.log(f"DoF (model.nv): {model.nv}, Actuators (model.nu): {model.nu}")

    # Reset simulation
    mujoco.mj_resetData(model, data)

    # Render single frame
    single_frame_renderer(model, data, steps=10)

    # Launch viewer
    if with_viewer:
        viewer.launch(model=model, data=data)


def main() -> None:
    """Load the robot from JSON and simulate it."""
    graph = load_graph_from_custom_json(JSON_PATH)
    console.log(f"Loaded graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges")

    core = construct_mjspec_from_graph(graph)
    run(core, with_viewer=True)


if __name__ == "__main__":
    main()
