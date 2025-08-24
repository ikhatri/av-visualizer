import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from einops import repeat
from pyquaternion import Quaternion

from av2_colors import AV2_COLORS


def main():
    server = viser.ViserServer()

    num_frames = 70

    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.05,
            step=1e-3,
            initial_value=0.03,
        )
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_playing = server.gui.add_checkbox("Playing", True)

    av2_dataset = Argoverse2CausalSceneFlow(
        root_dir=Path("/efs/argoverse2/val"),
        use_gt_flow=False,
        load_boxes=True,
        load_flow=False,
        use_cache=True,
        subsequence_length=num_frames,
    )
    point_nodes: list[viser.PointCloudHandle] = []

    frame_pair = av2_dataset[0]
    box_categories: set[str] = set()
    boxes_by_category: list[dict[str, dict[str, list]]] = []
    for i, frame in enumerate(frame_pair):
        points = frame.pc.full_ego_pc.points
        point_colors = repeat(
            np.array([0, 0, 255]).astype(np.uint8), "c -> n c", n=points.shape[0]
        )
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frame_{i + 1}/lidar_points",
                points=points,
                colors=point_colors,
                point_size=gui_point_size.value,
                visible=False,
                point_shape="rounded",
            )
        )
        box_categories.update(box.category for box in frame.boxes)

        boxes_i = {}
        for box in frame.boxes:
            category = box.category
            if category not in boxes_i:
                boxes_i[category] = {
                    "dims": [],
                    "positions": [],
                    "rotations": [],
                }
            boxes_i[category]["dims"].append([box.length, box.width, box.height])
            boxes_i[category]["positions"].append(box.pose.translation)
            boxes_i[category]["rotations"].append(
                Quaternion(matrix=box.pose.rotation_matrix).elements
            )
        boxes_by_category.append(boxes_i)

    trimesh_box = trimesh.creation.box(extents=[1, 1, 1])
    category_nodes: dict[str, viser.BatchedMeshHandle] = {}
    for category in box_categories:
        category_nodes[category] = server.scene.add_batched_meshes_simple(
            f"/boxes/{category}",
            trimesh_box.vertices,
            trimesh_box.faces,
            batched_positions=np.array(
                [[0.0, 0.0, 0.0]]
            ),  # Placeholder, will be updated.
            batched_wxyzs=np.array(
                [[1.0, 0.0, 0.0, 0.0]]
            ),  # Placeholder, will be updated.
            batched_scales=np.array([[1.0, 1.0, 1.0]]),
            batched_colors=AV2_COLORS[category],  # Broadcasted.
            opacity=0.5,
            flat_shading=True,
            lod="off",
        )

    # Playback update loop.
    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            # Toggle visibility.
            point_nodes[current_timestep].visible = True
            point_nodes[prev_timestep].visible = False

            # Update boxes for the current timestep.
            boxes_i = boxes_by_category[current_timestep]
            for category, node in category_nodes.items():
                box_dict = boxes_i.get(category, None)
                if box_dict is None:
                    node.visible = False
                    continue

                node.batched_positions = np.array(box_dict["positions"])
                node.batched_wxyzs = np.array(box_dict["rotations"])
                node.batched_scales = np.array(box_dict["dims"])

        prev_timestep = current_timestep
        server.flush()  # Optional!

    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames
            gui_timestep.disabled = True
        else:
            gui_timestep.disabled = False

        # Update point size of both this timestep and the next one! There's
        # redundancy here, but this will be optimized out internally by viser.
        #
        # We update the point size for the next timestep so that it will be
        # immediately available when we toggle the visibility.
        point_nodes[gui_timestep.value].point_size = gui_point_size.value
        point_nodes[
            (gui_timestep.value + 1) % num_frames
        ].point_size = gui_point_size.value

        time.sleep(1.0 / 10)


if __name__ == "__main__":
    main()
