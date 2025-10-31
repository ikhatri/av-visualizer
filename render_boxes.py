import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from einops import repeat
from pyquaternion import Quaternion
from trimesh.visual import TextureVisuals
from trimesh.visual.material import PBRMaterial

from av2_colors import AV2_COLORS


def main():
    server = viser.ViserServer()

    num_frames = 10

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
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)

    av2_dataset = Argoverse2CausalSceneFlow(
        root_dir=Path("/Users/ikhatri/Data/argoverse2/train"),
        use_gt_flow=False,
        load_boxes=True,
        load_flow=False,
        use_cache=True,
        subsequence_length=num_frames,
    )
    point_nodes: list[viser.PointCloudHandle] = []
    box_nodes: list[dict[str, viser.BatchedGlbHandle]] = []
    subsequence = av2_dataset[0]
    for i, frame in enumerate(subsequence):
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

        boxes_by_category = {}
        for box in frame.boxes:
            category = box.category
            if category not in boxes_by_category:
                m = PBRMaterial(
                    name=category,
                    baseColorFactor=[*AV2_COLORS[category], 128],  # Your RGBA color
                    alphaMode="BLEND",
                )
                base_box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
                base_box.visual = TextureVisuals(material=m)
                boxes_by_category[category] = {
                    "dims": [],
                    "positions": [],
                    "rotations": [],
                    "base_box": base_box,
                }

            boxes_by_category[category]["dims"].append(
                [box.length, box.width, box.height]
            )
            boxes_by_category[category]["positions"].append(box.pose.translation)
            boxes_by_category[category]["rotations"].append(
                Quaternion(matrix=box.pose.rotation_matrix).elements
            )

        category_nodes = {}
        for category in boxes_by_category.keys():
            category_nodes[category] = server.scene.add_batched_meshes_trimesh(
                name=f"/frame_{i + 1}/{category}_boxes",
                batched_positions=np.stack(boxes_by_category[category]["positions"]),
                batched_scales=np.array(boxes_by_category[category]["dims"]),
                batched_wxyzs=np.stack(boxes_by_category[category]["rotations"]),
                mesh=boxes_by_category[category]["base_box"],
            )
        box_nodes.append(category_nodes)

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
            for c in box_nodes[current_timestep]:
                box_nodes[current_timestep][c].visible = True

            point_nodes[prev_timestep].visible = False
            for c in box_nodes[prev_timestep]:
                box_nodes[prev_timestep][c].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    while True:
        # Update the timestep if we're playing.
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

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
