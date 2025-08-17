import time
from pathlib import Path

# from typing import Annotated
import numpy as np
import viser
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from bucketed_scene_flow_eval.datastructures.dataclasses import BoundingBox
from pyquaternion import Quaternion

# from dltype import FloatTensor, IntTensor, dltyped


# @dltyped() # Need to remove this until macos float128 bug fixed
def colorize_points_z_height(
    points: np.ndarray,  # Annotated[np.ndarray, FloatTensor["N 3"]],
) -> np.ndarray:  # Annotated[np.ndarray, IntTensor["N 3"]]:
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    normalized_z = (points[:, 2] - z_min) / (z_max - z_min)
    point_colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    point_colors[:, 0] = (normalized_z * 255).astype(np.uint8)  # Red channel.
    point_colors[:, 2] = ((1 - normalized_z) * 255).astype(np.uint8)  # Blue channel.
    return point_colors


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
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)

    av2_dataset = Argoverse2CausalSceneFlow(
        root_dir=Path("/efs/argoverse2/val"),
        use_gt_flow=False,
        flow_data_path=Path(
            "/efs/argoverse2/val_sceneflow_feather"
        ),
        load_boxes=True,
        load_flow=False,
        use_cache=True,
        subsequence_length=num_frames,
    )
    point_nodes: list[viser.PointCloudHandle] = []
    box_nodes: list[list[viser.BoxHandle]] = []
    frame_pair=av2_dataset[0]
    for i, frame in enumerate(frame_pair):
        points = frame.pc.full_ego_pc.points
        point_colors = colorize_points_z_height(points)
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
        boxes_t = []
        for box in frame.boxes:
            boxes_t.append(server.scene.add_box(
                name=f"/frame_{i + 1}/boxes/{box.track_uuid}",
                dimensions=(box.length, box.width, box.height),
                position=box.pose.translation,
                wxyz=Quaternion(matrix=box.pose.rotation_matrix).elements,
                wireframe=True,
                visible=False,
            ))
        box_nodes.append(boxes_t)

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
            for b in box_nodes[current_timestep]:
                b.visible=True
            point_nodes[prev_timestep].visible = False
            for b in box_nodes[prev_timestep]:
                b.visible=False
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
