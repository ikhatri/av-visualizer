import time
from pathlib import Path

import numpy as np
import viser
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from einops import repeat


def main():
    server = viser.ViserServer()

    num_frames = 150

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
        use_gt_flow=True,
        load_boxes=False,
        load_flow=True,
        use_cache=True,
        subsequence_length=num_frames,
    )
    point_nodes: list[viser.PointCloudHandle] = []
    flow_nodes: list[viser.LineSegmentsHandle] = []
    subsequence = av2_dataset[0]
    for i, frame in enumerate(subsequence):
        points = frame.pc.full_ego_pc.points
        point_colors = repeat(
            np.array([0, 0, 255]).astype(np.uint8), "c -> n c", n=points.shape[0]
        )
        flowed_points = frame.pc.flow(frame.flow).full_ego_pc.points
        flow_start_color = repeat(
            np.array([0, 0, 255]).astype(np.uint8), "c -> n c", n=points.shape[0]
        )
        flow_end_color = repeat(
            np.array([255, 0, 0]).astype(np.uint8), "c -> n c", n=points.shape[0]
        )
        flow_colors = np.stack([flow_start_color, flow_end_color], axis=1)
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
        flow_nodes.append(
            server.scene.add_line_segments(
                name=f"/frame_{i + 1}/flow_to_{i + 2}",
                line_width=1,
                points=np.stack([points, flowed_points], axis=1),
                colors=flow_colors,
                visible=False,
            )
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
            flow_nodes[current_timestep].visible = True

            point_nodes[prev_timestep].visible = False
            flow_nodes[prev_timestep].visible = False
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
