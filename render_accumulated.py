import time
from pathlib import Path

import numpy as np
import viser
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from einops import repeat


def main():
    server = viser.ViserServer()

    num_frames = 2

    with server.gui.add_folder("Playback"):
        gui_point_size = server.gui.add_slider(
            "Point size",
            min=0.001,
            max=0.05,
            step=1e-3,
            initial_value=0.03,
        )
        gui_flow_compensation = server.gui.add_checkbox("Flow Compensation", False)

    av2_dataset = Argoverse2CausalSceneFlow(
        root_dir=Path("/Users/ikhatri/Data/argoverse2/train"),
        use_gt_flow=True,
        load_boxes=False,
        load_flow=True,
        use_cache=True,
        subsequence_length=num_frames,
    )
    point_nodes: dict[str, viser.PointCloudHandle] = {}
    subsequence = av2_dataset[0]
    for i, frame in enumerate(subsequence):
        points = frame.pc.full_global_pc.points
        color = i * 50
        point_colors = repeat(
            np.array([color, color, color]).astype(np.uint8),
            "c -> n c",
            n=points.shape[0],
        )
        point_nodes[f"/frame_{i + 1}/lidar_points"] = server.scene.add_point_cloud(
            name=f"/frame_{i + 1}/lidar_points",
            points=points,
            colors=point_colors,
            point_size=gui_point_size.value,
            visible=True,
            point_shape="rounded",
        )

        # frame.flow.full_flow = frame.pc.pose.ego_to_global.inverse().transform_flow(
        # frame.flow.full_flow
        # )

        # First point cloud has no flow, use it as is
        points_compensated = frame.pc.flow(frame.flow).full_global_pc.points

        point_nodes[f"/frame_{i + 1}/lidar_points_compensated"] = (
            server.scene.add_point_cloud(
                name=f"/frame_{i + 1}/lidar_points_compensated",
                points=points_compensated,
                colors=point_colors,
                point_size=gui_point_size.value,
                visible=False,
                point_shape="rounded",
            )
        )

    @gui_flow_compensation.on_update
    def _(_) -> None:
        with server.atomic():
            # Toggle visibility.
            for i in range(num_frames):
                point_nodes[f"/frame_{i + 1}/lidar_points"].visible = not point_nodes[
                    f"/frame_{i + 1}/lidar_points"
                ].visible
                point_nodes[
                    f"/frame_{i + 1}/lidar_points_compensated"
                ].visible = not point_nodes[
                    f"/frame_{i + 1}/lidar_points_compensated"
                ].visible
        server.flush()  # Optional!

    while True:
        time.sleep(10.0)


if __name__ == "__main__":
    main()
