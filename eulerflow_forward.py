import time
from pathlib import Path

import numpy as np
import torch
import viser
from bucketed_scene_flow_eval.datasets import Argoverse2CausalSceneFlow
from bucketed_scene_flow_eval.datastructures.dataclasses import EgoLidarFlow
from einops import repeat

from model_components.eulerflow_raw_mlp import EulerFlowMLP, QueryDirection


class ModelWrapper(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model = EulerFlowMLP(num_layers=18)


def transform_pc(pc: "torch.Tensor", transform: "torch.Tensor") -> "torch.Tensor":
    import torch

    """
    Transform an Nx3 point cloud by a 4x4 transformation matrix.
    """

    homogenious_pc = torch.cat(
        (pc, torch.ones((pc.shape[0], 1), device=pc.device)), dim=1
    )
    homogenious_pc = homogenious_pc @ transform.T
    return homogenious_pc[:, :3]


def global_to_ego_flow(
    global_full_pc: "torch.Tensor",
    global_warped_full_pc: "torch.Tensor",
    global_to_ego: "torch.Tensor",
) -> "torch.Tensor":
    ego_full_pc0 = transform_pc(global_full_pc, global_to_ego)
    ego_warped_full_pc0 = transform_pc(global_warped_full_pc, global_to_ego)

    return ego_warped_full_pc0 - ego_full_pc0


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
        load_flow=False,
        use_cache=True,
        subsequence_length=num_frames,
    )
    point_nodes: list[viser.PointCloudHandle] = []
    flow_nodes: list[viser.LineSegmentsHandle] = []
    subsequence = av2_dataset[0]
    for i, frame in enumerate(subsequence):
        ego_points = frame.pc.full_ego_pc.points

        # Run inference
        log_id = "7c696d35-e34f-38b0-b4b4-e88803ad1f6a"
        model_checkpoint_path = Path(
            f"/Users/ikhatri/Data/argoverse2/eulerflow_train_ckpts/{log_id}/best_weights.pth"
        )
        checkpoint = torch.load(
            model_checkpoint_path, map_location="cpu", weights_only=False
        )
        model = ModelWrapper()
        model.load_state_dict(checkpoint["model"])
        global_flow_result = model.model.forward(
            pc=torch.tensor(
                frame.pc.full_global_pc.points, device="cpu", dtype=torch.float32
            ),
            idx=i,
            total_entries=num_frames,
            query_direction=QueryDirection.FORWARD,
        )

        # Convert flow result from global frame to ego frame
        global_flowed_pc = (
            torch.tensor(frame.pc.full_global_pc.points) + global_flow_result.flow
        )
        ego_flow = (
            global_to_ego_flow(
                torch.tensor(frame.pc.full_global_pc),
                torch.tensor(global_flowed_pc),
                torch.inverse(torch.tensor(frame.pc.pose.ego_to_global.to_array())),
            )
            .detach()
            .numpy()
        )

        # ego_flow = frame.pc.pose.ego_to_global.inverse().transform_flow(
        #     global_flow_result.flow.detach().numpy()
        # )

        point_colors = repeat(
            np.array([0, 0, 255]).astype(np.uint8), "c -> n c", n=ego_points.shape[0]
        )
        # flow_ego_obj = EgoLidarFlow(
        #     full_flow=global_flow_result.flow.detach().numpy(), mask=frame.pc.mask
        # )
        flowed_points = frame.pc.full_ego_pc.flow(ego_flow).points
        flow_start_color = repeat(
            np.array([0, 0, 255]).astype(np.uint8), "c -> n c", n=ego_points.shape[0]
        )
        flow_end_color = repeat(
            np.array([255, 0, 0]).astype(np.uint8), "c -> n c", n=ego_points.shape[0]
        )
        flow_colors = np.stack([flow_start_color, flow_end_color], axis=1)
        point_nodes.append(
            server.scene.add_point_cloud(
                name=f"/frame_{i + 1}/lidar_points",
                points=ego_points,
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
                points=np.stack([ego_points, flowed_points], axis=1),
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
    # log_id = "7c696d35-e34f-38b0-b4b4-e88803ad1f6a"
    # model_checkpoint_path = Path(
    #     f"/Users/ikhatri/Data/argoverse2/eulerflow_train_ckpts/{log_id}/best_weights.pth"
    # )
    # checkpoint = torch.load(
    #     model_checkpoint_path, map_location="cpu", weights_only=False
    # )
    main()
