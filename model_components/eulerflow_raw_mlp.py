import enum
import typing
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch._tensor import Tensor

from .nsfp_raw_mlp import ActivationFn, NSFPRawMLP


class QueryDirection(enum.Enum):
    FORWARD = 1
    REVERSE = -1


@dataclass
class ModelFlowResult:
    flow: torch.Tensor  # N x 3


class BaseEncoder(ABC, torch.nn.Module):
    @abstractmethod
    def encode(
        self, pc: Tensor, idx: int, total_entries: int, query_direction: QueryDirection
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    def forward(self, entries: tuple[Tensor, int, int, QueryDirection]) -> Tensor:
        (pc, idx, total_entries, query_direction) = entries
        return self.encode(pc, idx, total_entries, query_direction)


class SimpleEncoder(BaseEncoder):
    def _make_time_feature(
        self, idx: int, total_entries: int, device: torch.device
    ) -> torch.Tensor:
        # Make the time feature zero mean
        if total_entries <= 1:
            # Handle divide by zero
            return torch.tensor([0.0], dtype=torch.float32, device=device)
        max_idx = total_entries - 1
        return torch.tensor([(idx / max_idx) - 0.5], dtype=torch.float32, device=device)

    def encode(
        self, pc: Tensor, idx: int, total_entries: int, query_direction: QueryDirection
    ) -> Tensor:
        assert pc.shape[1] == 3, f"Expected 3, but got {pc.shape[1]}"

        assert pc.dim() == 2, f"Expected 2, but got {pc.dim()}"
        assert isinstance(query_direction, QueryDirection), (
            f"Expected QueryDirection, but got {query_direction}"
        )

        time_feature = self._make_time_feature(idx, total_entries, pc.device)  # 1x1

        direction_feature = torch.tensor(
            [query_direction.value], dtype=torch.float32, device=pc.device
        )  # 1x1
        pc_time_dim = time_feature.repeat(pc.shape[0], 1).contiguous()
        pc_direction_dim = direction_feature.repeat(pc.shape[0], 1).contiguous()

        # Concatenate into a feature tensor
        return torch.cat(
            [pc, pc_time_dim, pc_direction_dim],
            dim=-1,
        )

    def __len__(self):
        return 5  # point + time + direction


class EulerFlowMLP(NSFPRawMLP):
    def __init__(
        self,
        output_dim: int = 3,
        latent_dim: int = 128,
        act_fn: ActivationFn = ActivationFn.RELU,
        num_layers: int = 8,
        encoder: BaseEncoder = SimpleEncoder(),
    ):
        super().__init__(
            input_dim=len(encoder),
            output_dim=output_dim,
            latent_dim=latent_dim,
            act_fn=act_fn,
            num_layers=num_layers,
        )
        self.nn_layers = torch.compile(torch.nn.Sequential(encoder, self.nn_layers))  # pyright: ignore[reportArgumentType]

    @typing.no_type_check
    def forward(
        self,
        pc: torch.Tensor,
        idx: int,
        total_entries: int,
        query_direction: QueryDirection,
    ) -> ModelFlowResult:
        entries = (pc, idx, total_entries, query_direction)
        res = self.nn_layers(entries)
        return ModelFlowResult(flow=res)
