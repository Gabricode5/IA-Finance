from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class TreeNode:
    positive_rate: float
    samples: int
    positives: int
    feature: str | None = None
    threshold: float | None = None
    left: "TreeNode | None" = None
    right: "TreeNode | None" = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.left is not None:
            data["left"] = self.left.to_dict()
        if self.right is not None:
            data["right"] = self.right.to_dict()
        return data


@dataclass
class ScalingStat:
    feature: str
    clip_low: float
    clip_high: float
    mean: float
    std: float


@dataclass
class LogisticModel:
    features: list[str]
    weights: list[float]
    bias: float
    epochs: int
    learning_rate: float
    positive_weight: float


@dataclass
class GaussianNBModel:
    features: list[str]
    class_priors: dict[str, float]
    means: dict[str, list[float]]
    variances: dict[str, list[float]]