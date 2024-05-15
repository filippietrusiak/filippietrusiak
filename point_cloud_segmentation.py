"""
3D point cloud segmentation for industrial and geospatial environments.
Designed for Hexagon AB's survey and inspection pipelines.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn as nn


@dataclass
class PointCloudSegResult:
    labels: np.ndarray          # per-point class labels
    confidences: np.ndarray     # per-point confidence scores
    class_names: List[str]
    num_points: int
    num_classes: int


class SharedMLP(nn.Module):
    """1D convolution-based shared MLP for point-wise feature extraction."""

    def __init__(self, channels: List[int], bn: bool = True):
        super().__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv1d(channels[i], channels[i + 1], 1))
            if bn:
                layers.append(nn.BatchNorm1d(channels[i + 1]))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PointNetSegHead(nn.Module):
    """
    PointNet-style segmentation network for 3D point clouds.
    Input: (B, 3, N) XYZ coordinates (optionally + intensity)
    Output: (B, num_classes, N) per-point logits
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 8):
        super().__init__()
        self.local_feat = SharedMLP([in_channels, 64, 128, 256])
        self.global_pool = nn.AdaptiveMaxPool1d(1)
        self.seg_mlp = SharedMLP([256 + 256, 256, 128, num_classes], bn=False)
        self.global_proj = SharedMLP([256, 256])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        local_feat = self.local_feat(x)                         # (B, 256, N)
        global_feat = self.global_pool(local_feat)              # (B, 256, 1)
        global_feat = self.global_proj(global_feat)             # (B, 256, 1)
        global_feat_exp = global_feat.expand(-1, -1, N)         # (B, 256, N)
        concat = torch.cat([local_feat, global_feat_exp], dim=1)  # (B, 512, N)
        logits = self.seg_mlp(concat)                           # (B, num_classes, N)
        return logits


class PointCloudPreprocessor:
    """Normalizes and voxel-downsamples raw LiDAR/survey point clouds."""

    def __init__(self, voxel_size: float = 0.05, max_points: int = 16384):
        self.voxel_size = voxel_size
        self.max_points = max_points

    def normalize(self, pts: np.ndarray) -> np.ndarray:
        """Center to origin and scale to unit sphere."""
        centroid = pts[:, :3].mean(axis=0)
        pts = pts.copy()
        pts[:, :3] -= centroid
        scale = np.linalg.norm(pts[:, :3], axis=1).max()
        if scale > 0:
            pts[:, :3] /= scale
        return pts

    def voxel_downsample(self, pts: np.ndarray) -> np.ndarray:
        """Simple voxel grid downsampling."""
        voxel_idx = np.floor(pts[:, :3] / self.voxel_size).astype(np.int32)
        _, unique_idx = np.unique(voxel_idx, axis=0, return_index=True)
        return pts[unique_idx]

    def subsample(self, pts: np.ndarray) -> np.ndarray:
        if len(pts) > self.max_points:
            idx = np.random.choice(len(pts), self.max_points, replace=False)
            return pts[idx]
        return pts

    def process(self, pts: np.ndarray) -> np.ndarray:
        pts = self.voxel_downsample(pts)
        pts = self.subsample(pts)
        pts = self.normalize(pts)
        return pts


CLASS_NAMES = ["ground", "vegetation", "building", "vehicle", "pole",
               "wire", "facade", "clutter"]


class IndustrialPointCloudSegmenter:
    def __init__(self, weights_path: Optional[str] = None, in_channels: int = 3):
        self.model = PointNetSegHead(in_channels=in_channels, num_classes=len(CLASS_NAMES))
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.model.eval()
        self.preprocessor = PointCloudPreprocessor()

    def segment(self, pts: np.ndarray) -> PointCloudSegResult:
        """
        pts: (N, 3) or (N, 4) numpy array (XYZ or XYZI)
        """
        processed = self.preprocessor.process(pts)
        tensor = torch.FloatTensor(processed[:, :3].T).unsqueeze(0)  # (1, 3, N)
        with torch.no_grad():
            logits = self.model(tensor)[0]          # (num_classes, N)
            probs = torch.softmax(logits, dim=0).numpy()
        labels = probs.argmax(axis=0)
        confidences = probs.max(axis=0)
        return PointCloudSegResult(
            labels=labels,
            confidences=confidences,
            class_names=CLASS_NAMES,
            num_points=len(processed),
            num_classes=len(CLASS_NAMES),
        )
