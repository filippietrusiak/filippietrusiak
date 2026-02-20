"""
Computer vision inspection pipeline for industrial surface and structural analysis.
Built for Hexagon AB's automated quality control workflows.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image


@dataclass
class DefectDetection:
    defect_type: str
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    confidence: float
    severity: str  # 'low', 'medium', 'high'


@dataclass
class InspectionResult:
    pass_inspection: bool
    defects: List[DefectDetection]
    overall_score: float   # 0.0 (worst) to 1.0 (best)
    image_quality: str
    warnings: List[str] = field(default_factory=list)


DEFECT_CLASSES = ["crack", "corrosion", "delamination", "scratch", "void", "clean"]
SEVERITY_MAP = {"crack": "high", "corrosion": "high", "delamination": "medium",
                "scratch": "low", "void": "medium", "clean": "low"}


class ImageQualityChecker:
    """Fast pre-flight check: blur, exposure, and contrast assessment."""

    def assess(self, img: np.ndarray) -> Tuple[str, List[str]]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        warnings = []
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = gray.mean()
        contrast = gray.std()
        if blur_score < 50:
            warnings.append(f"Image too blurry (Laplacian var={blur_score:.1f})")
        if mean_brightness < 40:
            warnings.append("Image underexposed")
        elif mean_brightness > 215:
            warnings.append("Image overexposed")
        if contrast < 15:
            warnings.append("Low contrast")
        quality = "good" if not warnings else ("acceptable" if len(warnings) == 1 else "poor")
        return quality, warnings


class DefectDetector(nn.Module):
    """Lightweight CNN for surface defect classification."""

    def __init__(self, num_classes: int = len(DEFECT_CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class SlidingWindowExtractor:
    """Extracts overlapping patches from large inspection images."""

    def __init__(self, patch_size: int = 128, stride: int = 64):
        self.patch_size = patch_size
        self.stride = stride

    def extract(self, img: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        h, w = img.shape[:2]
        patches = []
        for y in range(0, h - self.patch_size + 1, self.stride):
            for x in range(0, w - self.patch_size + 1, self.stride):
                patch = img[y:y + self.patch_size, x:x + self.patch_size]
                patches.append((patch, (x, y)))
        return patches


class IndustrialInspectionPipeline:
    """
    Full inspection pipeline: quality check → patch extraction → defect detection → report.
    """
    CONFIDENCE_THRESHOLD = 0.55
    PASS_SCORE_THRESHOLD = 0.75

    def __init__(self, weights_path: Optional[str] = None):
        self.detector = DefectDetector()
        if weights_path:
            self.detector.load_state_dict(torch.load(weights_path, map_location="cpu"))
        self.detector.eval()
        self.quality_checker = ImageQualityChecker()
        self.extractor = SlidingWindowExtractor()
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _classify_patch(self, patch: np.ndarray) -> Tuple[str, float]:
        pil = Image.fromarray(cv2.cvtColor(patch, cv2.COLOR_BGR2RGB))
        tensor = self.transform(pil).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.detector(tensor), dim=-1)[0].numpy()
        idx = probs.argmax()
        return DEFECT_CLASSES[idx], float(probs[idx])

    def inspect(self, image_path: str) -> InspectionResult:
        img = cv2.imread(image_path)
        if img is None:
            return InspectionResult(False, [], 0.0, "unknown", [f"Cannot read: {image_path}"])
        quality, warnings = self.quality_checker.assess(img)
        patches = self.extractor.extract(img)
        defects = []
        clean_count = 0
        for patch, (px, py) in patches:
            label, conf = self._classify_patch(patch)
            if label == "clean":
                clean_count += 1
            elif conf >= self.CONFIDENCE_THRESHOLD:
                defects.append(DefectDetection(
                    defect_type=label,
                    bbox=(px, py, self.extractor.patch_size, self.extractor.patch_size),
                    confidence=round(conf, 4),
                    severity=SEVERITY_MAP.get(label, "low"),
                ))
        total = len(patches) or 1
        overall_score = round(clean_count / total, 4)
        high_severity = sum(1 for d in defects if d.severity == "high")
        pass_inspection = overall_score >= self.PASS_SCORE_THRESHOLD and high_severity == 0
        return InspectionResult(
            pass_inspection=pass_inspection,
            defects=defects,
            overall_score=overall_score,
            image_quality=quality,
            warnings=warnings,
        )
