"""
Stage 2: Multi-Task Classifier (MobileNetV3-Small Architecture)
- Head A: 11-Class Species Classifier (10 Indonesian market species + unknown)
- Head B: CIELab-invariant Freshness Grader (continuous eye & gill score 0.0 - 1.0)
Size budget: < 10 MB INT8/FP16 quantized.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class SqueezeExcitation(nn.Module):
    def __init__(self, in_c: int, squeeze_factor: int = 4):
        super().__init__()
        squeezed = max(1, in_c // squeeze_factor)
        self.fc1 = nn.Conv2d(in_c, squeezed, 1)
        self.fc2 = nn.Conv2d(squeezed, in_c, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class InvertedResidual(nn.Module):
    """MobileNetV3 Inverted Residual block."""
    def __init__(self, in_c: int, exp_c: int, out_c: int, kernel_size: int, stride: int, use_se: bool):
        super().__init__()
        self.use_res_connect = stride == 1 and in_c == out_c
        layers = []
        if exp_c != in_c:
            layers.extend([
                nn.Conv2d(in_c, exp_c, 1, bias=False),
                nn.BatchNorm2d(exp_c),
                nn.Hardswish(inplace=True)
            ])
        layers.extend([
            nn.Conv2d(exp_c, exp_c, kernel_size, stride, kernel_size // 2, groups=exp_c, bias=False),
            nn.BatchNorm2d(exp_c),
            nn.Hardswish(inplace=True)
        ])
        if use_se:
            layers.append(SqueezeExcitation(exp_c))
        layers.extend([
            nn.Conv2d(exp_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.block(x)
        return self.block(x)


class MobileFeatureExtractor(nn.Module):
    """Compact feature extractor based on MobileNetV3-Small."""
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )
        self.blocks = nn.Sequential(
            InvertedResidual(16, 16, 16, 3, 2, True),    # 56x56
            InvertedResidual(16, 72, 24, 3, 2, False),   # 28x28
            InvertedResidual(24, 88, 24, 3, 1, False),
            InvertedResidual(24, 96, 40, 5, 2, True),    # 14x14
            InvertedResidual(40, 240, 40, 5, 1, True),
            InvertedResidual(40, 120, 48, 5, 1, True),
            InvertedResidual(48, 144, 48, 5, 1, True),
            InvertedResidual(48, 288, 96, 5, 2, True),   # 7x7
            InvertedResidual(96, 576, 96, 5, 1, True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, 1, bias=False),
            nn.BatchNorm2d(576),
            nn.Hardswish(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x.flatten(1)  # [B, 576]


class FishGuardMultiTaskClassifier(nn.Module):
    """
    Unified Multi-Task Edge Classifier:
    1. Species Head (RGB): 11 classes (10 species + unknown).
    2. Freshness Head (CIELab): Predicts continuous biological freshness (0.0 to 1.0).
    """

    SPECIES_CLASSES = [
        "tongkol", "kembung", "cakalang", "bandeng", "kakap",
        "gurame", "nila", "lele", "kerapu", "tenggiri", "unknown"
    ]

    def __init__(self, num_species: int = 11):
        super().__init__()
        self.num_species = num_species

        # Feature Extractors
        self.species_backbone = MobileFeatureExtractor(in_channels=3)
        self.freshness_backbone = MobileFeatureExtractor(in_channels=3)

        # Species Head
        self.species_head = nn.Sequential(
            nn.Linear(576, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_species)
        )

        # Freshness Head (outputs continuous scalar in [0.0, 1.0])
        self.freshness_head = nn.Sequential(
            nn.Linear(576, 128),
            nn.Hardswish(inplace=True),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        whole_fish: torch.Tensor,
        eye_crop_lab: torch.Tensor,
        gill_crop_lab: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for complete inspection:
        - whole_fish: [B, 3, 224, 224] RGB
        - eye_crop_lab: [B, 3, 224, 224] CIELab
        - gill_crop_lab: [B, 3, 224, 224] CIELab
        """
        # Species Classification
        species_feat = self.species_backbone(whole_fish)
        species_logits = self.species_head(species_feat)

        # Eye Freshness Grading
        eye_feat = self.freshness_backbone(eye_crop_lab)
        eye_score = self.freshness_head(eye_feat)

        # Gill Freshness Grading
        gill_feat = self.freshness_backbone(gill_crop_lab)
        gill_score = self.freshness_head(gill_feat)

        return {
            "species_logits": species_logits,
            "eye_score": eye_score,
            "gill_score": gill_score
        }

    def predict_species(self, whole_fish_rgb: torch.Tensor) -> Dict[str, Any]:
        """
        Species classification inference with confidence threshold check.
        """
        self.eval()
        with torch.no_grad():
            feat = self.species_backbone(whole_fish_rgb)
            logits = self.species_head(feat)
            probs = F.softmax(logits, dim=-1)[0]
            top_prob, top_idx = torch.max(probs, dim=0)

            score = float(top_prob.item())
            predicted_label = self.SPECIES_CLASSES[top_idx.item()]

            # Fallback if below 0.70 confidence
            if score < 0.70 and predicted_label != "unknown":
                label = "unknown"
                effective_score = score
            else:
                label = predicted_label
                effective_score = score

            return {
                "label": label,
                "confidence": round(effective_score, 4),
                "top_prob": round(score, 4),
                "raw_label": predicted_label
            }

    def predict_freshness(
        self,
        eye_crop_lab: torch.Tensor,
        gill_crop_lab: torch.Tensor
    ) -> Dict[str, float]:
        """
        Freshness scoring for cropped anatomical regions.
        """
        self.eval()
        with torch.no_grad():
            eye_feat = self.freshness_backbone(eye_crop_lab)
            eye_score = float(self.freshness_head(eye_feat)[0, 0].item())

            gill_feat = self.freshness_backbone(gill_crop_lab)
            gill_score = float(self.freshness_head(gill_feat)[0, 0].item())

            return {
                "eye_score": round(eye_score, 4),
                "gill_score": round(gill_score, 4)
            }


class StandaloneSpeciesClassifier(nn.Module):
    """Export-ready single-input species classifier model."""
    def __init__(self, base_model: FishGuardMultiTaskClassifier):
        super().__init__()
        self.backbone = base_model.species_backbone
        self.head = base_model.species_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        logits = self.head(feat)
        return F.softmax(logits, dim=-1)


class StandaloneFreshnessGrader(nn.Module):
    """Export-ready single-input freshness score model for CIELab crops."""
    def __init__(self, base_model: FishGuardMultiTaskClassifier):
        super().__init__()
        self.backbone = base_model.freshness_backbone
        self.head = base_model.freshness_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.head(feat)
