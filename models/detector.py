"""
Stage 1: Lightweight Object Detector (YOLO-Nano Backbone)
Detects and localizes fish_eye and fish_gill with normalized bounding boxes [x, y, w, h].
Designed for ultra-fast, on-device mobile execution (< 6 MB INT8 quantized).
"""

import math
from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ConvBNAct(nn.Module):
    """Convolution + BatchNorm2d + SiLU activation."""
    def __init__(self, in_c: int, out_c: int, k: int = 3, s: int = 1, p: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Mobile-efficient Depthwise Separable Convolution."""
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.dw = ConvBNAct(in_c, in_c, k=3, s=stride, p=1, groups=in_c)
        self.pw = nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class C2fBlock(nn.Module):
    """Compact cross-stage partial bottleneck block (YOLOv8/11 style)."""
    def __init__(self, in_c: int, out_c: int, n: int = 1):
        super().__init__()
        c_mid = out_c // 2
        self.cv1 = ConvBNAct(in_c, out_c, k=1, s=1, p=0)
        self.cv2 = ConvBNAct(in_c + c_mid * n, out_c, k=1, s=1, p=0)
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c_mid, c_mid, k=3, s=1, p=1),
                ConvBNAct(c_mid, c_mid, k=3, s=1, p=1)
            ) for _ in range(n)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        for b in self.bottlenecks:
            y.append(b(y[-1]))
        return self.cv2(torch.cat(y, 1))


class FishEyeGillDetector(nn.Module):
    """
    Lightweight Nano Detector specifically trained for fish head anatomical landmarks:
    Class 0: fish_eye
    Class 1: fish_gill
    Input shape: [B, 3, 320, 320]
    Outputs: [B, 6, 2100] (4 box coords [cx, cy, w, h] + 2 class logits per anchor)
    """

    def __init__(self, num_classes: int = 2, input_size: int = 320):
        super().__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.class_names = ["fish_eye", "fish_gill"]

        # Backbone (Stem + 4 Downsamples)
        self.stem = ConvBNAct(3, 16, k=3, s=2, p=1)  # 160x160
        self.stage1 = nn.Sequential(
            ConvBNAct(16, 32, k=3, s=2, p=1),        # 80x80
            C2fBlock(32, 32, n=1)
        )
        self.stage2 = nn.Sequential(
            ConvBNAct(32, 64, k=3, s=2, p=1),        # 40x40 (P3)
            C2fBlock(64, 64, n=2)
        )
        self.stage3 = nn.Sequential(
            ConvBNAct(64, 128, k=3, s=2, p=1),       # 20x20 (P4)
            C2fBlock(128, 128, n=2)
        )
        self.stage4 = nn.Sequential(
            ConvBNAct(128, 256, k=3, s=2, p=1),      # 10x10 (P5)
            C2fBlock(256, 256, n=1)
        )

        # Decoupled Multi-scale Detection Heads (P3: 40x40, P4: 20x20, P5: 10x10)
        # P3: 1600 anchors, P4: 400 anchors, P5: 100 anchors -> Total 2100 anchors
        channels = [64, 128, 256]
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c, 64, k=3, s=1, p=1),
                nn.Conv2d(64, 4, kernel_size=1)  # [dx, dy, dw, dh]
            ) for c in channels
        ])
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                ConvBNAct(c, 64, k=3, s=1, p=1),
                nn.Conv2d(64, num_classes, kernel_size=1)  # [logit_eye, logit_gill]
            ) for c in channels
        ])

        self._generate_anchors(input_size)

    def _generate_anchors(self, input_size: int):
        strides = [8, 16, 32]
        anchors = []
        anchor_strides = []
        for s in strides:
            grid_h = input_size // s
            grid_w = input_size // s
            shift_x = (torch.arange(0, grid_w) + 0.5) * s
            shift_y = (torch.arange(0, grid_h) + 0.5) * s
            y, x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            grid = torch.stack([x.reshape(-1), y.reshape(-1)], dim=-1)
            anchors.append(grid)
            anchor_strides.append(torch.full((grid.shape[0], 1), s, dtype=torch.float32))

        self.register_buffer("anchors", torch.cat(anchors, dim=0))  # [2100, 2]
        self.register_buffer("anchor_strides", torch.cat(anchor_strides, dim=0))  # [2100, 1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw concatenated output:
        Shape: [B, 4 + num_classes, 2100] -> [B, 6, 2100]
        """
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)

        feats = [p3, p4, p5]
        reg_outs = []
        cls_outs = []

        for f, reg_h, cls_h in zip(feats, self.reg_heads, self.cls_heads):
            b, _, h, w = f.shape
            reg = reg_h(f).permute(0, 2, 3, 1).reshape(b, -1, 4)
            cls = cls_h(f).permute(0, 2, 3, 1).reshape(b, -1, self.num_classes)
            reg_outs.append(reg)
            cls_outs.append(cls)

        all_reg = torch.cat(reg_outs, dim=1)  # [B, 2100, 4]
        all_cls = torch.cat(cls_outs, dim=1)  # [B, 2100, 2]

        # Combine into [B, 6, 2100] for standard ONNX/TFLite export
        out = torch.cat([all_reg, all_cls], dim=-1).permute(0, 2, 1)
        return out

    def decode_predictions(
        self,
        output: torch.Tensor,
        conf_threshold: float = 0.35,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Decodes raw output into normalized bounding boxes [x_min, y_min, w, h]
        with class labels and confidence scores.
        """
        # output shape: [1, 6, 2100]
        preds = output[0].permute(1, 0)  # [2100, 6]
        reg = preds[:, :4]
        cls_logits = preds[:, 4:]
        cls_probs = torch.sigmoid(cls_logits)

        # Box decoding:
        # anchors: [2100, 2] (cx_anchor, cy_anchor)
        # strides: [2100, 1]
        strides = self.anchor_strides
        anchors = self.anchors

        cx = anchors[:, 0:1] + reg[:, 0:1] * strides
        cy = anchors[:, 1:2] + reg[:, 1:2] * strides
        w = torch.exp(reg[:, 2:3].clamp(-5, 5)) * strides * 2.0
        h = torch.exp(reg[:, 3:4].clamp(-5, 5)) * strides * 2.0

        # Normalized coordinates [0..1]
        norm_cx = cx / self.input_size
        norm_cy = cy / self.input_size
        norm_w = w / self.input_size
        norm_h = h / self.input_size

        x_min = (norm_cx - norm_w / 2).clamp(0.0, 1.0)
        y_min = (norm_cy - norm_h / 2).clamp(0.0, 1.0)
        box_w = norm_w.clamp(0.01, 1.0)
        box_h = norm_h.clamp(0.01, 1.0)

        results = []
        for class_idx, class_name in enumerate(self.class_names):
            scores = cls_probs[:, class_idx]
            mask = scores >= conf_threshold

            if not mask.any():
                continue

            sub_scores = scores[mask]
            sub_boxes = torch.cat([x_min[mask], y_min[mask], box_w[mask], box_h[mask]], dim=-1)

            # Sort by score descending
            order = torch.argsort(sub_scores, descending=True)
            keep = []
            while order.numel() > 0:
                i = order[0].item()
                keep.append(i)
                if order.numel() == 1:
                    break
                # Compute IoU between first box and rest
                b1 = sub_boxes[i]
                b_rest = sub_boxes[order[1:]]

                # IoU calculation for [x, y, w, h]
                b1_x2, b1_y2 = b1[0] + b1[2], b1[1] + b1[3]
                br_x2, br_y2 = b_rest[:, 0] + b_rest[:, 2], b_rest[:, 1] + b_rest[:, 3]

                inter_x1 = torch.max(b1[0], b_rest[:, 0])
                inter_y1 = torch.max(b1[1], b_rest[:, 1])
                inter_x2 = torch.min(b1_x2, br_x2)
                inter_y2 = torch.min(b1_y2, br_y2)

                inter_w = (inter_x2 - inter_x1).clamp(min=0)
                inter_h = (inter_y2 - inter_y1).clamp(min=0)
                inter_area = inter_w * inter_h

                b1_area = b1[2] * b1[3]
                br_area = b_rest[:, 2] * b_rest[:, 3]
                union_area = b1_area + br_area - inter_area
                iou = inter_area / union_area.clamp(min=1e-6)

                order = order[1:][iou <= iou_threshold]

            for idx in keep:
                box = sub_boxes[idx].tolist()
                results.append({
                    "class": class_name,
                    "confidence": float(sub_scores[idx].item()),
                    "bbox": [round(c, 4) for c in box]  # [x, y, w, h]
                })

        return results
