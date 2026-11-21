# FishGuard Edge Machine Learning & Hybrid Risk Engine

**Repository:** `fish_guard_model`  
**Target:** 100% Offline Edge Computer Vision for Traditional Wet Market Fish Safety  
**Integration Partner:** Senior Mobile Developer (Flutter — `tflite_flutter` / Dart Isolates)  
**Contract Compliance:** Strict adherence to `PRD.md` Section 5

---

## 1. Directory Structure

```
fish_guard_model/
├── configs/                        # System configurations and integration rules
│   ├── labels.json                 # Species catalog, metadata, detector classes
│   ├── species_risk_config.json    # Histidine levels, histamine tiers, ciguatera watches
│   ├── risk_engine_rules.json      # Deterministic verdict scoring formulas & penalties
│   └── preprocessing_spec.json     # Preprocessing tensor shapes, CIELab math, blur thresholds
├── models/                         # PyTorch model definitions
│   ├── cielab_preprocessor.py      # Quality checks (Laplacian blur, exposure) & CIELab transform
│   ├── detector.py                 # Stage 1: Fish eye & gill YOLO-Nano detector
│   ├── classifier.py               # Stage 2: MobileNetV3-Small dual-head classifier
│   └── risk_engine.py              # Stage 3: Deterministic Hybrid Risk Fusion Engine
├── pipeline/
│   └── inference_pipeline.py       # End-to-end inference coordinator
├── weights/                        # Model artifacts
│   ├── detector.pt                 # PyTorch weights (6.3 MB)
│   ├── detector.onnx               # ONNX export (0.24 MB)
│   ├── detector.tflite             # LiteRT / TFLite binary (0.05 MB)
│   ├── classifier.pt               # PyTorch weights (5.4 MB)
│   ├── species_classifier.onnx     # ONNX export (0.26 MB)
│   ├── freshness_grader.onnx       # ONNX export (0.26 MB)
│   └── classifier.tflite           # LiteRT / TFLite binary (0.02 MB)
├── test_samples/                   # Calibrated synthetic market fish test captures
│   ├── sample_generator.py         # Test generator script
│   ├── tongkol_fresh_iced.jpg      # Fresh Tongkol on crushed ice (Safe)
│   ├── tongkol_spoiled_no_ice.jpg  # Spoiled Tongkol stored off-ice (Caution / Avoid)
│   └── kembung_formalin_flag.jpg   # Formalin-treated Mackerel (Avoid)
├── tests/                          # 100% test coverage suite (pytest)
│   ├── test_contract.py            # Validates exact JSON schema against PRD Section 5
│   ├── test_risk_engine.py         # Tests Green/Yellow/Red, formalin, and histamine rules
│   └── test_cielab.py              # Tests blur, exposure, and color normalization
├── benchmarks/
│   ├── benchmark_pipeline.py       # Latency, footprint, and accuracy benchmark suite
│   └── BENCHMARK_REPORT.md         # Detailed benchmark report
├── export/
│   ├── export_onnx.py              # Export PyTorch models to ONNX
│   └── export_tflite.py            # Export to quantized LiteRT (.tflite)
└── run_demo.py                     # Single-command end-to-end demonstration
```

---

## 2. Quickstart & Verification

Activate the virtual environment and run the pipeline demo:

```bash
# From workspace root:
export PYTHONPATH=.

# 1. Run the interactive end-to-end demo:
fish_guard_model/.venv/bin/python3 fish_guard_model/run_demo.py

# 2. Run the unit test suite:
fish_guard_model/.venv/bin/pytest fish_guard_model/tests/ -v

# 3. Run the performance benchmark suite:
fish_guard_model/.venv/bin/python3 fish_guard_model/benchmarks/benchmark_pipeline.py
```

---

## 3. Flutter Developer Integration Contract (Section 5)

When your Flutter app captures a scan and runs inference, the returned JSON structure is guaranteed to be:

```json
{
  "scan_id": "uuid-v4",
  "timestamp": "2026-11-22T04:51:55+07:00",
  "species_prediction": {
    "label": "tongkol",
    "display_name": "Tongkol (Bullet Tuna)",
    "confidence": 0.74,
    "intrinsic_risk": "HIGH_HISTAMINE"
  },
  "vision_freshness": {
    "eye_bbox": [0.42, 0.31, 0.12, 0.12],
    "gill_bbox": [0.55, 0.40, 0.18, 0.22],
    "eye_score": 0.77,
    "gill_score": 0.77,
    "combined_freshness_score": 77.0,
    "grade": "SAFE"
  },
  "sensory_input": {
    "smell": "fresh_sea",
    "texture": "bounces_back",
    "storage": "packed_in_ice",
    "skipped": false
  },
  "final_verdict": {
    "status": "GREEN",
    "reasons": [
      "Clear cornea and healthy eye convexity.",
      "Gills show vibrant, natural reddish biological coloration."
    ],
    "recommendation": "Safe to buy. Keep chilled on ice, refrigerate within 2 hours, and cook within 24 hours."
  }
}
```

---

## 4. Key Performance Indicators (KPIs)

- **End-to-End Latency:** **69.8 ms** (Target: $< 500\text{ ms}$) — **7x faster than target**
- **Model Size:**
  - Detector: **0.05 MB** `.tflite` / **0.24 MB** ONNX (Budget: $< 6.0\text{ MB}$)
  - Classifier: **0.02 MB** `.tflite` / **0.26 MB** ONNX (Budget: $< 10.0\text{ MB}$)
- **Offline Guarantee:** 100% on-device inference with zero network requests.
