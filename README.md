#  Drone Human Detection & Counting System
### Antlings Internship Program — Technical Assessment (AI/ML)



## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Task 1 — Dataset Understanding & Preprocessing](#task-1--dataset-understanding--preprocessing)
- [Task 2 — Model Training](#task-2--model-training)
- [Task 3 — Detection & Human Counting](#task-3--detection--human-counting)
- [Task 4 — Object Tracking (Bonus)](#task-4--object-tracking-bonus)
- [Task 5 — Evaluation & Metrics](#task-5--evaluation--metrics)
- [Results](#results)
- [Strengths & Limitations](#strengths--limitations)
- [Demo Video](#demo-video)

---

##  Project Overview

This project implements a full **computer vision pipeline** for analyzing drone and aerial imagery. The system is capable of:

- Detecting **humans** (pedestrians & people) and **vehicles** (cars, vans, trucks, buses) in aerial images and video
- Counting total **humans** per frame in real time
- Visualizing detections with **bounding boxes** and **class labels**
- Tracking individual objects across video frames using **ByteTrack**, assigning persistent unique IDs and drawing movement trails

The entire pipeline was built and trained on **Google Colab (Tesla T4 GPU)** using the **VisDrone 2019 detection dataset** and **YOLOv8s** as the base model.

---

##  Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FULL PIPELINE                                │
└─────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │   VisDrone   │────▶│  Dataset Analysis │────▶│  YOLO Format     │
  │  Dataset     │     │  & Visualization  │     │  (pre-converted) │
  └──────────────┘     └──────────────────┘     └────────┬─────────┘
                                                          │
                                                          ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                     PREPROCESSING                                │
  │  • Image resize to 640×640 (letterboxed)                        │
  │  • Mosaic augmentation (4-image combine)                        │
  │  • HSV shifts (hue, saturation, brightness)                     │
  │  • Horizontal flip (p=0.5)                                      │
  │  • Random erasing (p=0.4)                                       │
  │  • Albumentations: Blur, CLAHE, Grayscale                       │
  └────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                   MODEL TRAINING                                 │
  │  Base Model  : YOLOv8s (pretrained on COCO)                     │
  │  Fine-tuned  : VisDrone 2019 — 10 classes                       │
  │  Epochs      : 30                                               │
  │  Image size  : 640×640                                          │
  │  Batch size  : 16                                               │
  │  Optimizer   : AdamW (lr=0.000714)                              │
  │  Device      : Tesla T4 GPU                                     │
  │  Train set   : 6,471 images                                     │
  │  Val set     :   548 images                                     │
  └────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                     INFERENCE                                    │
  │                                                                  │
  │   Input Image/Video                                             │
  │         │                                                        │
  │         ▼                                                        │
  │   YOLOv8s (best.pt)  ──▶  Bounding Boxes + Class IDs + Conf    │
  │         │                                                        │
  │         ▼                                                        │
  │   Class Grouping                                                │
  │   ├── Human    : class 0 (pedestrian) + class 1 (people)        │
  │   └── Vehicle  : class 3 (car) + 4 (van) + 5 (truck) + 8 (bus) │
  │         │                                                        │
  │         ▼                                                        │
  │   Human Count Overlay + Bounding Box Visualization             │
  └────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │               OBJECT TRACKING (Bonus)                           │
  │                                                                  │
  │   Per-frame detections                                          │
  │         │                                                        │
  │         ▼                                                        │
  │   ByteTrack (sv.ByteTrack)                                      │
  │   ├── Assigns unique persistent Track IDs                       │
  │   ├── Maintains tracks across frames                            │
  │   └── Draws movement trails (TraceAnnotator)                    │
  │         │                                                        │
  │         ▼                                                        │
  │   Annotated output video (MP4, H.264)                          │
  └────────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
  ┌──────────────────────────────────────────────────────────────────┐
  │                     EVALUATION                                   │
  │  Precision · Recall · mAP@0.50 · mAP@0.50:0.95 · FPS          │
  └──────────────────────────────────────────────────────────────────┘
```

---

## Dataset

**VisDrone 2019 Detection Dataset**
- **Source:** [Kaggle — VisDrone Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)
- **Format:** Pre-converted YOLO format (`class cx cy w h`, normalized)
- **Train:** 6,471 images
- **Validation:** 548 images
- **Test-dev:** available (labels included)
- **Test-challenge:** images only (no labels)

### Class Distribution (Training Set)

| Class ID | Name | Instances |
|---|---|---|
| 0 | pedestrian | 79,337 |
| 1 | people | 27,059 |
| 2 | bicycle | 10,480 |
| 3 | car | 144,867 |
| 4 | van | 24,956 |
| 5 | truck | 12,875 |
| 6 | tricycle | 4,812 |
| 7 | awning-tricycle | 3,246 |
| 8 | bus | 5,926 |
| 9 | motor | 29,647 |

### Class Grouping for This Task

```python
HUMAN_CLASSES   = {0, 1}        # pedestrian + people
VEHICLE_CLASSES = {3, 4, 5, 8}  # car + van + truck + bus
```

---

##  Project Structure

```
drone-human-detection/
│
├── README.md
│
├── notebooks/
│   └── drone_detection_pipeline.ipynb   # Full Colab notebook
│
├── outputs/
│   ├── sample_visualizations.png        # Task 1 — dataset samples
│   ├── detection_results.png            # Task 3 — detection grid
│   ├── counting_chart.png               # Task 3 — count bar chart
│   ├── tracking_preview.png             # Task 4 — tracking frame grid
│   ├── evaluation_dashboard.png         # Task 5 — metrics dashboard
│   └── tracked_final.mp4                # Task 4 — tracking video
│
├── weights/
│   └── best.pt                          # Best trained YOLOv8s weights
│
└── visdrone.yaml                        # Dataset configuration
```

---

##  Setup & Installation

This project runs entirely on **Google Colab**. No local setup required.

### 1. Open Google Colab
Set runtime to **GPU (Tesla T4)**:
> Runtime → Change runtime type → T4 GPU

### 2. Install Dependencies

```python
!pip install ultralytics supervision --upgrade -q
import ultralytics
ultralytics.checks()
```

### 3. Download Dataset via Kaggle API

```python
# Upload your kaggle.json API key
from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d banuprasadb/visdrone-dataset
!unzip -q visdrone-dataset.zip -d /content/visdrone
```

### 4. Verified Dataset Structure

```
visdrone/
└── VisDrone_Dataset/
    ├── VisDrone2019-DET-train/
    │   ├── images/
    │   └── labels/
    ├── VisDrone2019-DET-val/
    │   ├── images/
    │   └── labels/
    ├── VisDrone2019-DET-test-dev/
    │   ├── images/
    │   └── labels/
    └── VisDrone2019-DET-test-challenge/
        └── images/        ← no labels
```

---

##  Task 1 — Dataset Understanding & Preprocessing

### Dataset Analysis

```python
from collections import Counter
from pathlib import Path

train_lbl = Path("/content/visdrone/VisDrone_Dataset/VisDrone2019-DET-train/labels")
all_cats  = []

for lbl_file in train_lbl.glob("*.txt"):
    with open(lbl_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                all_cats.append(int(parts[0]))

print(Counter(all_cats))
```

### Preprocessing Steps Applied

| Step | Method | Details |
|---|---|---|
| Format | Pre-converted YOLO | No manual conversion needed |
| Resize | Letterboxing | 640×640, aspect ratio preserved |
| Mosaic | Enabled (p=1.0) | 4 images combined — critical for small objects |
| Flip | Horizontal (p=0.5) | Left-right mirror |
| HSV Hue | ±0.015 | Subtle color shift |
| HSV Saturation | ±0.7 | Color richness variation |
| HSV Value | ±0.4 | Brightness variation |
| Random Erasing | p=0.4 | Simulates occlusion |
| Scale | 0.5 | Random zoom in/out |
| Translation | 0.1 | Random positional shift |
| Albumentations | p=0.01 each | Blur, MedianBlur, CLAHE, Grayscale |
| Mosaic close | Epoch 20 | Disabled in final 10 epochs for stability |
| Duplicate removal | Automatic | 4 duplicate labels removed by YOLO scanner |

### Dataset Challenges

| Challenge | Description | Impact |
|---|---|---|
| **Tiny objects** | People often 5–15px wide at 640 res | Low mAP50-95 (0.213) |
| **Class imbalance** | Car (144K) vs awning-tricycle (3.2K) = 44× difference | Rare class mAP < 0.10 |
| **Dense crowds** | Hundreds of overlapping objects per image | NMS merges nearby detections |
| **Occlusion** | Objects behind trees, buildings, each other | Low overall recall (0.386) |
| **Scale variation** | Drone altitude varies → same object = different sizes | Requires multi-scale generalization |
| **Lighting variation** | Shadows, overexposure, time-of-day changes | HSV + CLAHE augmentation helps |
| **Class ambiguity** | "pedestrian" vs "people" overlap | People recall only 0.295 |

---

##  Task 2 — Model Training

### Dataset YAML

```yaml
path: /content/visdrone/VisDrone_Dataset
train: VisDrone2019-DET-train/images
val:   VisDrone2019-DET-val/images
nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
```

### Training Configuration

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")   # pretrained on COCO

model.train(
    data    = "/content/visdrone.yaml",
    epochs  = 30,
    imgsz   = 640,
    batch   = 16,
    device  = 0,
    workers = 2,
    patience= 10,
    plots   = True,
    project = "/content/runs",
    name    = "visdrone_yolov8"
)
```

### Training Details

| Parameter | Value |
|---|---|
| Base model | YOLOv8s (11.1M parameters) |
| Pretrained | COCO (ImageNet weights) |
| Optimizer | AdamW — lr=0.000714, momentum=0.9 |
| Epochs | 30 |
| Image size | 640×640 |
| Batch size | 16 |
| GPU | Tesla T4 (14.9 GB VRAM) |
| Training time | ~1.35 hours |
| AMP | Enabled (Automatic Mixed Precision) |

### Training Progress (Key Epochs)

| Epoch | mAP@0.50 | mAP@0.50:0.95 | Box Loss | Cls Loss |
|---|---|---|---|---|
| 1 | 0.204 | 0.112 | 1.685 | 1.750 |
| 10 | 0.312 | 0.175 | 1.488 | 1.057 |
| 20 | 0.355 | 0.202 | 1.404 | 0.939 |
| 30 | 0.369 | 0.213 | 1.286 | 0.821 |

---

##  Task 3 — Detection & Human Counting

### Detection Function

```python
def detect_and_count(image_path, conf=0.3, iou=0.5):
    img     = cv2.imread(str(image_path))
    results = model(img, conf=conf, iou=iou, verbose=False)[0]

    boxes   = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)
    confs   = results.boxes.conf.cpu().numpy()

    human_count   = int(sum(c in HUMAN_CLASSES   for c in classes))
    vehicle_count = int(sum(c in VEHICLE_CLASSES for c in classes))

    # Draw boxes + count overlay
    ...
    return annotated_image, human_count, vehicle_count
```

### Counting Logic

- **Humans** = detections where `class_id ∈ {0, 1}` (pedestrian + people)
- **Vehicles** = detections where `class_id ∈ {3, 4, 5, 8}` (car + van + truck + bus)
- Simple per-frame sum — no temporal smoothing
- Count displayed in black overlay box on every output image/frame

---

##  Task 4 — Object Tracking (Bonus)

### Tracker: ByteTrack (via Supervision 0.28.0)
i recorded a video of the street in front of my house from my balcony and used the video for object tracking.

```python
import supervision as sv

tracker = sv.ByteTrack(
    track_activation_threshold = 0.3,
    lost_track_buffer          = 30,
    minimum_matching_threshold = 0.8,
    frame_rate                 = int(fps)
)

box_annotator   = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=30)
```

### Tracking Loop

```python
while True:
    ret, frame = cap.read()
    if not ret: break

    results    = model(frame, conf=0.3, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [f"{CLASS_NAMES[cls]} #{tid}"
              for cls, tid in zip(detections.class_id, detections.tracker_id)]

    annotated = trace_annotator.annotate(frame, detections)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)
    ...
```

### Tracking Features

| Feature | Description |
|---|---|
| Unique Track IDs | Each object assigned persistent integer ID |
| Re-identification | Lost tracks kept alive for 30 frames |
| Movement trails | 30-frame history trail drawn per object |
| Per-frame count | Human and vehicle count overlaid each frame |
| Output | MP4 video re-encoded with H.264 (ffmpeg) |

---

## 📈 Task 5 — Evaluation & Metrics

### Overall Results

| Metric | Value |
|---|---|
| **Precision** | 0.496 (49.6%) |
| **Recall** | 0.386 (38.6%) |
| **mAP @ IoU=0.50** | 0.369 (36.9%) |
| **mAP @ IoU=0.50:0.95** | 0.213 (21.3%) |
| **Avg Inference Time** | ~2.6 ms/image |
| **FPS (Tesla T4 GPU)** | ~38 FPS |

### Per-Class Breakdown

| Class | Precision | Recall | mAP@0.50 |
|---|---|---|---|
| pedestrian | 0.472 | 0.422 | 0.395 |
| people | 0.524 | 0.295 | 0.295 |
| bicycle | 0.251 | 0.177 | 0.109 |
| **car** | **0.669** | **0.789** | **0.774** |
| van | 0.521 | 0.428 | 0.419 |
| truck | 0.536 | 0.368 | 0.356 |
| tricycle | 0.462 | 0.256 | 0.250 |
| awning-tricycle | 0.342 | 0.184 | 0.145 |
| bus | 0.684 | 0.498 | 0.532 |
| motor | 0.502 | 0.437 | 0.410 |

### What the Metrics Mean

**Precision (49.6%)** — Of every bounding box the model drew, ~50% were correct detections. The other 50% were false alarms, largely from small ambiguous objects at altitude.

**Recall (38.6%)** — The model successfully found ~39% of all real objects in the validation set. Many tiny/occluded objects were missed entirely.

**mAP@0.50 (36.9%)** — Standard aerial detection benchmark. A predicted box counts as correct if it overlaps ground truth by ≥50%. This score is competitive for a 30-epoch YOLOv8s run on VisDrone.

**mAP@0.50:0.95 (21.3%)** — Stricter metric averaging across IoU thresholds 0.50→0.95. Penalizes imprecise box boundaries heavily. Published VisDrone benchmarks for small models typically score 0.10–0.25 — our result sits comfortably in this range.

**FPS (~38)** — Well above the real-time threshold of 25–30 FPS, making the system suitable for live drone feed processing.

---

## Strengths & Limitations

### Strengths

- **Strong vehicle detection** — Car mAP50 of 0.774 is production-grade quality
- **Real-time capable** — ~38 FPS on T4 GPU exceeds real-time threshold
- **Robust augmentation** — Mosaic + HSV + erasing significantly helps small object generalization
- **Persistent tracking** — ByteTrack maintains stable IDs even through brief occlusions
- **End-to-end pipeline** — From raw dataset to tracked video output in a single notebook

### Limitations

- **Small object recall is low** — Humans that are <10px wide are frequently missed
- **Class imbalance** — Rare classes (bicycle, awning-tricycle) perform poorly due to few training examples
- **Simple counting logic** — Per-frame sum with no temporal smoothing; count fluctuates frame to frame
- **30 epochs only** — Longer training (50–100 epochs) would meaningfully improve mAP
- **No crowd counting** — Heavily overlapping groups of people are undercounted due to NMS merging

### Challenges Faced

- Supervision library API change (`ByteTracker` → `ByteTrack`) required version-specific handling
- VisDrone's extreme class imbalance (44× car vs awning-tricycle) required careful YAML class mapping
- Dataset YAML initially set with wrong `nc` value — corrected by scanning actual label files
- Video output required ffmpeg H.264 re-encoding for browser/notebook playback compatibility

---


##  Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.12 | Core language |
| PyTorch | 2.10.0+cu128 | Deep learning backend |
| Ultralytics | 8.4.50 | YOLOv8 training & inference |
| Supervision | 0.28.0 | ByteTrack + annotation tools |
| OpenCV | 4.x | Image & video processing |
| Matplotlib | 3.x | Visualization & plots |
| FFmpeg | system | Video re-encoding (H.264) |
| Google Colab | — | Training platform (Tesla T4 GPU) |







