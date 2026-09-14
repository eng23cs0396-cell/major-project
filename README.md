# Topology-Aware Deep Learning for Small Intracranial Aneurysm Detection

**Major Project Phase-I (2026–2027)**  
**Department of Computer Science and Engineering**  
**School of Engineering, Dayananda Sagar University (DSU), Bengaluru**

---

## 👥 Project Team & Supervision

- **Manya B A** (ENG23CS0360)
- **Mohammed Aseel Vanti** (ENG23CS0364)
- **Nisarga Jessy P** (ENG23CS0385)
- **Pavan P Yadav** (ENG23CS0396)

**Under the Supervision of:**  
**Prof. Sharath H A**, Assistant Professor, Department of CSE, Dayananda Sagar University

---

## 📌 Project Overview & Research Gap

Small intracranial aneurysms ($< 3\text{ mm}$ to $5\text{ mm}$) are challenging to distinguish from normal vascular bifurcations, junctions, and bends using volumetric image intensity features alone. Conventional 3D CNN architectures rely exclusively on local voxel intensities, failing to explicitly model the connectivity, branching structure, and topological context of the cerebral arterial tree.

This project investigates a **Topology-Aware Multimodal Deep Learning Architecture**:
1. **Volumetric Visual Feature Extraction**: 3D ResNet on 3D volumes of interest (VOIs) from CTA / TOF-MRA brain scans.
2. **Topological Feature Learning**: 3D vessel skeletonization &rarr; Vascular Graph construction ($G = (V, E)$) &rarr; **Graph Attention Network (GAT)** to model arterial connectivity, branching angles, and vessel caliber.
3. **Multimodal Feature Fusion**: Fuses 3D CNN appearance features with GAT structural context to detect and localize aneurysms across the entire brain scan.
4. **Hypothesis Evaluation**: Rigorous size-stratified benchmarking against image-only 3D CNN baselines, specifically measuring sensitivity and false-positive reduction on small aneurysms.

---

## 🏗️ System Architecture

```
Raw Full 3D Scan (.nii.gz)
         │
         ▼
3D Vessel Segmentation (3D U-Net)
         │
         ▼
Vascular Skeletonization & Graph Extraction (Nodes = Bifurcations, Edges = Vessels)
         │
         ├────────────────────────────────────────┐
         ▼                                        ▼
Branch A: 3D ResNet                      Branch B: Graph Attention Network (GAT)
(Volumetric Appearance Features)         (Vascular Topology & Connectivity Context)
         │                                        │
         └───────────────────┬────────────────────┘
                             ▼
                 Multimodal Feature Fusion
                             │
                             ▼
                 3D Multi-Task Detection Head
                 ├── Scan Diagnosis: Is there an aneurysm? (Yes / No)
                 ├── Precise 3D Coordinates: (x, y, z)
                 ├── Anatomical Vessel Branch Label
                 └── Predicted Lesion Size / Diameter (mm)
```

---

## 📂 Repository Structure

├── ROADMAP.md                # Detailed Phase-by-Phase Developer Roadmap & Guide for Teammates
├── walkthrough.md            # Live progress log & verification results for all completed phases
├── configs/                  # Hyperparameter, data splits & experiment configurations
├── src/
│   ├── data/                 # NIfTI loaders, patch extractors, augmentations
│   ├── topology/             # Skeletonization, graph builders, edge/node feature extractors
│   ├── models/               # 3D ResNet, GAT module, Multimodal Fusion network
│   ├── inference/            # Full-scan detector, 3D-NMS, FROC evaluation
│   └── utils/                # Metrics, visualization, coordinate transforms
├── README.md
└── .gitignore
```

---

## 📊 Dataset: TopAneu 2026

The project utilizes the **TopAneu 2026 Challenge Dataset**:
- **Total Scans**: 416 scans from 408 unique patients across 5 clinical institutions (Lausanne CHUV, Geneva HUG, Mie Chuo Japan, INSTED, and OpenNeuro).
- **Modalities**: TOF-MRA (307 scans) and CTA (109 scans).
- **Annotations**:
  - `location_masks/`: Multiclass 3D aneurysm segmentation masks.
  - `vessel_masks/`: Pre-computed 3D cerebral vessel masks.
  - `type_masks/`: Aneurysm morphology (saccular, fusiform, dissecting).
  - `location_jsons/`: Aneurysm anatomical location metadata.

> *Note: Due to file size constraints (20 GB), the raw dataset is excluded from Git tracking via `.gitignore`. Place the unzipped dataset inside the `dataset(topAneu)/` directory.*

---

## 🛠️ Tech Stack & Environment

- **Deep Learning**: PyTorch, PyTorch Geometric (PyG), MONAI
- **Medical Imaging**: SimpleITK, NiBabel, scikit-image
- **Hardware Target**: NVIDIA GeForce RTX 4060 GPU (8 GB VRAM) with CUDA acceleration
