# Practical Examination & Viva Voce Guide

> **Project Title:** Topology-Aware Deep Learning for Small Intracranial Aneurysm Detection  
> **Institution:** Dayananda Sagar University (DSU), Department of Computer Science & Engineering  
> **Academic Year:** 2026–2027 | Major Project  
> **Project Guide:** Prof. Sharath H A  
> **Team Members:** Manya B A, Mohammed Aseel Vanti, Nisarga Jessy P, Pavan P Yadav  

---

## Part 1: Written Examination Sheet Content

*(Write the following three sections in the Practical Examination Sheet.)*

### 1. Project Abstract

Intracranial aneurysms carry a significant risk of aneurysmal subarachnoid haemorrhage upon rupture, making reliable non-invasive diagnosis clinically important. However, automated detection of **small ($\le 5\text{ mm}$)** and **very small ($< 3\text{ mm}$)** aneurysms in 3D Time-of-Flight Magnetic Resonance Angiography (TOF-MRA) and Computed Tomography Angiography (CTA) remains challenging because small lesions share a similar local appearance with normal arterial bifurcations and bends, and 41.1% of aneurysms in our dataset are $\le 5\text{ mm}$. Furthermore, training a single unified convolutional model on both TOF-MRA and CTA creates a cross-modality trade-off due to differing imaging physics (flow-related enhancement vs. X-ray Hounsfield-unit attenuation and dense skull-base bone).

To address this, we propose a **Topology-Aware Dual-Expert Multimodal Pipeline** combining a **3D ResNet-18** visual encoder with a **Graph Attention Network (GATv2)** over the 3D arterial centerline graph. Binary arterial vessel masks are thinned via 3D medial-axis skeletonization into a geometric graph whose nodes encode normalized 3D coordinates, local vessel radius, degree, node type, radius-to-degree ratio, and intensity, and whose edges encode segment length and 3D unit tangent direction. Local 3-hop subgraphs are encoded by GATv2 and fused with $64 \times 64 \times 64$ visual patch embeddings via multi-head **Cross-Attention** to predict aneurysm probability, 3D centroid offset, and physical diameter. By routing scans by DICOM modality to modality-specific visual backbones (`best_fusion_mra.pt` and `best_fusion_cta.pt`) sharing the same graph encoder architecture, we achieved validation AUROCs of **0.8653 (MRA)** and **0.8971 (CTA)**, with **1.86 mm** diameter MAE on CTA. On a Center-4 CTA benchmark (8 scans, 10 lesions), the CTA expert achieved **72.9% sensitivity at 4 FP/scan** (**85.9%** on lesions $\le 5\text{ mm}$), compared to **26.0%** for the unified Option C model and **0%** for the 3D ResNet-18 baseline. In an ablation on one healthy TOF-MRA scan, confining candidate generation to the arterial vessel mask reduced the search space by **3.85×**, detections ($\ge 0.78$) from **43 to 12** (removing extracranial false alarms), and single-core CPU runtime from **877.3 s to 133.8 s**. An interactive **Gradio Clinical Workstation** provides synchronized multiplanar reconstruction (MPR), PHASES/ISUIA risk stratification, and exportable diagnostic reports.

---

### 2. Proposed Modules

1. **Data Preprocessing & Modality Normalization Module (`src/data/`)**
   - Performs patient-wise stratified splitting across center, imaging modality, and aneurysm presence (293 train / 62 validation / 61 test across 416 scans and 387 lesions) and physical size profiling ($< 3\text{ mm}$, $3\text{--}5\text{ mm}$, $> 5\text{ mm}$).
   - Resamples volumes to $0.5\text{ mm}$ isotropic resolution and applies modality-specific normalization: **TOF-MRA** clips **non-zero voxels** at the $0.5\text{th}$ and $99.5\text{th}$ percentiles followed by $Z$-score standardization; **CTA** windows intensities to $[100, 700]\text{ HU}$ and rescales linearly to $[0, 1]$.

2. **Vascular Topology & Graph Construction Module (`src/topology/`)**
   - **Skeletonizer (`skeletonizer.py`):** Performs 3D medial-axis thinning on binary vessel masks and prunes spurious branches shorter than 3 voxels.
   - **Graph Builder (`graph_builder.py`):** Converts the 26-connected skeleton into a PyTorch Geometric graph $G = (V, E)$ with 8-D node features (normalized coordinates, Euclidean Distance Transform radius, normalized degree, node type flag, radius-to-degree ratio, and intensity) and 4-D edge features (segment length in mm and 3D unit tangent vector).
   - **Subgraph Extractor (`subgraph_extractor.py`):** Extracts a 3-hop local arterial subgraph (up to 64 nodes, ~20–30 mm of local artery) around the closest skeleton node to each candidate.

3. **Dual-Expert Multimodal Detection & Fusion Module (`src/models/`)**
   - **3D Visual Backbone (`resnet3d.py`):** 3D ResNet-18 encoder producing a 512-D visual embedding $\mathbf{z}_{\text{vis}}$ from $64 \times 64 \times 64$ patches.
   - **Topological Encoder (`gat_module.py`):** 3-layer, 4-head edge-conditioned `GATv2Conv` encoder with mean+max pooling producing a 256-D embedding $\mathbf{z}_{\text{topo}}$.
   - **Cross-Attention Fusion & Multi-Task Heads (`dual_branch_detector.py`, `losses.py`):** Multi-head cross-attention ($\mathbf{z}_{\text{vis}}$ as query, $\mathbf{z}_{\text{topo}}$ as key/value) feeding linear heads for aneurysm probability (Binary Focal Loss), 3D centroid offset (Smooth L1), and maximum diameter in mm (Smooth L1).
   - **Dual-Expert Training (`train_fusion.py`):** Trains modality-specific visual experts (`best_fusion_mra.pt` and `best_fusion_cta.pt`) to avoid cross-modality trade-offs.

4. **Full-Scan Clinical Inference & FROC Evaluation Module (`src/inference/`)**
   - **Candidate Proposer (`candidate_proposer.py`):** Samples candidate sites along the arterial skeleton at all bifurcations and centerline waypoints (stride 6, or stride 3 on vessels with radius $\le 2\text{ mm}$), merging candidates within $2\text{ mm}$ and using vessel-mask conditioning (with a $>98\text{th}$-percentile intensity fallback when no mask is provided).
   - **3D Non-Maximum Suppression (`nms3d.py`):** Merges overlapping detections within a $6.0\text{ mm}$ neighborhood via weighted averaging of position/diameter and max probability.
   - **FROC & Generalization Evaluator (`froc_evaluator.py`, `test_generalization.py`):** Evaluates scan-level FROC sensitivity ($6\text{ mm}$ hit radius) across $0.5, 1, 2, 4\text{ FP/scan}$, stratified by lesion size, and benchmarks Center-4 CTA performance.

5. **Interactive Clinical Decision Support UI & Reporting Module (`src/ui/`)**
   - **Slice & MIP Visualizer (`slice_visualizer.py`):** Renders synchronized Axial, Coronal, and Sagittal MPR cross-sections and whole-brain Maximum Intensity Projections (MIP) with modality-specific windowing.
   - **Clinical Report Generator (`report_generator.py`):** Computes PHASES/ISUIA rupture risk stratification, maps anatomical arterial sites, and exports CSV and printable HTML diagnostic reports.
   - **Web Workstation (`app.py`):** Gradio interface supporting automated modality routing, curated clinical presets, and custom NIfTI uploads.

---

### 3. Conclusion

A topology-aware detector combining modality-specific 3D ResNet-18 visual experts with a shared `GATv2` arterial centerline graph encoder was designed and evaluated for small intracranial aneurysm detection in TOF-MRA and CTA. Using modality-specific visual backbones resolved the cross-modality trade-off seen in unified training, achieving validation AUROCs of **0.8653 (MRA)** and **0.8971 (CTA)**, **1.86 mm** diameter MAE on CTA (**3.43 mm** on MRA), and raising Center-4 CTA sensitivity at 4 FP/scan to **72.9%** overall and **85.9%** on small ($\le 5\text{ mm}$) lesions (compared to **26.0%** for the unified Option C model and **0%** for the 3D ResNet-18 baseline). On a healthy TOF-MRA scan, arterial vessel-mask conditioning reduced the search space by **3.85×**, detections ($\ge 0.78$) from **43 to 12** (removing 31 extracranial false positives, with 12 remaining at arterial junctions), and single-core CPU runtime from **877.3 s to 133.8 s**. Coupled with an interactive Gradio clinical workstation for MPR visualization and PHASES/ISUIA reporting, the pipeline provides a structured foundation for computer-aided aneurysm screening. Key limitations include small scan-level validation subsets (8 Center-4 CTA scans and 8 MRA scans), Center-4 scans being present in the patient-level training split rather than a true external test, and reliance on provided vessel masks; future work will focus on leave-one-centre-out retraining, held-out test-set evaluation, and automated 3D vessel segmentation.

---

## Part 2: Equal Individual Contribution Split (4 Students)

> **Design Principle:** Every team member contributed equally across **core technical implementation (Phases 4–6 & integration)**, **testing & benchmarking**, and **documentation / reports / research paper**. No member was limited to only documentation or only coding.

### Summary Matrix

| Team Member | Core Engineering Focus (Phases 4–6 + Integration) | Testing & Benchmarking | Reports, Diagrams & Research Paper |
| :--- | :--- | :--- | :--- |
| **Pavan P Yadav** | Dual-Expert Cross-Attention Fusion (`dual_branch_detector.py`, `train_fusion.py`), Data/Baseline foundation (`dataset.py`, `resnet3d.py`, `losses.py`), UI Expert Router (`app.py`) | `test_model_baseline.py`, `test_fusion_model.py`, MRA vs. CTA modality ablation | IEEE paper (Methodology & Dual-Expert Routing), `modality_report.md`, `artifacts/architecture_diagrams/` |
| **Manya B A** | 3D Vessel Skeletonization (`skeletonizer.py`), PyG Arterial Graph Builder (`graph_builder.py`), GATv2 Topological Encoder (`gat_module.py`) | `test_topology.py`, graph cache profiling, topology vs. pure-CNN feature ablation | IEEE paper (Vascular Graph Construction & Geometric DL), `ROADMAP.md` (Phases 4–5), `phase5_work_done.png` & `phase5_results.png` |
| **Mohammed Aseel Vanti** | 3-Hop Subgraph Extractor (`subgraph_extractor.py`), Full-Scan Candidate Proposer (`candidate_proposer.py`), 3D-NMS (`nms3d.py`), Vessel-Mask Conditioning | `test_inference.py`, single-core CPU runtime profiling (`877.3 s → 133.8 s`), extracranial FP forensics | IEEE paper (Candidate Generation & Mask Conditioning Ablation), `comparison_with_without_vessel_mask.md`, `phase6_work_done.png` & `phase6_results.png` |
| **Nisarga Jessy P** | FROC Evaluator (`froc_evaluator.py`), Center-4 Cross-Centre Benchmark (`test_generalization.py`), Clinical UI Visualizer & Risk Reporter (`slice_visualizer.py`, `report_generator.py`) | `test_ui.py`, `test_data_pipeline.py`, size-stratified FROC tables (`<3mm`, `3–5mm`, `>5mm`) | IEEE paper (Dataset, Size-Stratified FROC Results & Limitations), `results.md`, `walkthrough.md` |

---

### Detailed Individual Breakdown (For Viva Voce)

#### 1. Pavan P Yadav
- **Technical Implementation (Phases 4–6 & Project Integration):**
  - Built the patient-stratified data loader and modality-specific preprocessor (`src/data/dataset.py`, `src/data/preprocessor.py` using non-zero percentile clipping for MRA and `[100, 700]` HU windowing for CTA), plus the 3D ResNet-18 baseline and multi-task loss (`src/models/resnet3d.py`, `src/models/losses.py`).
  - Implemented the **Cross-Attention Multimodal Fusion** block in `src/models/dual_branch_detector.py` (visual queries attending to topological keys/values) and the **Dual-Expert training pipeline** in `src/models/train_fusion.py` (`best_fusion_mra.pt` and `best_fusion_cta.pt`).
  - Integrated dynamic modality routing into the Gradio workstation (`src/ui/app.py`).
- **Testing & Benchmarking:** Authored `tests/test_model_baseline.py` and `tests/test_fusion_model.py`; ran modality-specific vs. unified training benchmarks.
- **Reports & Paper:** Co-authored the IEEE paper (Abstract, Cross-Modality Interference, and Dual-Expert Methodology), authored `modality_report.md`, and created system/design architecture diagrams (`artifacts/architecture_diagrams/`).
- **Viva Talking Points:** Why unified MRA+CTA training creates a cross-modality trade-off (flow enhancement vs. skull-base bone attenuation), how Cross-Attention fuses 3D CNN and GATv2 embeddings, and how Focal + Smooth L1 multi-task losses work.

#### 2. Manya B A
- **Technical Implementation (Phases 4–5):**
  - Developed **3D Medial-Axis Skeletonization** (`src/topology/skeletonizer.py`) to extract 1-voxel-thick arterial centerlines from binary vessel masks and prune branches shorter than 3 voxels.
  - Implemented the **PyTorch Geometric Arterial Graph Builder** (`src/topology/graph_builder.py`), computing 8-D node features (normalized 3D coordinates, EDT radius, normalized degree, node type flag, radius-to-degree ratio, and voxel intensity) and 4-D edge features (segment length and 3D unit tangent).
  - Built the **GATv2 Graph Encoder** (`src/models/gat_module.py`) using 3-layer, 4-head edge-conditioned `GATv2Conv` layers with global mean+max pooling to produce 256-D topological embeddings.
- **Testing & Benchmarking:** Authored `tests/test_topology.py`, validated graph construction across real TopAneu scans, and benchmarked topological vs. pure visual representations.
- **Reports & Paper:** Co-authored the IEEE paper (Vascular Topology & Graph Construction sections), maintained `ROADMAP.md` Phases 4–5, and created Phase 5 presentation slides (`artifacts/presentation_phase5_phase6/phase5_work_done.png`, `phase5_results.png`).
- **Viva Talking Points:** How 3D medial-axis thinning and 26-connectivity degree classification work, what the 8 node and 4 edge features encode, and how `GATv2Conv` aggregates neighborhood topology.

#### 3. Mohammed Aseel Vanti
- **Technical Implementation (Phases 4 & 6):**
  - Implemented the **3-Hop Local Subgraph Extractor** (`src/topology/subgraph_extractor.py`) mapping 3D patch centers to up to 64 neighboring skeleton nodes (~20–30 mm of local artery).
  - Built the **Full-Scan Candidate Proposer** (`src/inference/candidate_proposer.py`) sampling arterial bifurcations and adaptive-stride centerline waypoints (stride 6, or 3 on vessels $\le 2\text{ mm}$ radius, merged within $2\text{ mm}$) and **3D Non-Maximum Suppression** (`src/inference/nms3d.py`, $6.0\text{ mm}$ neighborhood).
  - Executed the **3D Arterial Vessel-Mask Conditioning** ablation on a healthy TOF-MRA scan, measuring a $3.85\times$ search-space reduction ($1,096,366 \rightarrow 284,503$ voxels), candidate reduction (~3,200 $\rightarrow$ 910), false-positive reduction ($43 \rightarrow 12$ detections $\ge 0.78$), and $6.5\times$ single-core CPU speedup ($877.3\text{ s} \rightarrow 133.8\text{ s}$).
- **Testing & Benchmarking:** Authored `tests/test_inference.py`, profiled single-core CPU latency, and analyzed extracranial vs. arterial-junction false positives.
- **Reports & Paper:** Co-authored the IEEE paper (Candidate Generation & Mask Conditioning Ablation), authored `comparison_with_without_vessel_mask.md`, and created Phase 6 presentation slides (`artifacts/presentation_phase5_phase6/phase6_work_done.png`, `phase6_results.png`).
- **Viva Talking Points:** Why intensity thresholding (`>98th` percentile) triggers extracranial false positives (scalp fat, orbital fat, bone marrow), how 3D-NMS merges duplicate detections, and what the 12 remaining junction detections reveal.

#### 4. Nisarga Jessy P
- **Technical Implementation (Phases 6–7):**
  - Built the **Size-Stratified FROC Evaluator** (`src/inference/froc_evaluator.py`) measuring sensitivity ($6\text{ mm}$ hit radius) at $0.5, 1, 2, 4\text{ FP/scan}$ across $< 3\text{ mm}$, $3\text{--}5\text{ mm}$, $\le 5\text{ mm}$, and $> 5\text{ mm}$ cohorts.
  - Implemented **Center-4 CTA Benchmark Evaluation** (`src/inference/test_generalization.py`) on 8 Center-4 scans with 10 lesions (achieving 72.9% sensitivity at 4 FP/scan and 85.9% on $\le 5\text{ mm}$ lesions).
  - Developed the **Clinical Visualization & Risk Reporting Engine** (`src/ui/slice_visualizer.py` for synchronized Axial/Coronal/Sagittal MPR and MIP views; `src/ui/report_generator.py` for PHASES/ISUIA rupture risk scoring and HTML/CSV export).
- **Testing & Benchmarking:** Authored `tests/test_ui.py` and `tests/test_data_pipeline.py`; compiled FROC comparison tables and Center-4 benchmark metrics (`artifacts/froc_eval/`, `artifacts/generalization/`).
- **Reports & Paper:** Co-authored the IEEE paper (Dataset, Size-Stratified FROC Results & Limitations), compiled `results.md` and `walkthrough.md`, and curated clinical demo presets for external presentation.
- **Viva Talking Points:** How FROC evaluation differs from patch-level AUROC, why size-stratified ($<3\text{ mm}$, $3\text{--}5\text{ mm}$) evaluation matters clinically, why Center-4 is a cross-centre benchmark rather than a true external test, and how PHASES/ISUIA risk scoring works.
