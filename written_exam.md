# Practical Examination & Viva Voce Guide

> **Project Title:** Topology-Aware Deep Learning (Dual-Expert 3D CNN + GATv2) for Small Intracranial Aneurysm Detection in TOF-MRA and CTA  
> **Institution:** Dayananda Sagar University (DSU), Department of Computer Science & Engineering  
> **Academic Year:** 2026–2027 | Major Project  
> **Project Guide:** Prof. Sharath H A  
> **Team Members:** Manya B A, Mohammed Aseel Vanti, Nisarga Jessy P, Pavan P Yadav  

---

## Part 1: Written Examination Sheet Content

*(Write the following three sections in the Practical Examination Sheet.)*

### 1. Project Abstract

Intracranial aneurysms carry a high risk of fatal subarachnoid hemorrhage upon rupture, making early non-invasive detection critical. However, automated detection of **small ($\le 5\text{ mm}$)** and **very small ($< 3\text{ mm}$)** aneurysms in 3D Time-of-Flight Magnetic Resonance Angiography (TOF-MRA) and Computed Tomography Angiography (CTA) remains challenging. Standard 3D Convolutional Neural Networks (CNNs) suffer from high false-positive rates because small aneurysms visually resemble normal tortuous bends and arterial bifurcations, and unified models trained jointly on MRA and CTA experience severe cross-modality negative transfer due to conflicting imaging physics (flow-related enhancement vs. Hounsfield-unit bone attenuation).

To address these challenges, we propose a **Topology-Aware Dual-Expert Multimodal Framework** with **3D Arterial Vessel Mask Conditioning**. First, the cerebral vasculature is converted into a 3D medial-axis skeleton and represented as a topological graph where nodes encode spatial coordinates, local vessel radius, tortuosity, and degree centrality, and edges encode segment length, tangent direction, and bifurcation angles. A **Graph Attention Network (GATv2)** extracts structural embeddings that are fused with **3D ResNet-18** visual patch features via **Cross-Attention** to predict aneurysm probability, 3D centroid offsets, and physical diameter. Second, to eliminate cross-modality interference, an automated modality router dispatches scans to dedicated **MRA (`best_fusion_mra.pt`)** and **CTA (`best_fusion_cta.pt`)** expert backbones while sharing the topological reasoning module. Finally, vessel-mask conditioning restricts candidate generation strictly to the arterial lumen, eliminating out-of-distribution non-cerebral false alarms (scalp fat, bone marrow, venous sinuses). On zero-shot Japan Center-4 CTA holdout evaluation, the Dual-Expert CTA specialist achieves **85.9% sensitivity** on small ($\le 5\text{ mm}$) aneurysms and **0.8971 AUROC** (MRA specialist: **0.8653 AUROC**), while vessel-mask conditioning reduces false positives by **72.1%** and accelerates inference by **6.5×**. An interactive **Gradio Clinical Workstation** provides synchronized multiplanar reconstruction (MPR), PHASES/ISUIA rupture risk stratification, and exportable diagnostic reports.

---

### 2. Proposed Modules

1. **Data Preprocessing & Modality Normalization Module (`src/data/`)**
   - Performs patient-stratified dataset splitting (train, validation, test, and Center-4 zero-shot holdout) and physical aneurysm size profiling ($< 3\text{ mm}$, $3\text{--}5\text{ mm}$, $> 5\text{ mm}$).
   - Applies modality-specific intensity normalization: brain-masked $0.5\text{--}99.5$ percentile clipping and $Z$-score standardization for **TOF-MRA**, and vascular windowing ($[100, 700]\text{ HU}$) for **CTA**.

2. **Vascular Topology & Graph Construction Module (`src/topology/`)**
   - **Skeletonizer (`skeletonizer.py`):** Applies 3D medial-axis thinning on binary arterial vessel masks and prunes spurious micro-branches.
   - **Graph Builder (`graph_builder.py`):** Converts 3D skeletons into PyTorch Geometric graphs $G = (V, E)$ with node features (coordinates, Euclidean Distance Transform radius, tortuosity, degree) and edge features (length, unit tangent, bifurcation angle).
   - **Subgraph Extractor (`subgraph_extractor.py`):** Extracts local $k$-hop arterial subgraphs around 3D candidate coordinates.

3. **Dual-Expert Multimodal Detection & Fusion Module (`src/models/`)**
   - **3D Visual Backbone (`resnet3d.py`):** 3D ResNet-18 encoder extracting $512$-D visual embeddings from $64 \times 64 \times 64$ volumetric patches.
   - **Topological Encoder (`gat_module.py`):** Multi-layer `GATv2Conv` network with global mean+max pooling producing $256$-D topological embeddings.
   - **Cross-Attention Fusion & Multi-Task Heads (`dual_branch_detector.py`, `losses.py`):** Fuses visual queries with topological keys/values to jointly predict aneurysm probability (Focal Loss), 3D centroid offset (Smooth L1), and physical diameter in mm (Smooth L1).
   - **Dual-Expert Training (`train_fusion.py`):** Trains modality-dedicated specialists (`best_fusion_mra.pt` and `best_fusion_cta.pt`) to eliminate cross-modality negative transfer.

4. **Full-Scan Clinical Inference & FROC Evaluation Module (`src/inference/`)**
   - **Candidate Proposer (`candidate_proposer.py`):** Samples interrogation points along arterial bifurcations and high-curvature skeleton points using vessel-mask conditioning.
   - **3D Non-Maximum Suppression (`nms3d.py`):** Clusters overlapping detections within a $6.0\text{ mm}$ radius, retaining the highest-confidence prediction.
   - **FROC & Generalization Evaluator (`froc_evaluator.py`, `test_generalization.py`):** Computes size-stratified Free-Response ROC sensitivity at $0.5\text{--}4.0\text{ FP/scan}$ and evaluates zero-shot cross-center CTA generalization.

5. **Interactive Clinical Decision Support UI & Reporting Module (`src/ui/`)**
   - **Slice & MIP Visualizer (`slice_visualizer.py`):** Renders synchronized Axial, Coronal, and Sagittal MPR views and whole-brain Maximum Intensity Projections (MIP) with modality-specific window/leveling.
   - **Clinical Report Generator (`report_generator.py`):** Computes PHASES/ISUIA rupture risk stratification, maps anatomical arterial sites, and exports CSV and printable HTML diagnostic reports.
   - **Web Workstation (`app.py`):** Gradio interface supporting automated modality routing, curated clinical presets, and custom NIfTI volume uploads.

---

### 3. Conclusion

The proposed **Topology-Aware Dual-Expert Multimodal Framework** successfully overcomes the two primary bottlenecks in automated intracranial aneurysm detection: high false-positive rates on small ($\le 5\text{ mm}$) lesions and cross-modality performance collapse between TOF-MRA and CTA. By fusing 3D ResNet-18 visual features with `GATv2` arterial topology representations via Cross-Attention, the system distinguishes true pathological outpouchings from normal vascular bifurcations. Decoupling the visual backbones into dedicated MRA and CTA experts eliminates negative transfer, lifting zero-shot Center-4 CTA sensitivity on small ($\le 5\text{ mm}$) aneurysms from **0.0%** (baseline 3D CNN) and **12.0%** (unified fusion) to **85.9%**, while achieving **0.8971 AUROC** (CTA) and **0.8653 AUROC** (MRA) with sub-$2\text{ mm}$ diameter regression accuracy. Furthermore, 3D arterial vessel-mask conditioning shrinks the search space by **3.85×**, cuts false positives by **72.1%**, eradicates non-cerebral out-of-distribution artifacts, and speeds up full-scan inference by **6.5×**. Integrated into an interactive Gradio clinical workstation with MPR visualization and PHASES/ISUIA risk reporting, the system provides a reliable, interpretable, and clinically deployable decision-support tool. Future work will focus on integrating automated end-to-end 3D vessel segmentation for unsegmented raw scans and prospective multi-hospital clinical validation.

---

## Part 2: Equal Individual Contribution Split (4 Students)

> **Design Principle:** Every team member contributed equally across **core technical implementation (Phases 4–6 & integration)**, **testing & benchmarking**, and **documentation / reports / research paper**. No member was limited to only documentation or only coding.

### Summary Matrix

| Team Member | Core Engineering Focus (Phases 4–6 + Integration) | Testing & Benchmarking | Reports, Diagrams & Research Paper |
| :--- | :--- | :--- | :--- |
| **Pavan P Yadav** | Dual-Expert Cross-Attention Fusion (`dual_branch_detector.py`, `train_fusion.py`), Data/Baseline foundation (`dataset.py`, `resnet3d.py`, `losses.py`), UI Expert Router (`app.py`) | `test_model_baseline.py`, `test_fusion_model.py`, MRA vs. CTA modality ablation | `draft_paper.md` (Methodology & Dual-Expert Architecture), `modality_report.md`, `artifacts/architecture_diagrams/` |
| **Manya B A** | 3D Vessel Skeletonization (`skeletonizer.py`), PyG Arterial Graph Builder (`graph_builder.py`), GATv2 Topological Encoder (`gat_module.py`) | `test_topology.py`, graph cache profiling, topology vs. pure-CNN feature ablation | `draft_paper.md` (Vascular Graph Extraction & Geometric DL), `ROADMAP.md` (Phases 4–5), `phase5_work_done.png` & `phase5_results.png` |
| **Mohammed Aseel Vanti** | $k$-Hop Subgraph Extractor (`subgraph_extractor.py`), Full-Scan Candidate Proposer (`candidate_proposer.py`), 3D-NMS (`nms3d.py`), Vessel-Mask Conditioning | `test_inference.py`, full-volume latency profiling (~28–32 s/scan), OOD false-positive forensics | `draft_paper.md` (Vessel Mask Conditioning & Ablation), `comparison_with_without_vessel_mask.md`, `phase6_work_done.png` & `phase6_results.png` |
| **Nisarga Jessy P** | FROC Evaluator (`froc_evaluator.py`), Zero-Shot Center-4 Generalization (`test_generalization.py`), Clinical UI Visualizer & Risk Reporter (`slice_visualizer.py`, `report_generator.py`) | `test_ui.py`, `test_data_pipeline.py`, size-stratified FROC tables (`<3mm`, `3–5mm`, `>5mm`) | `draft_paper.md` (Experiments, FROC Results & Clinical Discussion), `results.md`, `walkthrough.md` |

---

### Detailed Individual Breakdown (For Viva Voce)

#### 1. Pavan P Yadav
- **Technical Implementation (Phases 4–6 & Project Integration):**
  - Built the patient-stratified data loader and modality-specific preprocessor (`src/data/dataset.py`, `src/data/preprocessor.py`), plus the 3D ResNet-18 baseline and multi-task loss (`src/models/resnet3d.py`, `src/models/losses.py`).
  - Implemented the **Cross-Attention Multimodal Fusion** block in `src/models/dual_branch_detector.py` (visual queries attending to topological keys/values) and the **Dual-Expert training pipeline** in `src/models/train_fusion.py` (`best_fusion_mra.pt` and `best_fusion_cta.pt`) to resolve cross-modality negative transfer.
  - Integrated dynamic modality routing into the Gradio workstation (`src/ui/app.py`).
- **Testing & Benchmarking:** Authored `tests/test_model_baseline.py` and `tests/test_fusion_model.py`; ran modality-isolated vs. unified training benchmarks.
- **Reports & Paper:** Co-authored `draft_paper.md` (Abstract, Cross-Modality Interference Dilemma, and Dual-Expert Methodology), authored `modality_report.md`, and created system/design architecture diagrams (`artifacts/architecture_diagrams/`).
- **Viva Talking Points:** Why unified MRA+CTA training suffers negative transfer (flow physics vs. $>1000\text{ HU}$ bone attenuation), how Cross-Attention fuses 3D CNN and GATv2 embeddings, and how Focal + Smooth L1 multi-task losses work.

#### 2. Manya B A
- **Technical Implementation (Phases 4–5):**
  - Developed **3D Medial-Axis Skeletonization** (`src/topology/skeletonizer.py`) to extract 1-voxel-thick arterial centerlines from binary vessel masks and prune spurious micro-branches.
  - Implemented the **PyTorch Geometric Arterial Graph Builder** (`src/topology/graph_builder.py`), computing node features (normalized 3D coordinates, EDT vessel radius, tortuosity, degree centrality) and edge features (Euclidean length, 3D unit tangent, bifurcation angle).
  - Built the **GATv2 Graph Encoder** (`src/models/gat_module.py`) using multi-head `GATv2Conv` layers and global mean+max pooling to produce $256$-D topological embeddings.
- **Testing & Benchmarking:** Authored `tests/test_topology.py`, validated graph construction across real TopAneu scans, and benchmarked topological vs. pure visual representations.
- **Reports & Paper:** Co-authored `draft_paper.md` (Geometric Deep Learning & Vascular Graph Extraction sections), maintained `ROADMAP.md` Phases 4–5, and created Phase 5 presentation slides (`artifacts/presentation_phase5_phase6/phase5_work_done.png`, `phase5_results.png`).
- **Viva Talking Points:** How 3D medial-axis thinning works, how Murray's Law and bifurcation geometry help distinguish aneurysms from normal junctions, and why `GATv2Conv` outperforms static attention on vascular graphs.

#### 3. Mohammed Aseel Vanti
- **Technical Implementation (Phases 4 & 6):**
  - Implemented the **$k$-Hop Local Subgraph Extractor** (`src/topology/subgraph_extractor.py`) mapping 3D patch centers to surrounding arterial tree neighborhoods.
  - Built the **Full-Scan Candidate Proposer** (`src/inference/candidate_proposer.py`) and **3D Non-Maximum Suppression** (`src/inference/nms3d.py`) with a $6.0\text{ mm}$ clustering radius for whole-volume inference.
  - Designed and executed the **3D Arterial Vessel-Mask Conditioning** ablation, proving a $3.85\times$ search-space reduction, $71.6\%$ fewer candidates, $72.1\%$ fewer false positives, and $6.5\times$ faster inference ($877.3\text{ s} \rightarrow 133.8\text{ s}$).
- **Testing & Benchmarking:** Authored `tests/test_inference.py`, profiled end-to-end scan-to-diagnosis latency (~28–32 s/scan), and performed out-of-distribution false-positive forensics (scalp fat, orbital fat, bone marrow).
- **Reports & Paper:** Co-authored `draft_paper.md` (Search-Space Explosion & Vessel-Mask Conditioning Ablation), authored `comparison_with_without_vessel_mask.md`, and created Phase 6 presentation slides (`artifacts/presentation_phase5_phase6/phase6_work_done.png`, `phase6_results.png`).
- **Viva Talking Points:** Why sliding-window/intensity thresholding fails on full 55M-voxel volumes, how 3D-NMS merges duplicate detections, and how vessel-mask conditioning eliminates OOD artifacts.

#### 4. Nisarga Jessy P
- **Technical Implementation (Phases 6–7):**
  - Built the **Size-Stratified FROC Evaluator** (`src/inference/froc_evaluator.py`) measuring sensitivity at $0.5, 1.0, 2.0, 3.0, 4.0\text{ FP/scan}$ across $< 3\text{ mm}$, $3\text{--}5\text{ mm}$, $\le 5\text{ mm}$, and $> 5\text{ mm}$ cohorts.
  - Implemented **Zero-Shot Multi-Center Generalization Testing** (`src/inference/test_generalization.py`) on the Japan Center-4 CTA holdout cohort (85.9% sensitivity on $\le 5\text{ mm}$ lesions).
  - Developed the **Clinical Visualization & Risk Reporting Engine** (`src/ui/slice_visualizer.py` for synchronized Axial/Coronal/Sagittal MPR and MIP views; `src/ui/report_generator.py` for PHASES/ISUIA rupture risk scoring and HTML/CSV export).
- **Testing & Benchmarking:** Authored `tests/test_ui.py` and `tests/test_data_pipeline.py`; compiled FROC comparison tables and Center-4 generalization metrics (`artifacts/froc_eval/`, `artifacts/generalization/`).
- **Reports & Paper:** Co-authored `draft_paper.md` (Experimental Setup, Size-Stratified FROC Results & Clinical Discussion), compiled `results.md` and `walkthrough.md`, and curated clinical demo presets for external presentation.
- **Viva Talking Points:** How FROC evaluation differs from patch-level AUROC, why size-stratified ($<3\text{ mm}$, $3\text{--}5\text{ mm}$) evaluation is the core clinical proof, and how PHASES/ISUIA risk scoring and MPR views support radiologists.
