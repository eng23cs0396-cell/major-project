# 🧠 Topology-Aware Deep Learning for Small Intracranial Aneurysm Detection
## Comprehensive Phase-by-Phase Roadmap & Development Guide

> **Institution**: Dayananda Sagar University (DSU), Department of Computer Science & Engineering  
> **Academic Year**: 2026–2027 | Major Project Phase-I  
> **Project Guide**: Prof. Sharath H A  
> **Project Team**: Manya B A, Mohammed Aseel Vanti, Nisarga Jessy P, Pavan P Yadav  
> **GitHub Repository**: [https://github.com/eng23cs0396-cell/major-project](https://github.com/eng23cs0396-cell/major-project)  
> **Dataset (TopAneu on Hugging Face)**: [https://huggingface.co/datasets/SlaYeRRRRRdwdd/topaneu](https://huggingface.co/datasets/SlaYeRRRRRdwdd/topaneu)

---

## 🎯 Clinical Problem Statement & Core Hypothesis

### The Clinical Problem
Intracranial aneurysms carry a catastrophic risk of subarachnoid hemorrhage upon rupture. However, detecting **small ($\le 5\text{ mm}$)** and **very small ($< 3\text{ mm}$)** aneurysms on 3D Magnetic Resonance Angiography (TOF-MRA) and Computed Tomography Angiography (CTA) scans remains one of the hardest challenges in clinical radiology. On standard visual scans, tiny aneurysms appear virtually identical to tortuous normal arterial bends, loopings, and infundibular bifurcations.

### The Research Hypothesis
Purely visual 3D CNNs suffer high false-positive rates on small lesions because local pixel intensities alone lack structural context. By **modeling the vascular tree as a topological geometric graph** and fusing a **3D Convolutional Neural Network (CNN)** with a **Graph Attention Network (GAT)**, our system leverages both local voxel appearance and global arterial branching geometry. This allows the model to distinguish genuine pathological outpouchings from normal vascular junctions, substantially boosting sensitivity for $\le 5\text{ mm}$ aneurysms while keeping false positives per scan low.

---

## 🗺️ Project Architecture & Status Summary

| Phase | Phase Name | Status | Key Deliverables |
| :---: | :--- | :---: | :--- |
| **Phase 1** | **Environment & Project Architecture** | ✅ **COMPLETED** | Python 3.10 venv, PyTorch 2.5.1 + CUDA 12.1 (RTX 4060 GPU), PyG, MONAI, NiBabel, SimpleITK, `tests/test_environment.py`. |
| **Phase 2** | **Data Pipeline, Splitting & Size Profiling** | ✅ **COMPLETED** | Patient-stratified splits (293 train / 62 val / 61 test / 62 Center-4 holdout), physical size profiler (41.1% $\le 5\text{ mm}$), modality preprocessor (MRA & CTA), `tests/test_data_pipeline.py`. |
| **Phase 3** | **3D CNN Baseline Model (ResNet3D)** | ✅ **COMPLETED** | 3D ResNet-18 backbone, multi-task heads (classification, 3D offset, diameter), mixed-precision trainer, `tests/test_model_baseline.py`. |
| **Phase 4** | **Vascular Topology (Skeleton & Graph)** | ✅ **COMPLETED** | 3D medial axis thinning, branch-point detection, PyTorch Geometric arterial graph builder, $k$-hop subgraph extractor, `tests/test_topology.py`. |
| **Phase 5** | **GAT & Multimodal Cross-Attention Fusion** | 🚀 **NEXT** | GATv2 graph encoder, Cross-Attention fusion block, end-to-end dual-branch architecture, `tests/test_fusion_model.py`. |
| **Phase 6** | **Full-Scan Detector, 3D-NMS & FROC Evaluation**| ⏳ *Planned* | Skeleton candidate proposer, 3D Non-Maximum Suppression, FROC curves (sensitivity @ 0.5, 1.0, 2.0 FPs/scan), size-stratified validation (<3mm, 3-5mm, >5mm), Center-4 zero-shot generalization. |
| **Phase 7** | **Interactive Clinical UI & Decision Support** | ⏳ *Planned* | Web-based diagnostic dashboard (MRA/CTA scan viewer, 3D aneurysm localization with coordinates, diameter, risk assessment, and exportable clinical summary report). |

---

## 🛠️ Step-by-Step Guide for Teammates: What To Do in Each Phase

### 🚀 Phase 3: 3D CNN Baseline Architecture (Completed)
*Goal: Build the pure visual 3D convolutional baseline against which our topology-aware graph model will be benchmarked.*

#### What Files to Create:
1. **`src/models/resnet3d.py`**:
   - Implement a modular 3D ResNet encoder (ResNet-18/34 3D architecture adapted for volumetric medical patches of size $64 \times 64 \times 64$).
   - Output representation: 512-dimensional visual feature embedding $\mathbf{z}_{\text{vis}} \in \mathbb{R}^{512}$.
   - Attach multi-task prediction heads:
     1. **Aneurysm Probability Head**: Linear layer $\rightarrow$ Sigmoid $\rightarrow P(\text{aneurysm}) \in [0, 1]$.
     2. **3D Centroid Offset Head**: Linear layer $\rightarrow (\Delta x, \Delta y, \Delta z)$ in voxels (refines the exact center).
     3. **Physical Diameter Regression Head**: Linear layer $\rightarrow \text{ReLU} \rightarrow \hat{D}_{\text{max}}$ in millimeters.
2. **`src/models/losses.py`**:
   - **Classification Loss**: Focal Loss ($\alpha=0.25, \gamma=2.0$) or Weighted BCE to handle candidate class imbalance.
   - **Bounding/Centroid Offset Loss**: Smooth L1 (Huber) Loss for $(\Delta x, \Delta y, \Delta z)$.
   - **Size Regression Loss**: Smooth L1 Loss for physical diameter $\hat{D}_{\text{max}}$.
   - Multi-task total loss: $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{focal}} + \lambda_{\text{offset}} \mathcal{L}_{\text{offset}} + \lambda_{\text{size}} \mathcal{L}_{\text{size}}$.
3. **`src/models/trainer.py`**:
   - PyTorch training engine utilizing `torch.cuda.amp.autocast()` and `GradScaler` for 16-bit mixed-precision execution on the RTX 4060 (8GB VRAM).
   - Training on `configs/train_split.csv` and validation tracking on `configs/val_split.csv`.
   - Logging metrics: AUROC, Sensitivity, Specificity, Mean Absolute Error (MAE) on diameter regression.
   - Model checkpointing: Saves `best_baseline_resnet3d.pt` based on validation AUROC.
4. **`tests/test_model_baseline.py`**:
   - Unit test checking forward and backward passes with dummy inputs of shape `(B=4, C=1, D=64, H=64, W=64)` on both CPU and GPU.
   - Test verifying loss computation and gradient backpropagation.

#### How to Run & Verify:
```bash
.venv\Scripts\python -m pytest tests/test_model_baseline.py -v
.venv\Scripts\python -m src.models.trainer --epochs 50
```

Smoke check with a tiny subset (does not replace full training):
```bash
.venv\Scripts\python -m src.models.trainer --epochs 1 --batch-size 2 --max-train-samples 8 --max-val-samples 4
```

Checkpoints are written to `checkpoints/best_baseline_resnet3d.pt` and `checkpoints/last_baseline_resnet3d.pt`.


---

### 🕸️ Phase 4: Vascular Topology Pipeline (Skeletonization & Graph Construction)
*Goal: Transform binary 3D cerebral vessel segmentations (`vessel_masks/`) into mathematical arterial graphs.*

#### What Files to Create:
1. **`src/topology/skeletonizer.py`**:
   - Load full 3D vessel mask NIfTI (`vessel_masks/*.nii.gz`).
   - Run 3D medial axis thinning (`skimage.morphology.skeletonize_3d` or SimpleITK thinning).
   - Prune spurious micro-branches and skeleton noise (< 3 voxels).
2. **`src/topology/graph_builder.py`**:
   - Convert skeleton voxels into a mathematical graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$:
     - **Nodes $\mathcal{V}$**:
       - Bifurcations (degree $\ge 3$)
       - Endpoints (degree $= 1$)
       - Centerline waypoints sampled along vessel segments (degree $= 2$)
       - **Node Feature Vector $\mathbf{x}_v$**:
         - 3D normalized spatial coordinates $(x, y, z)$.
         - Local vessel caliber / radius $r$ (computed via 3D Euclidean Distance Transform of vessel mask).
         - Local vessel tortuosity (ratio of path length to Euclidean distance).
         - Degree centrality of the node.
     - **Edges $\mathcal{E}$**:
       - Arterial connectivity connecting adjacent skeleton points.
       - **Edge Feature Vector $\mathbf{e}_{uv}$**:
         - Euclidean segment length.
         - 3D unit tangent direction vector $(\Delta x, \Delta y, \Delta z) / \|\Delta \mathbf{x}\|$.
         - Bifurcation angle between parent and daughter branches.
   - Return as a PyTorch Geometric `torch_geometric.data.Data` object.
3. **`src/topology/subgraph_extractor.py`**:
   - Given any 3D candidate coordinate $(x, y, z)$, identify the closest vascular skeleton node.
   - Extract a $k$-hop local arterial subgraph (e.g., $k=3$ or $k=5$ hops, covering $\sim 20-30\text{ mm}$ of surrounding vascular tree).
4. **`tests/test_topology.py`**:
   - Unit test running on a real patient vessel mask from the dataset.
   - Verify that skeletonization completes, nodes/edges are extracted, and PyG graph object matches expected dimensions.

#### How to Run & Verify:
```bash
.venv\Scripts\python -m pytest tests/test_topology.py -v
```

---

### ⚡ Phase 5: GAT & Multimodal Cross-Attention Fusion Architecture
*Goal: Construct the core innovation — fusing 3D volumetric visual features with arterial topology features.*

#### What Files to Create:
1. **`src/models/gat_module.py`**:
   - Implement a multi-layer Graph Attention Network using `torch_geometric.nn.GATv2Conv`.
   - Node representations updated with multi-head attention over neighboring vessel branches.
   - Global graph pooling (mean + max pooling) to generate graph topological embedding $\mathbf{z}_{\text{topo}} \in \mathbb{R}^{256}$.
2. **`src/models/dual_branch_detector.py`**:
   - Dual-input architecture:
     - Input 1: 3D volumetric image patch $(1 \times 64 \times 64 \times 64)$.
     - Input 2: Local vascular PyG subgraph $(V, E)$ centered at the patch.
   - **Cross-Attention Fusion Module**:
     - Visual features $\mathbf{z}_{\text{vis}}$ serve as Queries ($Q$).
     - Topological features $\mathbf{z}_{\text{topo}}$ serve as Keys ($K$) and Values ($V$).
     - Cross-attention weights prioritize visual features that align with abnormal arterial branching geometry.
   - Fused representation $\mathbf{z}_{\text{fused}} \in \mathbb{R}^{512}$ feeds into the final multi-task prediction heads ($P(\text{aneurysm})$, centroid offsets, diameter).
3. **`src/models/train_fusion.py`**:
   - PyTorch training loop for the end-to-end dual-branch model.
4. **`tests/test_fusion_model.py`**:
   - Unit test passing both a batch of 3D image patches and a batch of PyG subgraphs through the complete dual-branch model to verify end-to-end backprop.

#### How to Run & Verify:
```bash
.venv\Scripts\python -m pytest tests/test_fusion_model.py -v
```

---

### 📈 Phase 6: Full-Scan Clinical Detection, 3D-NMS & FROC Evaluation
*Goal: Evaluate real-world clinical utility — answering "Is there an aneurysm, and where is it?" across whole 3D volumes.*

#### What Files to Create:
1. **`src/inference/candidate_proposer.py`**:
   - Real-world clinical scans have millions of voxels. Searching all voxels is inefficient.
   - Traverse the patient's vascular skeleton to place candidate interrogation points at all arterial bifurcations and high-curvature points.
2. **`src/inference/nms3d.py`**:
   - 3D Non-Maximum Suppression: Clusters overlapping detections within a radius of $6.0\text{ mm}$ and retains the candidate with highest confidence.
3. **`src/inference/froc_evaluator.py`**:
   - Clinical Free-Response Receiver Operating Characteristic (FROC) calculation:
     - Match predicted coordinates to ground-truth coordinates (hit criteria: Euclidean distance $\le 6.0\text{ mm}$).
     - Measure Detection Sensitivity at benchmark false-positive rates: **0.5, 1.0, 2.0, 3.0, and 4.0 False Positives per scan**.
   - **Crucial Size-Stratified Breakdown** (The core thesis proof!):
     - Compute separate FROC sensitivities for:
       - **Very Small Aneurysms ($< 3\text{ mm}$)**
       - **Small Aneurysms ($3 - 5\text{ mm}$)**
       - **Overall Small ($\le 5\text{ mm}$)**
       - **Medium & Large ($> 5\text{ mm}$)**
   - **Baseline vs Proposed Comparison Table**:
     - Compare pure 3D CNN Baseline vs Dual-Branch CNN+GAT Model.
4. **`src/inference/test_generalization.py`**:
   - Zero-shot cross-center evaluation on `configs/center4_holdout_split.csv` (Japan Center-4 CTA cohort) to demonstrate robust domain transfer from MRA to CTA.

#### How to Run & Verify:
```bash
.venv\Scripts\python src/inference/froc_evaluator.py --config configs/default_config.yaml
```

---

### 🖥️ Phase 7: Interactive Clinical Decision Support UI & Report Generator
*Goal: Provide a clean, intuitive diagnostic tool for clinicians and demonstration during project reviews.*

#### What Files to Create:
1. **`src/ui/app.py`**:
   - Web application (Gradio / Streamlit / Modern Web).
   - Allows a user/doctor to:
     1. Upload any patient 3D NIfTI scan (TOF-MRA or CTA).
     2. Automatic modality detection and intensity normalization.
     3. Click **"Analyze Cerebral Vasculature"**.
     4. Display interactive orthogonal 2D slices (Axial, Coronal, Sagittal) with bounding boxes and centroid markers around detected aneurysms.
     5. Present the **Clinical Findings Table**:
        - Aneurysm Index (#1, #2, ...)
        - 3D Centroid Coordinates $(x, y, z)$ in voxel and physical space.
        - Physical Maximum Diameter in mm (with warning flag if $< 3\text{ mm}$ or $3-5\text{ mm}$).
        - Anatomical Arterial Location (e.g. ACom, MCA M1 bifurcation, ICA-PCom).
        - Rupture Risk Category (Low, Moderate, High based on size and location).
     6. Export formatted PDF / CSV Diagnostic Summary Report.

---

## 💻 Instructions for Teammates Working on the Project

### 1. How to Set Up the Environment on a New Machine
```bash
# Clone the repository
git clone https://github.com/eng23cs0396-cell/major-project.git
cd "major project"

# Create Python 3.10 virtual environment
python -m venv .venv

# Activate the virtual environment
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat

# Install PyTorch with CUDA 12.1 acceleration
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install PyG (PyTorch Geometric)
pip install torch-geometric==2.8.0

# Install medical imaging dependencies
pip install -r requirements.txt
```

### 2. How to Verify That Everything Is Working
Always run the automated test suite before writing new code:
```bash
.venv\Scripts\python -m pytest tests/ -v
```
All tests in `test_environment.py` and `test_data_pipeline.py` should show `[PASSED]`.

### 3. How to Access the Data
The project uses the TopAneu dataset with 416 scans:
- Stored locally in `dataset(topAneu)/`:
  - `images/`: 416 raw 3D NIfTI volumes (`*.nii.gz`).
  - `vessel_masks/`: 416 aligned cerebral vessel masks (`*.nii.gz`).
  - `annotations/annotations.json`: Detailed 3D coordinate and physical size annotations.
- Full cloud backup verified on Hugging Face: [https://huggingface.co/datasets/SlaYeRRRRRdwdd/topaneu](https://huggingface.co/datasets/SlaYeRRRRRdwdd/topaneu).

### 4. Git Collaboration Guidelines
- Always pull the latest changes before starting work:
  ```bash
  git pull origin main
  ```
- Make focused, meaningful commits with descriptive messages:
  ```bash
  git add <modified_files>
  git commit -m "feat(models): implement 3D ResNet backbone with multi-task heads"
  git push origin main
  ```
- Always update `walkthrough.md` when completing a phase or major milestone!

---
*Document Version: 1.0 (September 2026)*  
*Maintained by the Major Project Phase-I Team & Antigravity AI Assistant.*
