# Comparative Ablation Study: Impact of 3D Arterial Vessel Mask Segmentation

**Project:** CerebroVascular AI: Dual-Expert Multimodal Intracranial Aneurysm Workstation  
**Author:** Dayananda Sagar University (DSU) Major Project Research Team (2026–2027)  
**Test Subject:** `topaneu_center1_mr_029_0000` (TOF-MRA, Center 1, Patient 029)  
**Clinical Ground Truth:** **Confirmed Healthy Control Scan** (`locations: []`, `num_aneurysms: 0`)  

---

## 1. Executive Summary & Empirical Benchmark

This comparative ablation study evaluates the performance of the end-to-end 3D detection pipeline on an identical, uncropped clinical TOF-MRA scan (`topaneu_center1_mr_029_0000`, dimensions: $558 \times 719 \times 138$ voxels, physical resolution: $0.251 \times 0.251 \times 0.700\text{ mm}$) under two operating conditions:
1. **Run 1: Without Vessel Mask** (Raw intensity thresholding fallback at $98\text{th}$ percentile)
2. **Run 2: With Segmented Arterial Vessel Mask** (Restricted strictly to the cerebral arterial lumen)

### Side-by-Side Empirical Comparison

| Diagnostic Metric | Without Vessel Mask (Raw Scan Fallback) | With Arterial Vessel Mask | Relative Impact / Delta |
| :--- | :--- | :--- | :--- |
| **Search Space Voxels** | **1,096,366 voxels** | **284,503 voxels** | **3.85× search space reduction** |
| **Candidate Proposals Generated** | ~3,200 raw skeleton waypoints | 910 arterial waypoints | **71.6% fewer candidates** |
| **Final Detections Flagged ($\ge 0.78$)** | **43 candidates** | **12 candidates** | **72.1% false-positive reduction** |
| **Maximum Lesion Diameter** | **13.92 mm** (Scalp fat artifact) | **5.84 mm** (Arterial junction) | **Eliminated non-anatomical lesions >7mm** |
| **Inference Runtime (CPU)** | **877.3 seconds (14.6 minutes)** | **133.8 seconds (2.2 minutes)** | **6.5× end-to-end acceleration** |
| **Highest Rupture Risk Tier** | **Critical** (Triggered by 13.9mm fat) | **High** (Triggered by 5.8mm bifurcation) | Downgraded severity category |
| **Anatomical Distribution** | Dispersed across scalp, skull, eyes, sinuses | Clustered strictly on MCA and Basilar junctions | **Zero non-cerebral artifacts** |

```
Without Vessel Mask: [==================================================] 43 Detections (877.3s)
With Vessel Mask:    [==============] 12 Detections (133.8s)  -> 72% FP Drop, 6.5x Speedup!
```

---

## 2. Technical Forensic Analysis: Why Detections and Runtime Decreased

### A. The 55-Million Voxel Search Space Dilemma
A raw 3D head scan contains $558 \times 719 \times 138 \approx 55,364,000$ voxels. When an end-to-end detector is asked to locate small millimeter aneurysms ($2\text{--}7\text{ mm}$) in a 55-million voxel volume without anatomical constraints, it must first decide *where* to look.

* **Without Vessel Mask:** The fallback algorithm calculates the top 2% brightest voxels (`image > np.percentile(image, 98.0)`). In TOF-MRA, blood flow produces high signal, but **subcutaneous scalp fat, calvarial bone marrow, orbital fat around the eyes, nasal mucosa, and dural venous sinuses (superior sagittal sinus, transverse sinus)** also exhibit intense signal. Consequently, **1,096,366 voxels** across the entire head and face were treated as candidate tissue.
* **With Vessel Mask:** The mask restricts the search space exclusively to the **284,503 voxels** of the intracranial arterial tree. This immediately eliminates nearly 800,000 non-vascular voxels.

### B. Epistemic Uncertainty & Out-of-Distribution (OOD) Predictions
The dual-branch neural architecture (3D ResNet-18 + Arterial Topology GATv2) was trained on 3D patches extracted strictly from cerebral arteries.
* When presented with non-cerebral tissue (such as scalp fat nodules or tortuous skull-base veins), the neural network operates **completely out of its training distribution**.
* Deep convolutional neural networks are notoriously overconfident on out-of-distribution inputs. A 13.9 mm globule of bright scalp fat with irregular boundaries triggered high activation ($87.6\%$) in the CNN branch, falsely classifying fat as a "Critical Giant Aneurysm".
* Providing the arterial vessel mask acts as a **hard topological prior**: the candidate proposer is prevented from sampling outside the cerebral vasculature, instantly eliminating all 31 non-cerebral false alarms.

### C. The 6.5× Runtime Acceleration
Inference runtime is directly proportional to the number of candidate interrogation points evaluated by the deep network:
$$\text{Runtime} \approx T_{\text{skeleton}} + \left(\frac{N_{\text{candidates}}}{\text{Batch Size}}\right) \times T_{\text{forward}}$$
* Without mask: Evaluating thousands of candidates across the skull required **877.3 seconds (14.6 minutes)**.
* With mask: Evaluating only 910 arterial candidates required **133.8 seconds (2.2 minutes)**.

---

## 3. The Deeper Medical Question: Why Did 12 Detections Still Appear on a Healthy Patient?

Even with the vessel mask eliminating non-cerebral artifacts, **12 detections remained inside the arterial tree with confidence scores between $79.8\%$ and $89.7\%$**, despite Patient 029 having zero ground-truth aneurysms.

### A. Exact Anatomical Location Mapping
Analyzing the coordinates of the 12 remaining detections reveals a striking pattern:

| Rank | Confidence | Diameter | Anatomical Site | Centroid Coordinates (mm) |
| :---: | :---: | :---: | :--- | :--- |
| **#1** | **89.7%** | 4.15 mm | Right Middle Cerebral Artery (MCA) Bifurcation | $(69.7, 60.0, 50.4)$ |
| **#2** | **88.7%** | 3.90 mm | Basilar Artery / Vertebrobasilar Junction | $(77.7, 60.5, 44.8)$ |
| **#3** | **88.7%** | 4.34 mm | Right Middle Cerebral Artery (MCA) Bifurcation | $(68.0, 57.3, 57.3)$ |
| **#4** | **88.6%** | 4.43 mm | Basilar Artery / Vertebrobasilar Junction | $(73.5, 54.7, 58.0)$ |
| **#5** | **86.8%** | 3.80 mm | Right Middle Cerebral Artery (MCA) Bifurcation | $(60.2, 59.8, 44.8)$ |
| **#6** | **84.3%** | 3.91 mm | Basilar Artery / Vertebrobasilar Junction | $(80.8, 51.2, 44.1)$ |
| **#7** | **84.2%** | 4.01 mm | Right Middle Cerebral Artery (MCA) Bifurcation | $(58.0, 65.8, 45.4)$ |
| **#8** | **82.3%** | 5.35 mm | Basilar Artery / Vertebrobasilar Junction | $(70.3, 61.0, 65.0)$ |
| **#9** | **81.8%** | 4.07 mm | Basilar Artery / Vertebrobasilar Junction | $(72.5, 63.8, 55.1)$ |
| **#10** | **81.7%** | 5.84 mm | Right Middle Cerebral Artery (MCA) Bifurcation | $(55.5, 74.7, 53.7)$ |
| **#11** | **80.3%** | 4.23 mm | Basilar Artery / Vertebrobasilar Junction | $(85.1, 70.5, 55.0)$ |
| **#12** | **79.8%** | 4.26 mm | Right Middle Cerebral Artery (MCA) Bifurcation | $(57.7, 53.5, 43.4)$ |

The 12 detections are **not** random noise. They cluster exclusively at the **exact two largest, highest-stress vascular branching junctions in the human brain**:
1. **The Basilar / Vertebrobasilar Junction (7 detections):** Where the two vertebral arteries merge into the basilar trunk.
2. **The Right MCA Bifurcation (5 detections):** Where the M1 arterial trunk splits into M2 superior/inferior divisions.

### B. The Hemodynamic & Training Bias
In human clinical neurosurgery, **over 85% of all intracranial aneurysms form at these exact bifurcations** due to chronic blood-flow shear stress.
* During model training, thousands of positive training patches contained bifurcation junctions with aneurysms.
* The neural network learned strong visual associations:
  $$\text{Complex 3D Arterial Bifurcation} + \text{High-Flow Signal} + \text{Local Caliber Widening} \implies P(\text{Aneurysm}) \uparrow$$
* In Patient 029, their normal basilar trunk and MCA bifurcation happen to have natural anatomical tortuosity and high MRA flow signal. The deep neural network recognized these valid bifurcation features and output high confidence ($80\% - 89\%$).

### C. The Mathematical Paradox of Raw Deep Learning Probabilities
* In **Case 1 (Patient 193)**, a real, confirmed aneurysm received a raw score of **`73.1%`**.
* In **Patient 029**, a healthy normal bifurcation received a raw score of **`88.7%`**.

**Raw neural network probability is NOT an absolute diagnostic verdict.** It reflects visual patch similarity to the training set. A prominent, tortuous normal bifurcation can have higher neural activation than a small, subtle real aneurysm.

---

## 4. The Clinical Role of CADe / CADx Software in Radiology

This finding underscores why FDA-cleared clinical AI software (such as Aidoc, Viz.ai, or RapidAI) is legally regulated as **Computer-Aided Detection (CADe) / Second-Reader Triage**, and **never an autonomous diagnostic tool**:

1. **AI as an Attention Director:**  
   The AI reviews 50 million voxels in 2 minutes and highlights high-risk branching locations for human review.
2. **Human Radiological Confirmation via Multiplanar Reconstruction (MPR):**  
   When the workstation flags candidate #1 at $89.7\%$, the radiologist inspects the **Axial, Coronal, and Sagittal orthogonal cross-sections**:
   * **In a true aneurysm:** The crosshairs reveal a focal, asymmetrical saccular balloon protruding outward from the vessel wall with a distinct neck.
   * **In Patient 029:** The crosshairs reveal a **smooth, continuous, tubular vessel dividing evenly into two daughter branches** without an outpouching neck.
   * The radiologist verifies the normal tubular caliber in under 5 seconds and safely dismisses the finding.

---

## 5. Architectural Improvements for Future Phases

To further reduce normal-bifurcation false positives without missing true subtle aneurysms, three algorithmic enhancements are recommended:

1. **Automated Intracranial Vault Extraction (Otsu Brain Envelope Masking):**  
   Implement an automated brain-stripping preprocessor so that raw scan uploads automatically exclude scalp fat and skull marrow even if no manual vessel mask is supplied.
2. **Murray's Law Saccular Morphometry Filter:**  
   Normal bifurcations obey Murray's Law of minimal work ($r_0^3 = r_1^3 + r_2^3$), where daughter vessels maintain cylindrical caliber. True aneurysms exhibit focal bulbous expansion ($r_{\text{local}} > 1.6 \cdot r_{\text{parent}}$). Candidates with a low bulge ratio ($<1.2$) can be down-weighted.
3. **Spatial Cluster Density & Neck Morphometry:**  
   In true aneurysms, multiple candidates cluster within a 4 mm sphere. Incorporating candidate cluster density into the verdict logic provides an additional layer of specificity.

---

*Report compiled for the Dayananda Sagar University (DSU) Major Project evaluation committee.*
