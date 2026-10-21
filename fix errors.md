# Patent Document Revision Guide: Corrections and Replacements for check.docx

This document lists every factual error, corrupted formula, and typo identified in **check.docx**, providing the exact current text and the exact replacement text using plain text and standard symbols only (strictly no dollar symbols or LaTeX formatting).

---

## 1. Critical Formula and Clinical Logic Corrections

### Item 1: CTA Intensity Windowing Clinical Meaning
- **Location in check.docx:** Paragraph [92] (Section 2: Preprocessing of the volume - CTA)
- **Current Text (Incorrect):**
  > "To remove vessels from the images, and normalize away bone and soft tissue, the volume was clipped to 100-700 HU then normalized:"
- **Why it is wrong:**
  The purpose of windowing between 100 and 700 Hounsfield Units (HU) is the exact opposite: it isolates and highlights the contrast-enhanced blood vessels while suppressing dense bone (>700 HU) and brain parenchyma (<100 HU). Stating "To remove vessels" is a severe clinical error.
- **Replacement Text (Corrected):**
  > "To isolate and enhance contrast-filled arterial vessels, and to suppress dense skull bone (> 700 HU) and soft brain parenchyma (< 100 HU), the volume was clipped to the window [100, 700] HU, then normalized to the range [0, 1]:"

---

### Item 2: CTA Normalization Mathematical Formula
- **Location in check.docx:** Paragraph [93] (Section 2: Preprocessing of the volume - CTA)
- **Current Text (Corrupted):**
  > "Vnorm = (Vclipped 100) / (600 + 10), so it range is 0-1."
- **Why it is wrong:**
  1. The minus sign between Vclipped and 100 was dropped.
  2. The denominator "(600 + 10)" is completely wrong. In the code (src/data/preprocessor.py), an epsilon of 10^-6 (0.000001) is added for numerical stability to prevent division by zero. Dropping LaTeX converted 10^-6 into 10, changing the denominator to 610.
- **Replacement Text (Corrected):**
  > "V_norm = (V_clipped - 100) / (600 + 10^-6), ensuring values lie strictly within the range [0, 1]."

---

### Item 3: 3D Patch Boundary Padding Calculation
- **Location in check.docx:** Paragraph [95] (Section 2.3: Patch extraction)
- **Current Text (Corrupted):**
  > "The padding size in front of the volume along each dimension d is max(0, 32 c d ) and the padding size behind of the volume along each dimension d is max(0, c d + 32 D d ), where D d is the size of the volume along the dimension d."
- **Why it is wrong:**
  Minus signs were stripped out during text cleanup, making "32 - c_d" look like "32 c d" and "c_d + 32 - D_d" look like "c d + 32 D d".
- **Replacement Text (Corrected):**
  > "The padding size in front of the volume along each dimension d is max(0, 32 - c_d), and the padding size behind the volume along each dimension d is max(0, c_d + 32 - D_d), where D_d is the dimension length of the volume along axis d."

---

### Item 4: Search Space Reduction Magnitude
- **Location in check.docx:** Paragraph [207] (Section 8: Advantages of the invention, Point 4)
- **Current Text (Corrupted):**
  > "reduces the total number of search locations from O(3 10) voxels to a few hundred points..."
- **Why it is wrong:**
  "O(3 10)" is corrupted text. The original scientific value was 3 × 10^7 (30 million voxels).
- **Replacement Text (Corrected):**
  > "reduces the total number of search locations from approximately 3 × 10^7 voxels (~30 million voxels) to roughly 300–600 anatomically targeted candidate points..."

---

### Item 5: Multi-Task Centroid Offset Notation
- **Location in check.docx:** Paragraph [40] (Summary), Paragraph [220] (Claim 1), and Paragraph [238] (Claim 11)
- **Current Text (Inaccurate):**
  > "and a 3D sub-voxel (x, y,z) centroid offset (voxel)"
- **Why it is wrong:**
  The output head predicts a relative 3D displacement vector (delta x, delta y, delta z) from the patch center to the true aneurysm centroid, not the absolute volume coordinate (x, y, z).
- **Replacement Text (Corrected):**
  > "and a 3D sub-voxel centroid displacement offset vector (Δx, Δy, Δz) in voxels."

---

## 2. Dataset Cohort, Centers, and Size Statistics

### Item 6: Total Patient Scans and Documented Aneurysms
- **Location in check.docx:** Paragraph [74] and Paragraph [79] (Section 1: Data and patient cohort)
- **Current Text (Mismatched):**
  > "For example, it was created and evaluated on the multi-centre Top Aneu benchmark of 478 patient scans from four imaging centres... The cohort has 504 documented aneurysms on 478 scans."
- **Why it is wrong:**
  In our actual project dataset (configs/dataset_manifest.csv and artifacts/eda/aneurysm_annotations_summary.csv):
  - Total scans = 416 scans (Train: 293 + Validation: 62 + Test: 61 = 416).
  - Total aneurysms = 387 ground-truth lesions.
  The number 478 occurred because someone added the 62 Centre 4 holdout scans a second time to 416 (416 + 62 = 478). But Centre 4 is already part of the 416 scans.
- **Replacement Text (Corrected):**
  > "For example, it was created and evaluated on a multi-centre benchmark of 416 patient scans with 387 documented ground-truth aneurysms from four imaging centres:"

---

### Item 7: Imaging Center Names (Center 5 vs Center 3)
- **Location in check.docx:** Paragraphs [75] to [78] (Section 1: Data and patient cohort)
- **Current Text (Mismatched):**
  > "Centre 1: 3D TOF-MRA scans (voxel spacing) of 0.31 0.31 0.50 mm;  
  > Centre 2: 3D TOF-MRA and CTA scans;\  
  > Centre 3: 3D TOF-MRA scans;  
  > Centre 4: a HOLDOUT SET comprised of 62 CTA images..."
- **Why it is wrong:**
  In our project dataset and split files, the four centres are Centre 1, Centre 2, Centre 5, and Centre 4. There is no "Centre 3"; the third MRA cohort is designated as Centre 5.
- **Replacement Text (Corrected):**
  > "Centre 1: 199 3D TOF-MRA scans (mean voxel spacing: 0.30 × 0.30 × 0.55 mm);  
  > Centre 2: 87 scans comprising both 3D TOF-MRA and CTA modalities;  
  > Centre 5: 68 3D TOF-MRA scans;  
  > Centre 4: an independent holdout set of 62 CTA scans acquired with different scanner hardware and reconstruction kernels, reserved for zero-shot cross-center evaluation."

---

### Item 8: Patient Split Arithmetic and Cohort Breakdown
- **Location in check.docx:** Paragraph [79]
- **Current Text (Mismatched):**
  > "The training set contains 293 patient volumes (572 balanced candidate patches), the validation set has 62 patient volumes (96 balanced candidate patches), the test set has 61 patient volumes, and the Centre 4 holdout set contains 62 patient volumes. The cohort has 504 documented aneurysms on 478 scans."
- **Replacement Text (Corrected):**
  > "The splitting is strictly partitioned by patient with zero cross-set leakage: the training set contains 293 patient volumes (providing 572 balanced candidate patches), the validation set contains 62 patient volumes (providing 96 balanced candidate patches), and the test set contains 61 patient volumes. In a separate cross-center generalization evaluation, the 62 CTA scans from Centre 4 serve as an external holdout test set. Across the full 416 scans, 387 ground-truth aneurysms are documented."

---

### Item 9: Aneurysm Physical Size Stratification Percentages
- **Location in check.docx:** Paragraph [21] (Section 2.2) and Paragraph [79] (Section 1)
- **Current Text (Mismatched Sub-categories):**
  > "Size-wise, 10.7 % are <3 mm, 30.4 % are 3-5 mm (so 41.1 % are 5 mm or less), 24.0% are 5-7 mm and 34.9 % are >7 mm."
- **Why it is wrong:**
  The overall 41.1% for lesions ≤ 5 mm is exact (159 out of 387 lesions = 41.085%), but our exact Exploratory Data Analysis (EDA) size profiler extracted:
  - Very Small (< 3 mm): 25 lesions = 6.5% (not 10.7%)
  - Small (3–5 mm): 134 lesions = 34.6% (not 30.4%)
  - Medium (5–10 mm): 160 lesions = 41.3%
  - Large (> 10 mm): 68 lesions = 17.6%
- **Replacement Text (Corrected):**
  > "In physical size profiling across all 387 ground-truth lesions, small aneurysms (maximum physical diameter ≤ 5.0 mm) constitute 41.1% (159 lesions) of the entire cohort: 6.5% (25 lesions) are very small (< 3.0 mm), 34.6% (134 lesions) are 3.0–5.0 mm, 41.3% (160 lesions) are 5.0–10.0 mm, and 17.6% (68 lesions) exceed 10.0 mm."

---

## 3. Typographical and Grammatical Corrections

### Item 10: Word Splitting Typo in "aneurysms"
- **Location in check.docx:** Paragraph [204] (Section 8: Advantages of the invention, Point 1)
- **Current Text (Typo):**
  > "missed rate of aneuri sms was down from 8 to 4 in the validation data set."
- **Replacement Text (Corrected):**
  > "missed rate of aneurysms dropped from 8 down to 4 in the validation dataset."

---

### Item 11: Spur Pruning Grammar
- **Location in check.docx:** Paragraph [72] (Section 4: Summary)
- **Current Text (Typo):**
  > "pruning off spur 3 voxels or shorter caused by segmentation noise."
- **Replacement Text (Corrected):**
  > "pruning off terminal spurs 3 voxels or shorter caused by segmentation surface noise."

---

### Item 12: Heading Numbering Rogue Period
- **Location in check.docx:** Paragraph [209]
- **Current Text (Typo):**
  > ".9. Variations"
- **Replacement Text (Corrected):**
  > "9. Variations"

---

### Item 13: Edge Length and Tangent Notation in Abstract
- **Location in check.docx:** Paragraph [243] (Abstract)
- **Current Text:**
  > "while the edge vectors are four dimensional (length and 3D tangent)."
- **Replacement Text (Clearer):**
  > "while the edge vectors are 4-dimensional (physical segment length in mm and a normalized 3D unit tangent direction vector [dx/len, dy/len, dz/len])."

---

## 4. Verification Checklist: What check.docx Got 100% Right (Do NOT Change)

The following core components in **check.docx** are completely accurate and match our actual implementation:

1. **Table 1 Validation Metrics (Exact Match):**
   - Validation AUROC: Baseline = 0.8129 (81.3%), Proposed = 0.8728 (87.3%) [+5.99 points]
   - Sensitivity at threshold 0.5: Baseline = 83.3%, Proposed = 91.7% [+8.4 points]
   - Specificity at threshold 0.5: Baseline = 64.6%, Proposed = 72.9% [+8.3 points]
   - Youden Optimal Sensitivity: Baseline = 79.2%, Proposed = 89.6% [+10.4 points]
   - Youden Optimal Specificity: Baseline = 72.9%, Proposed = 79.2% [+6.3 points]
   - Optimal Decision Threshold: Baseline = 0.5317, Proposed = 0.5571
   - Validation Loss: Baseline = 0.1935, Proposed = 0.1796
   - Diameter MAE: Baseline = 2.89 mm, Proposed = 2.73 mm (best epoch: 2.47 mm)

2. **Confusion Matrix at Optimal Threshold 0.5571:**
   - 44 of 48 true aneurysms detected (only 4 false negatives).
   - 38 of 48 normal bifurcations correctly cleared.
   - False positives dropped from 17 down to 10 (41.2% relative reduction).
   - False negatives dropped from 8 down to 4 (50% relative reduction).

3. **Topology Pipeline Parameters:**
   - 1D medial axis thinning via 3D topological thinning.
   - Caliber extraction via 3D Euclidean Distance Transform (EDT) scaled by voxel spacing (dx, dy, dz).
   - 26-connectivity neighbor categorization: Endpoints (degree 1), Waypoints (degree 2), Bifurcations (degree ≥ 3).
   - Terminal spur pruning threshold = 3 voxels.
   - Node feature vector dimension = 8; Edge feature vector dimension = 4.
   - Local subgraph neighborhood: k = 3 hops, bounded at a maximum of 64 nodes.
   - Fallback graph: Deterministic 1-node dummy graph when no vessel lies within 30.0 mm.

4. **Deep Learning Architecture:**
   - Branch A: 3D ResNet-18 taking 1 × 64 × 64 × 64 patch, yielding 512D visual embedding.
   - Branch B: 3-layer GATv2 encoder with 4 attention heads, edge features, and dual global pooling (mean + max = 128 + 128 = 256D topological embedding).
   - Fusion: Multi-head cross-attention (8 heads) where visual features act as Queries and topological features act as Keys and Values, followed by LayerNorm and GELU FFN, outputting 512D fused embedding.
   - Multi-Task Loss: Binary Focal Loss (alpha = 0.75, gamma = 2.0) + Smooth L1 (Huber) for offset (weight = 0.1) and diameter (weight = 0.1) on positive samples only.
   - 3D-NMS clustering radius = 6.0 mm.
