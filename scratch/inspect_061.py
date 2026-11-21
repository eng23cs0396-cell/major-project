import sys, os
sys.path.insert(0, ".")
import numpy as np
import nibabel as nib
from src.inference.scan_detector import ScanDetector

img_nii = nib.load('dataset(topAneu)/images/topaneu_center1_mr_061_0000.nii.gz')
img = img_nii.get_fdata()
spacing = tuple(float(s) for s in img_nii.header.get_zooms()[:3])

detector = ScanDetector(model_type='fusion')
detector.conf_threshold = 0.05

dets = detector.detect_scan(img, vessel_mask=None, modality='mr', spacing=spacing, scan_id='topaneu_center1_mr_061')
print(f'Total raw detections: {len(dets)}')
for d in dets:
    p = d['aneurysm_prob']
    if p >= 0.40:
        c = tuple(round(float(x), 1) for x in d['phys_coord'])
        diam = round(float(d.get('diameter_mm', 0)), 2)
        bulge = round(float(d.get('bulge_ratio', 0)), 3)
        ctype = d.get('candidate_type', 'unknown')
        print(f'Coord: {c}, Prob: {p:.3f}, Diam: {diam}, Bulge: {bulge}, Type: {ctype}')
