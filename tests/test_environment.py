import sys
import os

def test_python_version():
    assert sys.version_info >= (3, 10), f"Python 3.10+ required, got {sys.version}"
    print(f"[PASS] Python version: {sys.version.split()[0]}")

def test_torch_and_cuda():
    import torch
    print(f"[PASS] PyTorch version: {torch.__version__}")
    assert torch.cuda.is_available(), "CUDA is not available in PyTorch!"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[PASS] CUDA available: True | Device: {gpu_name}")
    # Verify tensor allocation on GPU
    x = torch.randn(100, 100, device="cuda")
    y = x @ x
    assert y.is_cuda, "GPU computation failed!"
    print(f"[PASS] GPU Tensor Matrix Multiplication Successful!")

def test_monai_and_nibabel():
    import nibabel as nib
    import monai
    print(f"[PASS] MONAI version: {monai.__version__}")
    print(f"[PASS] NiBabel version: {nib.__version__}")
    
    # Test loading a real scan from dataset(topAneu)
    sample_scan = "dataset(topAneu)/images/topaneu_center2_mr_002_0000.nii.gz"
    if os.path.exists(sample_scan):
        img = nib.load(sample_scan)
        shape = img.shape
        zooms = img.header.get_zooms()
        print(f"[PASS] Real Scan Load: {sample_scan}")
        print(f"       Shape: {shape} | Voxel Spacing: {zooms}")

def test_pyg():
    import torch
    import torch_geometric
    from torch_geometric.data import Data
    from torch_geometric.nn import GATv2Conv
    print(f"[PASS] PyTorch Geometric version: {torch_geometric.__version__}")
    
    # Test a small graph forward pass
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long, device="cuda")
    x = torch.randn(3, 8, device="cuda")
    conv = GATv2Conv(8, 16, heads=2).to("cuda")
    out = conv(x, edge_index)
    assert out.shape == (3, 32), f"Expected shape (3, 32), got {out.shape}"
    print(f"[PASS] GATv2Conv GPU forward pass verified!")

def test_skeletonization():
    import numpy as np
    from skimage.morphology import skeletonize
    vol = np.zeros((20, 20, 20), dtype=bool)
    vol[5:15, 10, 10] = True
    skel = skeletonize(vol)
    assert skel.sum() > 0, "Skeletonization failed"
    print(f"[PASS] 3D Skeletonization operational!")

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ENVIRONMENT VERIFICATION SUITE")
    print("=" * 60)
    test_python_version()
    try:
        test_torch_and_cuda()
    except Exception as e:
        print(f"[FAIL] PyTorch/CUDA: {e}")
    try:
        test_monai_and_nibabel()
    except Exception as e:
        print(f"[FAIL] MONAI/NiBabel: {e}")
    try:
        test_pyg()
    except Exception as e:
        print(f"[FAIL] PyTorch Geometric: {e}")
    try:
        test_skeletonization()
    except Exception as e:
        print(f"[FAIL] Skeletonization: {e}")
    print("=" * 60)
