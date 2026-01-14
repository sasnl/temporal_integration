
import numpy as np
import sys
import os

# Add shared folder to path
# code/TI_code/test.py -> code/TI_code/shared
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shared'))
from pipeline_utils import apply_cluster_threshold

def test_cluster_threshold():
    print("Testing cluster thresholding...")
    
    # Create a 3D volume (10x10x10)
    vol = np.zeros((10, 10, 10), dtype=np.float32)
    
    # Create Cluster 1: Size 5
    vol[1:6, 1, 1] = 1.0 # 5 voxels in a line
    
    # Create Cluster 2: Size 20
    vol[1:6, 4:8, 4] = 2.0 # 5 * 4 = 20 voxels
    
    print(f"Original non-zero voxels: {np.sum(vol > 0)}")
    
    # Test 1: Threshold = 0 (Should keep all)
    res_0 = apply_cluster_threshold(vol, 0)
    assert np.sum(res_0 > 0) == 25
    print("Test 1 (k=0): PASSED")
    
    # Test 2: Threshold = 10 (Should remove cluster 1 (size 5), keep cluster 2 (size 20))
    res_10 = apply_cluster_threshold(vol, 10)
    assert np.sum(res_10 > 0) == 20
    # Check that the remaining voxels are from Cluster 2 (value 2.0)
    assert np.all(res_10[res_10 > 0] == 2.0)
    print("Test 2 (k=10): PASSED")
    
    # Test 3: Threshold = 30 (Should remove all)
    res_30 = apply_cluster_threshold(vol, 30)
    assert np.sum(res_30 > 0) == 0
    print("Test 3 (k=30): PASSED")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_cluster_threshold()
