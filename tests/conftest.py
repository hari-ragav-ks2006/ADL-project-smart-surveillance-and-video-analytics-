import os
import sys
import cv2
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
cv2.setNumThreads(1)
torch.set_num_threads(1)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def pytest_sessionfinish(session, exitstatus):
    """Ensure process terminates cleanly on Windows when C-threads are active."""
    import os
    os._exit(exitstatus)
