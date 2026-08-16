"""
tracker/depth_estimator.py
Uses MiDaS small to estimate Z-axis depth maps.
"""
import cv2
import numpy as np

class DepthEstimator:
    def __init__(self, device="auto"):
        import torch
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        print(f"[Depth] Loading MiDaS_small on {self.device}...")
        # PyTorch 2.6 weights_only fix for torch.hub
        _orig_load = torch.load
        def _patched_load(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return _orig_load(*args, **kwargs)
        torch.load = _patched_load
        
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()
            
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
        finally:
            torch.load = _orig_load

    def get_depth(self, frame_rgb: np.ndarray, x: int, y: int) -> float:
        """
        Returns the relative Z depth at (x, y). 
        Higher value = closer to camera.
        """
        import torch
        input_batch = self.transform(frame_rgb).to(self.device)
        
        with torch.no_grad():
            prediction = self.model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
        depth_map = prediction.cpu().numpy()
        
        # Clamp coordinates
        h, w = depth_map.shape
        x = max(0, min(w - 1, int(x)))
        y = max(0, min(h - 1, int(y)))
        
        return float(depth_map[y, x])
