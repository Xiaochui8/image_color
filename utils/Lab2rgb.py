import torch
from skimage.color import lab2rgb

def Lab2rgb(L, ab):
    # L [0, 1] -> [0, 100] ab [0, 1] -> [-128, 127]
    # L (b, 1, h, w) ab (b, 2, h, w)
    L = L * 100
    ab = ab * 255 - 128
    Lab = torch.cat([L, ab], dim=1).permute(0, 2, 3, 1).cpu().numpy()
    rgb = torch.tensor(lab2rgb(Lab)).permute(0, 3, 1, 2) # (b, h, w, 3) -> (b, 3, h, w)
    
    return rgb