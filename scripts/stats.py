import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
import lpips
import torchvision.transforms as transforms

loss_fn = None


# SSIM and PSNR (expects NumPy arrays in [0, 1] range)
def compare_ssim(img1: np.ndarray, img2: np.ndarray):
    assert img1.shape == img2.shape, "Images must have the same shape"
    ssim_score = ssim(img1, img2, channel_axis=2, data_range=1.0)
    return ssim_score


def compare_psnr(img1: np.ndarray, img2: np.ndarray):
    assert img1.shape == img2.shape, "Images must have the same shape"
    psnr_score = psnr(img1, img2, data_range=1.0)
    return psnr_score


# LPIPS (expects PyTorch tensors in [-1, 1] range)
def compare_lpips(img1: np.ndarray, img2: np.ndarray, net_type='alex'):
    global loss_fn
    assert img1.shape == img2.shape, "Images must have the same shape"

    preprocess = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Converts to [-1, 1]
    ])

    img1_tensor = preprocess(img1).unsqueeze(0)
    img2_tensor = preprocess(img2).unsqueeze(0)

    if loss_fn is None:
        loss_fn = lpips.LPIPS(net='alex')  # 'alex', 'vgg', or 'squeeze'

    with torch.no_grad():
        lpips_score = loss_fn(img1_tensor, img2_tensor).item()

    return np.float32(lpips_score)
