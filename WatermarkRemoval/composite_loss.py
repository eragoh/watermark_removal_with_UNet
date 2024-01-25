import torch
import lpips  # LPIPS package, you need to install it
from pytorch_msssim import ssim  # SSIM package, you need to install it

# Device configuration (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def psnr_loss(output, target, max_pixel=1.0):
    mse = ((output - target) ** 2).mean()
    if mse == 0:
        return 0  # Return 0 for a perfect match (infinite PSNR)
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return 1 / psnr  # Inverting PSNR
    

def ssim_loss(output, target):
    return 1 - ssim(output, target, data_range=1, size_average=True)  # 1 - SSIM for loss

# LPIPS loss initialization
lpips_loss = lpips.LPIPS(net='alex').to(device)  # You can choose different networks as the base

# Composite loss function
def composite_loss(output, target, alpha, beta, gamma):
    """
    Composite loss function that combines PSNR, SSIM, and LPIPS.
    The weights alpha, beta, and gamma control the contribution of each loss component.
    """
    loss_psnr = psnr_loss(output, target).mean()
    loss_ssim = ssim_loss(output, target).mean()
    loss_lpips = lpips_loss(output, target).mean()

    total_loss = alpha * loss_psnr + beta * loss_ssim + gamma * loss_lpips
    return total_loss.mean(), loss_psnr, loss_ssim, loss_lpips  # Ensuring the final loss is a scalar
