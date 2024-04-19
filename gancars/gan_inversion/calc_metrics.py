import os
import torch
import numpy as np
from PIL import Image
from glob import glob
from dnnlib_util import get_feature_detector, vgg16_url
from skimage.metrics import structural_similarity


def calc_ssim(gt_images, pred_images):
    ssim = np.mean([structural_similarity(gt_images[i], pred_images[i], channel_axis=2, data_range=1.0) for i in range(len(gt_images))])
    return ssim


def calc_ssim_no_bg(gt_images, pred_images):    
    ssim_no_bg = []
    for i in range(len(gt_images)):
        top = 37
        bottom = -37
        ssim_no_bg.append(structural_similarity(gt_images[i][top:bottom], pred_images[i][top:bottom], channel_axis=2, data_range=1.0))
    ssim_no_bg = np.mean(ssim_no_bg)
    return ssim_no_bg


def calc_psnr(gt_images, pred_images):
    mse = np.mean([((gt_images[i] - pred_images[i]) ** 2).mean() for i in range(len(gt_images))])
    psnr = 10 * np.log10(1.0 / mse)
    return psnr


def calc_psnr_no_bg(gt_images, pred_images):
    mse_no_bg = []
    for i in range(len(gt_images)):
        top = 37
        bottom = -37
        mse_no_bg.append(np.mean((gt_images[i][top:bottom] - pred_images[i][top:bottom]) ** 2))
    mse_no_bg = np.mean(mse_no_bg)
    psnr_no_bg = 10 * np.log10(1.0 / mse_no_bg)
    return psnr_no_bg


def calc_lpips(gt_images, pred_images):
    vgg16 = get_feature_detector(vgg16_url, device="cuda")
    gt_images_tensor = 255.0 * torch.tensor(np.array(gt_images)).permute(0,3,1,2).cuda()
    pred_images_tensor = 255.0 * torch.tensor(np.array(pred_images)).permute(0,3,1,2).cuda()

    gt_features = vgg16(gt_images_tensor, resize_images=False, return_lpips=True)
    pred_features = vgg16(pred_images_tensor, resize_images=False, return_lpips=True)
    lpips = (gt_features - pred_features).square().sum(1).mean()

    return lpips


def calc_lpips_no_bg(gt_images, pred_images):
    vgg16 = get_feature_detector(vgg16_url, device="cuda")

    gt_features = []
    pred_features = []
    lpips_no_bg = []
    for i in range(len(gt_images)):
        top = 37
        bottom = -37
        gt_image_tensor = 255.0 * torch.tensor(np.array(gt_images[i:i+1])[:, top:bottom], dtype=torch.float32).permute(0,3,1,2).cuda()
        pred_image_tensor = 255.0 * torch.tensor(np.array(pred_images[i:i+1])[:, top:bottom], dtype=torch.float32).permute(0,3,1,2).cuda()
        gt_features = vgg16(gt_image_tensor, resize_images=False, return_lpips=True)
        pred_features = vgg16(pred_image_tensor, resize_images=False, return_lpips=True)
        lpips_no_bg.append((gt_features - pred_features).square().sum().item())
    lpips_no_bg = np.mean(lpips_no_bg)

    return lpips_no_bg

def load_images_array(dir, image_type="renders"):
    assert image_type in {"renders", "normal"}

    gt_paths = sorted(glob(os.path.join(dir, f"gt/*{image_type}*png")))
    pred_paths = sorted(glob(os.path.join(dir, f"pred/*{image_type}*png")))
    assert len(gt_paths) == len(pred_paths)
    assert len(gt_paths) > 0

    # .resize((128, 128), Image.LANCZOS)
    gt_images = [np.array(Image.open(gt_path).convert("RGB")).astype(np.float32) / 255.0 for gt_path in gt_paths]
    pred_images = [np.array(Image.open(pred_path)).astype(np.float32) / 255.0 for pred_path in pred_paths]
    return gt_images, pred_images

def rgb2rgray(image):
    return 0.2989 * image[:, :, 0] + 0.5870 * image[:, :, 1] + 0.1140 * image[:, :, 2]

def run_metrics(dir, save_diff_images=True):
    gt_images, pred_images = load_images_array(dir)

    psnr = calc_psnr(gt_images, pred_images)
    ssim = calc_ssim(gt_images, pred_images)
    lpips = calc_lpips(gt_images, pred_images)

    psnr_no_bg = calc_psnr_no_bg(gt_images, pred_images)
    ssim_no_bg = calc_ssim_no_bg(gt_images, pred_images)
    lpips_no_bg = calc_lpips_no_bg(gt_images, pred_images)

    if save_diff_images:
        gt_gray = [rgb2rgray(img) for img in gt_images]
        pred_gray = [rgb2rgray(img) for img in pred_images]
        diff_images = [Image.fromarray((255 * np.abs(gt - pred).clip(0, 1)).astype(np.uint8)) for (gt, pred) in zip(*(gt_gray, pred_gray))]
        for i in range(len(diff_images)):
            filename = os.path.join(dir, "pred", f"{str(i).zfill(4)}_diff.png")
            diff_images[i].save(filename)

        

    print("Results on full images:")
    print(f"PSNR {psnr:.2f} - SSIM {ssim:.3f} - LPIPS {lpips:.4f}")
    print()
    print("Results without black background:")
    print(f"PSNR {psnr_no_bg:.2f} - SSIM {ssim_no_bg:.3f} - LPIPS {lpips_no_bg:.4f}")

    return (psnr, ssim, lpips), (psnr_no_bg, ssim_no_bg, lpips_no_bg)
