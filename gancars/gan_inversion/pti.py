
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import cv2 as cv
import json
import torch
import numpy as np
from tqdm import tqdm
from tqdm import tqdm
from torch.optim import Adam
import torch.nn.functional as F
from einops import rearrange, repeat
from cam_utils import blender2eg3d

from abc import ABC, abstractmethod

from dnnlib_util import get_feature_detector, vgg16_url
from torchvision.transforms.functional import to_pil_image


vgg16 = get_feature_detector(vgg16_url, device="cuda")


class GANWraper(ABC):
    @abstractmethod
    def __call__(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def model(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def w_init(self):
        raise NotImplementedError
    

def project(
    outdir, 
    G, 
    real_image, 
    real_c, 
    num_steps=500, 
    learning_rate=0.1,
    lr_rampdown_length=0.25, 
    lr_rampup_length=0.05, 
    ):

    device = real_image.device
    batch_size = real_image.shape[0]
    os.makedirs(os.path.join(outdir, "w_inversion"), exist_ok=True)
    
    w_pivot = G.w_init.to(device)
    optimizer = Adam([w_pivot], betas=(0.9, 0.999), lr=learning_rate)

    target_images = (real_image.to(device).to(torch.float32) + 1.0) * (255/2)
    real_features = vgg16(target_images, resize_images=False, return_lpips=True)
    l2_criterion = torch.nn.MSELoss(reduction='mean')

    pbar = tqdm(initial=0, total=int(num_steps))
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        fake_image = G(w_pivot.repeat(batch_size, G.model.backbone.num_ws, 1), real_c)
        fake_image_save = fake_image.clone().detach().cpu()
        fake_image = (fake_image + 1.0) * (255.0 / 2.0)
        assert not torch.any(torch.isnan(fake_image)).item()

        fake_features = vgg16(fake_image, resize_images=False, return_lpips=True)

        lpips_loss = (fake_features - real_features).square().sum(1).mean()

        # MSE loss
        mse_loss = l2_criterion(target_images, fake_image)

        loss = lpips_loss + mse_loss

        assert not torch.any(torch.isnan(lpips_loss)).item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pbar.update(1)
        pbar.set_description(f"LPIPS Loss {lpips_loss.item():.4f} - MSE Loss {mse_loss.item():.4f}")
        if step % 10 == 0:
            for view in range(batch_size):
                filename = os.path.join(outdir, "w_inversion",f"inversion_view{view}_{step:04d}.png")
                image = to_pil_image(fake_image_save[view].clip(-1, 1) * 0.5 + 0.5)
                image.save(filename)
    return w_pivot.clone().detach()


def pivotal_tuning(
    outdir,
    G,
    real_image,
    real_c,
    w_pivot,
    num_steps=500,
    learning_rate = 3e-4,
    lr_rampdown_length=0.25, 
    lr_rampup_length=0.05, 
):  
    device = real_image.device
    os.makedirs(os.path.join(outdir, "pivotal"), exist_ok=True)
    from copy import deepcopy
    G_original = deepcopy(G)
    G_original.model.eval().to(device)
    G.model.requires_grad_(True).to(device)
    w_pivot.detach().requires_grad_(False)

    # l2 criterion
    l2_criterion = torch.nn.MSELoss(reduction='mean')

    # Features for target image.
    target_images = (real_image[:, :, 75:-75].to(device).to(torch.float32) + 1.0) * (255/2)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(target_images, size=(256, 256), mode='area')
    real_features = vgg16(target_images, resize_images=False, return_lpips=True)

    # initalize optimizer
    optimizer = Adam(G.model.parameters(), lr=learning_rate)
    batch_size = real_image.shape[0]


    # run optimization loop
    pbar = tqdm(initial=0, total=int(num_steps))
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        fake_image = G(w_pivot.repeat(batch_size, G.model.backbone.num_ws, 1), real_c)[:, :, 75:-75]
        fake_image_save = fake_image.clone().detach()
        fake_image = (fake_image + 1.0) * (255.0 / 2.0)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if fake_image.shape[2] > 256:
            fake_image = F.interpolate(fake_image, size=(256, 256), mode='area')
        
        # LPIPS loss
        fake_features = vgg16(fake_image, resize_images=False, return_lpips=True)
        lpips_loss = (fake_features - real_features).square().sum(1).mean()
        
        # MSE loss
        mse_loss = l2_criterion(target_images, fake_image)

        # space regularizer
        reg_loss = space_regularizer_loss(G, G_original, w_pivot, vgg16, real_c)

        # Step
        optimizer.zero_grad()
        loss = mse_loss + lpips_loss + reg_loss
        loss.backward()
        optimizer.step()

        pbar.set_description(f"LPIPS Loss {lpips_loss.item():.4f} - MSE Loss {mse_loss.item():.4f} - Reg Loss {reg_loss.item():.4f}")
        pbar.update(1)
        if step % 10 == 0:
            for view in range(batch_size):
                filename = os.path.join(outdir, "pivotal", f"pivotal_view{view}_{step:04d}.png")
                image = to_pil_image(fake_image_save[view].clip(-1, 1) * 0.5 + 0.5)
                image.save(filename)
    return G


def get_morphed_w_code(new_w_code, fixed_w, regularizer_alpha=30):
    interpolation_direction = new_w_code - fixed_w
    interpolation_direction_norm = torch.norm(interpolation_direction, p=2)
    direction_to_move = regularizer_alpha * interpolation_direction / interpolation_direction_norm
    result_w = fixed_w + direction_to_move
    return result_w


def space_regularizer_loss(
    G_pti,
    G_original,
    w_batch,
    vgg16,
    real_c,
    lpips_lambda=10,
):

    num_of_sampled_latents = real_c.shape[0]
    z_samples = np.random.randn(num_of_sampled_latents, G_original.model.z_dim)
    z_samples = torch.from_numpy(z_samples).to(w_batch.device)

    w_samples = G_original.model.mapping(z_samples, real_c, truncation_psi=0.5)
    territory_indicator_ws = torch.cat([get_morphed_w_code(w_code.unsqueeze(0), w_batch) for w_code in w_samples], 0)

    new_img = G_pti(territory_indicator_ws, real_c)
    old_img = G_original(territory_indicator_ws, real_c)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
    if new_img.shape[-1] > 256:
        new_img = F.interpolate(new_img, size=(256, 256), mode='area')
        old_img = F.interpolate(old_img, size=(256, 256), mode='area')

    new_feat = vgg16(new_img, resize_images=False, return_lpips=True)
    old_feat = vgg16(old_img, resize_images=False, return_lpips=True)
    lpips_loss = lpips_lambda * (old_feat - new_feat).square().sum(1).mean()

    return lpips_loss / territory_indicator_ws.shape[0]


def run_pti(outdir, G, real_image, real_c):
    # w_pivot = torch.load(os.path.join(outdir, "w_pivot.pt"))[:, 0]
    w_pivot = project(outdir, G, real_image, real_c)
    G_pti = pivotal_tuning(outdir, G, real_image, real_c, w_pivot)
    return w_pivot.repeat(1, G.model.backbone.num_ws, 1), G_pti


def load_images_and_cams(invdata_path, mode="train"):
    assert mode in {"train", "test"}

    img_path_list = []
    json_path = os.path.join(invdata_path, f"transforms_{mode}.json")

    with open(json_path, 'r') as json_file:
        jsondata = json.load(json_file)

    cam_list = []
    for idx in range(len(jsondata["frames"])):
        cam_list.append(np.array(jsondata["frames"][idx]["transform_matrix"]))
        img_id = jsondata["frames"][idx]["file_path"].replace("\\", "/").split("/")[-1] 
        img_path_list.append(os.path.join(invdata_path, "train", img_id))

    real_image = []
    for filepath in img_path_list:
        img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
        img = (img[:, :, :3].astype(np.float32) * (img[:, :, 3:4].astype(np.float32)/ 255.0)).astype(np.uint8)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        real_image.append(img)

    real_image = 2.0 * np.array(real_image, dtype=np.float32) / 255.0 - 1.0
    real_image = torch.from_numpy(real_image).float()
    real_image = rearrange(real_image, " b h w c -> b c h w")

    cam2world = np.array([blender2eg3d(cam) for cam in cam_list])
    cam2world[:, :3, -1] = 1.2 * cam2world[:, :3, -1] / np.linalg.norm(cam2world[:, :3, -1], axis=1).max()
    cam2world = torch.from_numpy(cam2world)

    f_x = 355.55555555555554 # unnormalized focal length x
    f_y = 355.55555555555554 # unnormalized focal length y
    x0 = 128.0               # unnormalized principal point offset x
    y0 = 128.0               # unnormalized principal point offset y
    intrinsics = torch.tensor([[f_x / 256.0, 0, x0 / 256.0], [0, f_y / 256.0, y0 / 256.0], [0, 0, 1]])

    assert real_image.shape[0] == cam2world.shape[0]
    real_c = torch.cat([cam2world.reshape(-1, 16), repeat(intrinsics, "h w -> b (h w)", b=cam2world.shape[0])], 1).float()

    return real_image, real_c


def save_img_group_as_exr(outdir, img_list_vec, img_type_str):
    
    os.makedirs(outdir, exist_ok=True)
    img_list_vec = img_list_vec.clip(-1, 1)
    for view in range(img_list_vec.shape[0]):
        filename = outdir + f"/gancars_test_{img_type_str}_{view}.exr"
        cv.imwrite(filename, cv.cvtColor(np.array(img_list_vec[view].permute(1, 2, 0).detach().cpu().numpy(), 
            dtype=np.float32), cv.COLOR_RGB2BGR))

def save_img_group_as_png(outdir, img_list_vec, img_type_str):
    
    os.makedirs(outdir, exist_ok=True)
    img_list_vec = img_list_vec.clip(-1, 1)
    for view in range(img_list_vec.shape[0]):
        filename = outdir + f"/gancars_test_{img_type_str}_view{view}.png"
        if torch.is_tensor(img_list_vec[view]):
            cv.imwrite(filename, cv.cvtColor(np.array(255 * (0.5 * img_list_vec[view].permute(1, 2, 0).detach().cpu().numpy() + 0.5), dtype=np.uint8), cv.COLOR_RGB2BGR))
        elif isinstance(img_list_vec[view], np.ndarray):
            cv.imwrite(filename, cv.cvtColor(np.array(255 * (0.5 * img_list_vec[view] + 0.5), dtype=np.uint8), cv.COLOR_RGB2BGR))