
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import sys
sys.path.append("./gancars/gan_inversion")
import cv2 as cv
import torch
import numpy as np
import dnnlib
import click
import legacy
from einops import repeat
import pickle
from glob import glob

from viz.envmap_select_widget import load_envmaps
from reload_modules import reload_modules
from gan_inversion.pti import load_images_and_cams, run_pti, save_img_group_as_png, GANWraper
from gan_inversion.calc_metrics import run_metrics


class GANCarsWraper(GANWraper):
    def __init__(self, G, specular_envmap, diffuse_envmap):
        self.specular_envmap = specular_envmap
        self.diffuse_envmap = diffuse_envmap
        self._G = G
        num_ws = G.backbone.num_ws
        w_avg = G.backbone.mapping.w_avg
        self._w_init = w_avg.clone().detach().requires_grad_(True)

    def __call__(self, w, real_c):
        shading_kwargs = {
            "scale_env": 1.0,
            "metallic": 0.7,
            "roughness": 0.2,
            "apply_lowpass": False,
        }
        fake_image = self.model.synthesis(w, real_c, self.specular_envmap, self.diffuse_envmap, noise_mode='const', 
                                          shading_kwargs=shading_kwargs)["sr_image"]
        return fake_image
    
    @property
    def model(self):
        return self._G
    
    @property
    def w_init(self):
        return self._w_init
        

def run_inversion_metrics(network, data_dir, envmap_dir, outdir):
    torch.manual_seed(1996)
    os.makedirs(outdir, exist_ok=True)
    device = torch.device('cuda')

    # Load envmaps
    print(f"Loading envmap from: {envmap_dir}")
    specular_envmap, diffuse_envmap = load_envmaps(envmap_dir, device)
    specular_envmap = torch.clip(specular_envmap, 1e-9)
    diffuse_envmap = torch.clip(diffuse_envmap, 1e-9)


    print(f"Loading networks from: {network}")
    with dnnlib.util.open_url(network) as f:
        G_data = legacy.load_network_pkl(f)
        G = G_data['G_ema'].to(device) # type: ignore
    G = reload_modules(G).to(device)

    # IMAGE INVERSION
    # ------------------------------------------------------------------------------------------------------------------
    # Load images & cameras
    print(f"Loading cams and train images from: {data_dir}")
    real_image_train, real_c_train = load_images_and_cams(data_dir, mode="train")
    specular_envmap_train = repeat(specular_envmap, "1 c h w -> b c h w", b=real_image_train.shape[0])
    diffuse_envmap_train = repeat(diffuse_envmap, "1 c h w -> b c h w", b=real_image_train.shape[0])
    G = GANCarsWraper(G, specular_envmap_train.to(device), diffuse_envmap_train.to(device))

    # Inversion 1st step: Pivot computation    

    w_pivot, G = run_pti(outdir, G, real_image_train.to(device), real_c_train.to(device))
    torch.save(w_pivot, os.path.join(outdir, "w_pivot.pt"))

    # Save PTI generator
    G_data["G_ema"] = G.model
    G_pti_pkl = os.path.join(outdir, f'network-snapshot-pti.pkl')
    with open(G_pti_pkl, 'wb') as f:
        pickle.dump(G_data, f)

    # Delete
    G_pti_pkl = os.path.join(outdir, f'network-snapshot-pti.pkl')
    print(f"Loading networks from: {G_pti_pkl}")
    with dnnlib.util.open_url(G_pti_pkl) as f:
        G_data = legacy.load_network_pkl(f)
        G = G_data['G_ema'].to(device) # type: ignore
    G = reload_modules(G).to(device)
    real_image_train, real_c_train = load_images_and_cams(data_dir, mode="train")
    specular_envmap_train = repeat(specular_envmap, "1 c h w -> b c h w", b=real_image_train.shape[0])
    diffuse_envmap_train = repeat(diffuse_envmap, "1 c h w -> b c h w", b=real_image_train.shape[0])
    G = GANCarsWraper(G, specular_envmap_train.to(device), diffuse_envmap_train.to(device))
    w_pivot = torch.load(os.path.join(outdir, "w_pivot.pt"))

    # GENERATE TEST VIEWS
    # ------------------------------------------------------------------------------------------------------------------
    # Load test cameras
    print(f"Loading cams and test images from: {data_dir}")
    real_image_test, real_c_test = load_images_and_cams(data_dir, mode="test")
    batch_size_test = real_image_test.shape[0]

    specular_envmap_test = repeat(specular_envmap, "1 c h w -> b c h w", b=batch_size_test)
    diffuse_envmap_test = repeat(diffuse_envmap, "1 c h w -> b c h w", b=batch_size_test)

    # Generate fake test images
    shading_kwargs = {
        "scale_env": 1.0,
        "metallic": 0.7,
        "roughness": 0.2,
        "apply_lowpass": False,
    }
    fake_synthesis = G.model.synthesis(w_pivot.repeat(batch_size_test, 1, 1), 
                                       real_c_test.to(device), 
                                       specular_envmap_test.to(device),
                                        diffuse_envmap_test.to(device), 
                                        noise_mode='const', 
                                        shading_kwargs=shading_kwargs)
    fake_renders = fake_synthesis["sr_image"]
    normal_imgs = fake_synthesis["image_normal"]
    fake_diffuse = fake_synthesis["base_color_image"]
    alpha_imgs = fake_synthesis["alpha"]

    normal_imgs = (0.5 * alpha_imgs + 0.5) * normal_imgs

    # Blender compatible
    fake_normals = normal_imgs.clone()
    # Z -> -Z
    fake_normals[:, 2, ...] = - fake_normals[:, 2, ...]
    # Z -> Y and Y -> Z
    fake_normals[:, 1, ...] = fake_normals[:, 2, ...]
    fake_normals[:, 2, ...] = normal_imgs[:, 1, ...]

    ## Output PNG for visualization
    outdir_metrics_pred = os.path.join(outdir, "metrics", "pred")
    outdir_metrics_gt = os.path.join(outdir, "metrics", "gt")
    os.makedirs(outdir_metrics_pred, exist_ok=True)
    os.makedirs(outdir_metrics_gt, exist_ok=True)

    img_groups = {"fake_normal": [fake_normals, outdir_metrics_pred], 
                  "fake_basecolor": [fake_diffuse, outdir_metrics_pred], 
                  "fake_renders": [fake_renders, outdir_metrics_pred], 
                  "gt_renders": [real_image_test, outdir_metrics_gt]
                  }
    for k, v in img_groups.items():
        save_img_group_as_png(v[1], v[0], k)

    for image_type in [("Normal", "gt_normal"), ("Diffuse", "gt_basecolor")]:
        paths = sorted(glob(os.path.join(data_dir, "test", f"{image_type[0]}*.exr")))
        gt = np.array([cv.resize(cv.cvtColor(cv.imread(path, cv.IMREAD_UNCHANGED), cv.COLOR_BGR2RGB), (128, 128), 
                                 interpolation=cv.INTER_LANCZOS4) for path in paths])
        save_img_group_as_png(outdir_metrics_gt, gt, image_type[1])
    print("DONE metric images!")

    run_metrics(os.path.join(outdir, "metrics"))
    print("DONE test metrics!")


@click.command()
@click.option('--network', type=str, required=True)
@click.option('--data_dir', type=str, required=True)
@click.option('--envmap_dir', type=str, required=True)
@click.option('--outdir', help='Output directory', type=str, required=True)
def main(**kwargs):
    run_inversion_metrics(**kwargs)

if __name__ == "__main__":
    main()