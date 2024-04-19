import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
from tqdm import tqdm
from glob import glob
import cv2 as cv
import numpy as np
from tempfile import TemporaryDirectory
import click


def adjust_exposure_laval(image):
    per = 0.1533203125
    factor = per / np.percentile(image, 50)
    return image * factor


def compute_irradiance(cmgen_exe, specular_envmap_path):
    with TemporaryDirectory() as temp_dir:
        print("\nProcessing:", os.path.basename(specular_envmap_path))
        cmd_str = cmgen_exe 
        cmd_str += " --type=equirect"
        cmd_str += " --format=hdr"
        cmd_str += " --no-mirror"
        cmd_str += " --clamp"
        cmd_str += " --ibl-samples=4096"
        cmd_str += f" --ibl-irradiance=\"{temp_dir}\" "
        cmd_str += specular_envmap_path
        os.system(cmd_str)

        irradiance_temp_dir = os.path.join(temp_dir, os.path.basename(specular_envmap_path).split(".")[0], "irradiance.hdr")
        assert os.path.exists(irradiance_temp_dir)
        irradiance = cv.imread(irradiance_temp_dir, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        irradiance = np.nan_to_num(irradiance, nan=np.nanmax(irradiance))
        img_name = specular_envmap_path.split(os.sep)[-2]
        irradiance_path = os.path.join(os.path.dirname(specular_envmap_path), f"{img_name}_diffuse.hdr")
        cv.imwrite(irradiance_path, irradiance)

@click.command()
@click.option('--outdir',  type=str, required=True)
@click.option('--polyhaven_dir',  type=str, required=True)
@click.option('--ihdri_dir',  type=str, required=True)
@click.option('--laval_dir',  type=str, required=False, default=None)
@click.option('--cmgen_exe',  type=str, required=False, default=None)
def main(outdir, polyhaven_dir, ihdri_dir, laval_dir=None, cmgen_exe=None):
    h, w = 512, 1024

    polyhaven_paths = glob(os.path.join(polyhaven_dir, "*.hdr"))
    ihdri_paths = glob(os.path.join(ihdri_dir, "*.hdr"))
    if laval_dir is None:
        laval_paths = []
    else:
        laval_paths = glob(os.path.join(laval_dir, "*.hdr"))

    all_paths =  polyhaven_paths + ihdri_paths + laval_paths
    for idx, path in enumerate(tqdm(all_paths)):
        folder = str(idx // 1000).zfill(5)
        img_name = f"img{str(idx).zfill(8)}"
        filename = os.path.join(outdir, folder, img_name, f"{img_name}_specular.hdr")
        
        os.makedirs(os.path.join(outdir, folder, img_name), exist_ok=True)
        image = cv.imread(path, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH)
        image = np.nan_to_num(image, nan=np.nanmax(image), posinf=np.nanmax(image), neginf=0)
        if path in laval_paths:
            image = adjust_exposure_laval(image)
            
        image = cv.resize(image, [w, h], cv.INTER_LANCZOS4)
        cv.imwrite(filename, image)
        if cmgen_exe is not None:
            compute_irradiance(cmgen_exe, filename)

if __name__ == "__main__":
    main()