# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import math
import cv2 as cv
import torch
import torch.nn.functional as F
from einops import rearrange
from kornia.filters import joint_bilateral_blur, median_blur
from kornia.morphology import erosion, dilation
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import adjust_hue

import dnnlib
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from training.networks_stylegan2 import MappingNetwork
from training.networks_stylegan2 import SynthesisLayer
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_sampler import RaySampler
from training.volumetric_rendering.math_utils import torch_dot


def get_azimuth(vector):
    """
    Azimuth angle is the angle in zx plane, it can go from 0 to 2 * pi (full circle). We normalize to [-1, 1]
    """
    azimuth_angle = torch.atan2(vector[:, :, :, 2:3], vector[:, :, :, 0:1]) / math.pi
    return azimuth_angle


def get_elevation(vector):
    """
    Elevation angle is the angle starting from the y (vertical) axis, goes from [0, pi]. We normalize to [-1, 1]
    """
    elevation_angle = 2.0 * torch.arccos(torch.clamp(vector[:, :, :, 1:2], min=-0.9999, max=0.9999)) / math.pi - 1.0
    return elevation_angle


def dirs2envmap_coords(dirs):
    azimuth = get_azimuth(dirs)
    elevation = get_elevation(dirs)
    coords = torch.cat([azimuth, elevation], -1)
    return coords


def linear_to_srgb(input, gamma=2.4):
    input = input / (input + 1.0)
    input = torch.where(input <= 0.0031308, input * 12.92, 1.055 * abs(input) ** (1 / gamma) - 0.055)
    return input


def vec_dot(x, y, dim_=-1):
    return torch.sum(x * y, dim=dim_, keepdim=True)


def vec_length(x, dim=-1, eps = 1e-20):
    return torch.sqrt(torch.clamp(vec_dot(x, x, dim), min=eps))


def safe_normalize(x, eps = 1e-20):
    return x / vec_length(x, eps=eps)


def ground_hdri_mapping(worldspace_coords, camera_coords, scale = 10.0, offset = (0,0,0)):
    
    direction = torch.zeros_like(camera_coords)
    direction[..., 1] = 1.0 
    offset_vec = torch.zeros_like(camera_coords)
    for idx in [0, 1, 2]:
        offset_vec[..., idx] = offset[idx]
    camera_coords = camera_coords - offset_vec
    ndir = safe_normalize(direction)
    dotCD = vec_dot(camera_coords, ndir)
    dotWD = vec_dot(worldspace_coords, ndir)
    a = safe_normalize(- worldspace_coords * (dotCD / dotWD) + camera_coords - scale * ndir)
    return torch.lerp(a, worldspace_coords, torch.clamp((dotWD >= 0).type(torch.float32), min=0.0, max=1.0))


def get_ground_shadow(
    world_pos, 
    size=(0.5, 0.75), 
    round=torch.tensor([0.5,0.5,0.5,0.5]), 
    radius=0.2):

    def smoothstep(edge0, edge1, x):
        t = torch.clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def sdRoundBox(pos, b, r):
        new_rx = torch.where(pos[..., 0] > 0.0, r[0], r[2]) #p.x
        new_ry = torch.where(pos[..., 0] > 0.0, r[1], r[3]) #p.x
        new_rx = torch.where(pos[..., 2] > 0.0, new_rx, new_ry) #p.y
        q = torch.zeros_like(pos)
        q[..., 0] = torch.abs(pos[..., 0]) - b[0] + new_rx
        q[..., 2] = torch.abs(pos[..., 2]) - b[1] + new_rx
        d = torch.min(
             torch.max(q[..., 0], q[..., 2]), 
             torch.zeros_like(q[..., 0])) + vec_length(
                torch.max(q, torch.zeros_like(q))).squeeze() - new_rx
        return torch.cat([d.unsqueeze(-1), d.unsqueeze(-1), d.unsqueeze(-1)], axis=-1)
    
    round = torch.min(round, min(size[0],size[1]) * torch.ones_like(round))
    d = sdRoundBox(world_pos, size, round.to(world_pos.device))

    shadow = torch.where(
        d > 0.0, 
        torch.ones_like(world_pos), 
        torch.where(
            d < -radius, 
            torch.zeros_like(world_pos), 
            1. - smoothstep(0.0, 1.0, - d / radius)
        ))
	
    return torch.where(world_pos[..., 1] > 0, torch.ones_like(shadow[..., 0]), shadow[..., 0])[:, None, ...]

@persistence.persistent_class
class TriPlaneGenerator(torch.nn.Module):
    def __init__(
        self,
        z_dim,  # Input latent (Z) dimensionality.
        c_dim,  # Conditioning label (C) dimensionality.
        w_dim,  # Intermediate latent (W) dimensionality.
        img_resolution,  # Output resolution.
        img_channels,  # Number of output color channels.
        sr_num_fp16_res=0,
        mapping_kwargs={},  # Arguments for MappingNetwork.
        rendering_kwargs={},
        sr_kwargs={},
        **synthesis_kwargs,  # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.renderer = ImportanceRenderer()
        self.ray_sampler = RaySampler(rendering_kwargs.get("image_resolution")[0])
        self.backbone = StyleGAN2Backbone(
            z_dim,
            c_dim,
            w_dim,
            img_resolution=rendering_kwargs.get("plane_resolution", 256),
            img_channels=32, # now it's only one plane
            mapping_kwargs=mapping_kwargs,
            **synthesis_kwargs,
        )
        self.cam_mapping = MappingNetwork(
            z_dim=0,
            c_dim=16,
            w_dim=w_dim,
            num_layers=2,
            num_ws=1,
            w_avg_beta=None
        )
        self.cam_adjustment = SynthesisLayer(
            in_channels=z_dim,
            out_channels=4,
            w_dim=w_dim,
            resolution=0,
            kernel_size=1,
            use_noise=False
        )
        self.superresolution = dnnlib.util.construct_class_by_name(
            class_name=rendering_kwargs["superresolution_module"],
            channels=3,
            img_resolution=img_resolution,
            sr_num_fp16_res=sr_num_fp16_res,
            sr_antialias=rendering_kwargs["sr_antialias"],
            **sr_kwargs,
        )
        self.decoder = OSGDecoder(
            32, {"decoder_lr_mul": rendering_kwargs.get("decoder_lr_mul", 1), "decoder_output_dim": 3}
        )
        self.neural_rendering_resolution = 64
        self.rendering_kwargs = rendering_kwargs
        self._last_planes = None
        brdf_lut_path = './gancars/training/volumetric_rendering/fresnel_lut.hdr'
        assert os.path.exists(brdf_lut_path)
        self.brdf_lut = torch.from_numpy(cv.cvtColor(cv.imread(brdf_lut_path, cv.IMREAD_ANYDEPTH|cv.IMREAD_ANYCOLOR), cv.COLOR_BGR2RGB))
        self.envmap_lowpass = GaussianBlur(5)


    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs["c_gen_conditioning_zero"]:
            c = torch.zeros_like(c)
        return self.backbone.mapping(
            z,
            c * self.rendering_kwargs.get("c_scale", 0),
            truncation_psi=truncation_psi,
            truncation_cutoff=truncation_cutoff,
            update_emas=update_emas,
        )

    def synthesis(
        self,
        ws,
        c,
        specular_envmap,
        diffuse_envmap,
        z_cam=None,
        neural_rendering_resolution=None,
        update_emas=False,
        cache_backbone=False,
        use_cached_backbone=False,
        shading_kwargs = {},
        ground_kwargs = {},
        **synthesis_kwargs,
    ):
        extrinsics = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if z_cam is not None:
            intrinsics, extrinsics = self.adjust_camera(z_cam, intrinsics, extrinsics, update_emas=update_emas)

        scale_env = shading_kwargs.get("scale_env", 3.0)

        specular_envmap = specular_envmap * scale_env
        diffuse_envmap = diffuse_envmap * scale_env

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(extrinsics, intrinsics, neural_rendering_resolution)

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and self._last_planes is not None:
            planes = self._last_planes
        else:
            planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        if cache_backbone:
            self._last_planes = planes

        # Perform volume rendering
        feature_samples, depth_samples, normal_image, weights_samples, normal_coords = self.renderer(planes, self.decoder, ray_origins, ray_directions, self.rendering_kwargs)


        # Reshape into 'raw' neural-rendered image
        H = W = self.neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)

        if shading_kwargs.get("hue_shift", 0) != 0:
            feature_image = adjust_hue(feature_image, shading_kwargs.get("hue_shift", 0))

        basecolor = feature_image
        if shading_kwargs.get("filter_normals", True):
            filtered_normal = joint_bilateral_blur(rearrange(normal_image, " b (h w) c -> b c h w", h=H), basecolor.detach(), kernel_size=(9, 9), sigma_color=0.1, sigma_space=(1.5, 1.5), border_type='reflect', color_distance_type='l1')
            normal_image = rearrange(filtered_normal, "b c h w -> b (h w) c")
            normal_image = F.normalize(normal_image, dim=-1)

        reflect_directions =  2.0 * torch.sum(- ray_directions * normal_image, dim=-1, keepdim=True) * normal_image + ray_directions
        reflect_directions = reflect_directions / (torch.norm(reflect_directions, dim=-1, keepdim=True) + 1e-8)

        background_color = torch.zeros(size=(N, 3, H, W)).to(planes.device)

        base_color_image = feature_image.clone()

        # https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/pbr.frag
        metallic = shading_kwargs.get("metallic", 0.7)
        diffuseColor = (1 - metallic) * basecolor
        baseF0 = 0.04 * (1 - metallic) + metallic * basecolor
        baseAlpha = shading_kwargs.get("roughness", 0.2) ** 2

        nDotv = torch.clamp(torch_dot(-ray_directions, normal_image), min=0, max=1)
        
        diffuse = diffuseColor * self.get_diffuse_color(diffuse_envmap, normal_image.view(N, H, W, 3))
        scale_specular = shading_kwargs.get("scale_specular", 1/4)
        scaled_specular_envmap = F.interpolate(specular_envmap, scale_factor=scale_specular, mode="bilinear", antialias=True)
        if shading_kwargs.get("apply_lowpass", True):
            scaled_specular_envmap = self.envmap_lowpass(scaled_specular_envmap)
        specular = self.get_specular_color(scaled_specular_envmap, reflect_directions.view(N, H, W, 3), nDotv, baseAlpha, baseF0)

        if shading_kwargs.get("use_clear_coat", False):
            clearcoatF0         = 0.04
            clearcoatF90        = 1.0
            clearcoatFactor     = 1.0
            clearcoatRoughness  = 0.05 # ~0.0

            clearcoatFresnel    = self.F_Schlick(clearcoatF0, nDotv)
            clearcoat           = self.get_specular_color(scaled_specular_envmap, reflect_directions.view(N, H, W, 3), nDotv, clearcoatRoughness, clearcoatF0, clearcoatF90)
            clearcoatFresnel    = clearcoatFresnel.repeat(1,1,3).reshape(clearcoat.shape)
            clearcoat           = clearcoatFactor * clearcoat

            lr_image = (diffuse + specular) * (1.0 - clearcoatFactor * clearcoatFresnel) + clearcoat
        
        else:
            lr_image = (diffuse + specular)

        alpha = weights_samples.view(N, 1, H, W)
        lr_image = linear_to_srgb(alpha * lr_image + (1-alpha) * background_color) * 2.0 - 1.0

        # Run superresolution to get final image
        sr_image = self.superresolution(
            lr_image,
            ws,
            noise_mode=self.rendering_kwargs["superresolution_noise_mode"],
            **{k: synthesis_kwargs[k] for k in synthesis_kwargs.keys() if k != "noise_mode"},
        )

        # Prepare for visualization

        nDotv = (alpha * nDotv.view(N, 1, H, W)) * 2.0 - 1.0
        normal_image = normal_image.permute(0, 2, 1).view(N, 3, H, W)
        background_color = linear_to_srgb(background_color) * 2.0 - 1.0
        diffuse = (alpha * diffuse) * 2.0 - 1.0
        specular = (alpha * specular) * 2.0 - 1.0
        base_color_image = base_color_image * 2.0 - 1.0

        sr_alpha = F.interpolate(alpha, size=sr_image.shape[-1], mode="bicubic", antialias=True).clip(0, 1)
        if shading_kwargs.get("filter_mask", True):
            sr_alpha = erosion(sr_alpha, torch.ones(3, 3, device=sr_alpha.device))
            sr_alpha = dilation(sr_alpha, torch.ones(1, 1, device=sr_alpha.device))
            sr_alpha = median_blur(sr_alpha, 11)

        ground_scale = ground_kwargs.get("scale", 0.55)
        ground_offset_y = ground_kwargs.get("offset", -0.15)
        shadow_size_x = ground_kwargs.get("size_x", 0.5)
        shadow_size_y = ground_kwargs.get("size_y", 0.75)
        shadow_radius = ground_kwargs.get("radius", 0.3)
        sr_background = F.interpolate(
            self.get_background_color(
                specular_envmap, 
                ray_directions.view(N, H, W, 3), 
                ray_origins.view(N, H, W, 3),
                ground_kwargs,
            ), 
            size=sr_image.shape[-1], 
            mode="bilinear")
        sr_composited = sr_image * sr_alpha + (1 - sr_alpha) * (linear_to_srgb(sr_background) * 2.0 - 1.0)

        alpha = alpha  * 2.0 - 1.0
        specular_envmap = linear_to_srgb(specular_envmap) * 2.0 - 1.0

        results = {
            "sr_image": sr_image, 
            "lr_image": lr_image, 
            "alpha": alpha, 
            "image_depth": depth_image, 
            "background": background_color,
            "image_env": specular_envmap,
            "image_normal": normal_image,
            "base_color_image": base_color_image, 
            "diffuse": diffuse,
            "specular": specular,
            "sr_composited": sr_composited,
            "normal_grayscale": nDotv,
            "normal_coords": normal_coords,
            "planes": planes,
        }
        return results

    # if we change the roughness, then we need the IBL mipmaps
    def get_ibl_sample(self, env_map, directions):
        env_map_coords_reflect = dirs2envmap_coords(directions)
        envmap_color = F.grid_sample(env_map, env_map_coords_reflect, align_corners=True, mode='bilinear', padding_mode='reflection')
        return envmap_color
    
    def get_background_color(self, env_map, world_pos, camera_coords, ground_kwargs):
        if ground_kwargs.get("use_ground_shadow", False):
            scale = ground_kwargs.get("scale", 0.55)
            offset = (0, ground_kwargs.get("offset", -0.15), 0)
            world_pos = ground_hdri_mapping(world_pos, camera_coords, scale, offset)
            env_map_coords = dirs2envmap_coords(world_pos)
            background_color = F.grid_sample(env_map, env_map_coords, align_corners=True, mode='bilinear', padding_mode='border')
            size=(ground_kwargs.get("size_x", 0.5), ground_kwargs.get("size_y", 0.75))
            radius=ground_kwargs.get("radius", 0.3)
            ground_shadow = get_ground_shadow(world_pos, size=size, radius=radius)
            return background_color * ground_shadow

        env_map_coords = dirs2envmap_coords(world_pos)
        background_color = F.grid_sample(env_map, env_map_coords, align_corners=True, mode='bilinear', padding_mode='border')
        return background_color

    # from https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/brdf.glsl
    def F_Schlick(self, f0, VdotH):
        f90 = 1.0 # clamp(50.0 * f0, 0.0, 1.0)
        x = torch.clip(1.0 - VdotH, min=0.0, max=1.0)
        x2 = x * x
        x5 = x * x2 * x2
        return f0 + (f90 - f0) * x5

    # from https://google.github.io/filament/Filament.md.html#lighting/imagebasedlights
    def get_diffuse_color(self, env_map, normal):
        diffuse = self.get_ibl_sample(env_map, normal)
        return diffuse # 1/pi lambertian term directly encoded into the prefiltered diffuse

    # from https://google.github.io/filament/Filament.md.html#lighting/imagebasedlights
    def get_specular_color(self, env_map, reflect_dir, cos_elevation, roughness, F0, F90 = 1.0):
        cos_elevation = cos_elevation.unsqueeze(-1)
        indirect_specular = self.get_ibl_sample(env_map, reflect_dir)

        # lazy copy to GPU
        if not self.brdf_lut.is_cuda:
            self.brdf_lut = self.brdf_lut.to(cos_elevation.device)
        
        lut_values = F.grid_sample(
            self.brdf_lut.unsqueeze(0).repeat(cos_elevation.shape[0], 1, 1, 1).permute(0, 3, 1, 2), 
            torch.cat([2.0 * cos_elevation - 1.0, torch.ones_like(cos_elevation)*roughness * 2.0 - 1.0], dim=2).unsqueeze(-2), 
            align_corners=True, 
            mode='bilinear', 
            padding_mode='zeros')
        lut_x = lut_values[:, 0:1, :, 0]
        lut_y = lut_values[:, 1:2, :, 0]
        lut_x = lut_x.repeat(1,1,3).reshape(indirect_specular.shape)
        lut_y = lut_y.repeat(1,1,3).reshape(indirect_specular.shape)

        specularColor = F0 * lut_x + F90 * lut_y

        return indirect_specular * specularColor

    def adjust_camera(self, z_cam, intrinsics, extrinsics, update_emas=False):
        ws_cam = self.cam_mapping(None, extrinsics.view(-1, 16), update_emas=update_emas)
        fov_bias = 0.2
        c_x_bias = 0.02
        c_y_bias = 0.1
        depth_bias = 0.1

        cam_adj = torch.tanh(self.cam_adjustment(z_cam.unsqueeze(-1).unsqueeze(-1), ws_cam[:, 0]))

        adjusted_intrinsics = intrinsics.clone().detach()

        # FOV adjustment
        adjusted_intrinsics[:, 0, 0] += cam_adj[:, 0, 0, 0] * fov_bias
        adjusted_intrinsics[:, 1, 1] += cam_adj[:, 0, 0, 0] * fov_bias
        
        # c_x, c_y adjustment
        adjusted_intrinsics[:, 0, 2] += cam_adj[:, 1, 0, 0] * c_x_bias
        adjusted_intrinsics[:, 1, 2] += cam_adj[:, 2, 0, 0] * c_y_bias

        # Camera distance adjustment
        adjusted_extrinsics = extrinsics.clone().detach()
        lookat_dir = F.normalize(extrinsics[:, :3, -1], p=2, dim=1)
        adjusted_extrinsics[:, :3, -1] += cam_adj[:, 3, 0] * lookat_dir * depth_bias

        return adjusted_intrinsics, adjusted_extrinsics

    def sample(
        self,
        coordinates,
        directions,
        z_obj,
        c,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        **synthesis_kwargs,
    ):
        # Compute RGB features, density for arbitrary 3D coordinates. Mostly used for extracting shapes.
        ws = self.mapping(z_obj, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        # planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])
        out = self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)
        return out

    def sample_mixed(
        self,
        coordinates,
        directions,
        ws,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        **synthesis_kwargs,
    ):
        # Same as sample, but expects latent vectors 'ws' instead of Gaussian noise 'z'
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        out = self.renderer.run_model(planes, self.decoder, coordinates, directions, self.rendering_kwargs)
        return out

    def sample_planes(
        self,
        coordinates,
        directions,
        ws,
        truncation_psi=1,
        truncation_cutoff=None,
        update_emas=False,
        **synthesis_kwargs,
    ):
        planes = self.backbone.synthesis(ws, update_emas=update_emas, **synthesis_kwargs)
        return planes

    def forward(
        self,
        z,
        c,
        specular_envmap,
        diffuse_envmap,
        truncation_psi=1,
        truncation_cutoff=None,
        neural_rendering_resolution=None,
        update_emas=False,
        cache_backbone=False,
        use_cached_backbone=False,
        **synthesis_kwargs,
    ):
        # Render a batch of generated images.
        z_obj = z[:, :self.z_dim]
        z_cam = z[:, self.z_dim:]
        ws = self.mapping(z_obj, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(
            ws,
            c,
            specular_envmap,
            diffuse_envmap,
            z_cam=z_cam,
            update_emas=update_emas,
            neural_rendering_resolution=neural_rendering_resolution,
            cache_backbone=cache_backbone,
            use_cached_backbone=use_cached_backbone,
            **synthesis_kwargs,
        )


from training.networks_stylegan2 import FullyConnectedLayer


class OSGDecoder(torch.nn.Module):
    def __init__(self, n_features, options):
        super().__init__()
        self.hidden_dim = 64

        self.shared = torch.nn.Sequential(
            FullyConnectedLayer(n_features, self.hidden_dim, lr_multiplier=options["decoder_lr_mul"]),
            torch.nn.Softplus(),
        )
        self.to_color = FullyConnectedLayer(self.hidden_dim, options["decoder_output_dim"], lr_multiplier=options["decoder_lr_mul"])
        self.to_density = FullyConnectedLayer(self.hidden_dim, 1, lr_multiplier=options["decoder_lr_mul"])

    def forward(self, sampled_features, ray_directions):
        # Aggregate features
        sampled_features = sampled_features.mean(1)
        x = sampled_features

        N, M, C = x.shape
        x = x.view(N * M, C)

        x = self.shared(x)
        rgb = torch.sigmoid(self.to_color(x)) * (1 + 2 * 0.001) - 0.001  # Uses sigmoid clamping from MipNeRF
        sigma = self.to_density(x)
        return {"rgb": rgb.view(N, M, -1), "sigma": sigma.view(N, M, -1)}
