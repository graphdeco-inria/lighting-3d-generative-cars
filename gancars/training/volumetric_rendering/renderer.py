# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The renderer is a module that takes in rays, decides where to sample along each
ray, and computes pixel colors using the volume rendering equation.
"""
import torch
import torch.nn.functional as F
from torch.autograd import grad

from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.volumetric_rendering import math_utils
from einops import rearrange, repeat
import math

def scale(x, min, max):
    return (max - min) * x + min


def generate_planes():
    """
    Defines planes by the three vectors that form the "axes" of the
    plane. Should work with arbitrary number of planes and planes of
    arbitrary orientation.
    """
    return torch.tensor([[[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]],
                            [[1, 0, 0],
                            [0, 0, 1],
                            [0, 1, 0]],
                            [[0, 0, 1],
                            [0, 1, 0],
                            [1, 0, 0]]], dtype=torch.float32)


def project_onto_planes(planes, coordinates):
    """
    Does a projection of a 3D point onto a batch of 2D planes,
    returning 2D plane coordinates.

    Takes plane axes of shape n_planes, 3, 3
    # Takes coordinates of shape N, M, 3
    # returns projections of shape N*n_planes, M, 2
    """
    N, M, C = coordinates.shape
    n_planes=1
    coordinates = coordinates.unsqueeze(1).expand(-1, n_planes, -1, -1).reshape(N * n_planes, M, 3)
    inv_planes = torch.linalg.inv(planes).unsqueeze(0).expand(N, -1, -1, -1).reshape(N * n_planes, 3, 3)
    projections = torch.bmm(coordinates, inv_planes)
    projections = torch.zeros(N,)
    return projections[..., :2]


def sample_from_planes(plane_axes, plane_features, coordinates, bbox_min, bbox_max, mode="bilinear", padding_mode="zeros"):
    assert padding_mode == "zeros"
    N, C, H, W = plane_features.shape
    _, M, _ = coordinates.shape

    bbox_min = torch.tensor(bbox_min, dtype=coordinates.dtype, device=coordinates.device)
    bbox_max = torch.tensor(bbox_max, dtype=coordinates.dtype, device=coordinates.device)
    # # Debug: Grid Sample coords convetion is x=width, y=height (opposite of tensor convetion!)
    # from torchvision.transforms.functional import to_pil_image
    # yy, zz = torch.meshgrid(torch.linspace(0, 1, 100), torch.linspace(0, 1, 100))
    # coordinates = torch.cat([yy[..., None], zz[..., None], torch.zeros_like(yy)[..., None]], dim=-1)
    # coordinates = repeat(coordinates, "h w c -> b (h w) c",b=N).to(plane_features.device)

    coordinates = (coordinates - bbox_min)/(bbox_max - bbox_min)

    xy_coords = coordinates[..., [0, 1]]
    xz_coords = coordinates[..., [0, 2]]
    zy_coords = coordinates[..., [2, 1]]

    # Pairs of coordinates
    xy_coords[:, :, 0] = scale(xy_coords[:, :, 0], 0, 1)
    xy_coords[:, :, 1] = scale(xy_coords[:, :, 1], 0, 1)

    zy_coords[:, :, 0] = scale(zy_coords[:, :, 0], -1, 1)
    zy_coords[:, :, 1] = scale(zy_coords[:, :, 1], -1, 0)

    xz_coords[:, :, 0] = scale(xz_coords[:, :, 0], -1, 0) 
    xz_coords[:, :, 1] = scale(xz_coords[:, :, 1], 0, 1)

    projected_coordinates = torch.cat([xy_coords, xz_coords, zy_coords], dim=1).float().unsqueeze_(1)
    output_features = torch.nn.functional.grid_sample(plane_features, projected_coordinates, mode=mode, padding_mode=padding_mode, align_corners=False)
    output_features = rearrange(output_features, "b c 1 (n_planes m) -> b n_planes m c",n_planes=3)
    # # Debug: only for viz
    # output_features = torch.nn.functional.grid_sample(plane_features, xy_coords.unsqueeze(1), mode=mode, padding_mode=padding_mode, align_corners=False)
    # output_features = rearrange(output_features, "b c 1 (h w) -> b c h w", h=100)

    return output_features


def sample_from_3dgrid(grid, coordinates):
    """
    Expects coordinates in shape (batch_size, num_points_per_batch, 3)
    Expects grid in shape (1, channels, H, W, D)
    (Also works if grid has batch size)
    Returns sampled features of shape (batch_size, num_points_per_batch, feature_channels)
    """
    batch_size, n_coords, n_dims = coordinates.shape
    sampled_features = torch.nn.functional.grid_sample(
        grid.expand(batch_size, -1, -1, -1, -1),
        coordinates.reshape(batch_size, 1, 1, -1, n_dims),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    N, C, H, W, D = sampled_features.shape
    sampled_features = sampled_features.permute(0, 4, 3, 2, 1).reshape(N, H * W * D, C)
    return sampled_features


class ImportanceRenderer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ray_marcher = MipRayMarcher2()
        self.plane_axes = generate_planes()

    def forward(self, planes, decoder, ray_origins, ray_directions, rendering_options):
        self.plane_axes = self.plane_axes.to(ray_origins.device)

        if rendering_options["ray_start"] == rendering_options["ray_end"] == "auto":
            ray_start, ray_end = math_utils.get_ray_limits_box(
                ray_origins, ray_directions, bbox_min=rendering_options["bbox_min"], bbox_max=rendering_options["bbox_max"]
            )
            is_ray_valid = ray_end > ray_start
            if torch.any(is_ray_valid).item():
                ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
                ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
            depths_coarse = self.sample_stratified(
                ray_origins,
                ray_start,
                ray_end,
                rendering_options["depth_resolution"],
                rendering_options["disparity_space_sampling"],
                rendering_options.get("near_plane", 0.15),
            )
        else:
            # Create stratified depth samples
            depths_coarse = self.sample_stratified(
                ray_origins,
                rendering_options["ray_start"],
                rendering_options["ray_end"],
                rendering_options["depth_resolution"],
                rendering_options["disparity_space_sampling"],
            )

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

        # Coarse Pass
        sample_coordinates_coarse = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)

        out = self.run_model(planes, decoder, sample_coordinates_coarse, sample_directions, rendering_options)

        colors_coarse = out["rgb"]
        densities_coarse = out["sigma"]
        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

        # Fine Pass
        N_importance = rendering_options["depth_resolution_importance"]
        if N_importance > 0:
            _, _, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

            depths_fine = self.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
            sample_coordinates = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

            out = self.run_model(planes, decoder, sample_coordinates, sample_directions, rendering_options)

            colors_fine = out["rgb"]
            densities_fine = out["sigma"]
            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

            all_depths, all_colors, all_densities = self.unify_samples(depths_coarse, colors_coarse, densities_coarse, depths_fine, colors_fine, densities_fine)

            # Aggregate
            rgb_final, depth_final, weights = self.ray_marcher(all_colors, all_densities, all_depths, rendering_options)
        else:
            rgb_final, depth_final, weights = self.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        # Normals at expected termination
        num_samples = 10
        h = w = int(math.sqrt(rgb_final.shape[1]))
        depth_normal = depth_final + torch.linspace(-0.05, 0.0, num_samples).to(depth_final.device)
        depth_normal = depth_normal[..., None]
        coords = (ray_origins.unsqueeze(-2) + depth_normal * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
        all_normal_dirs, out = self.get_normals(planes, decoder, coords, rendering_options)
        normal_coords = rearrange(coords, "b (h w s) c -> s b c h w", b=batch_size, s=num_samples, h=h, w=w)[-1] # only in expected termination point

        normal_density = rearrange(out["sigma"], "b (r s) c -> b r s c", b=batch_size, s=num_samples)
        normal_rgb = rearrange(out["rgb"], "b (r s) c -> b r s c", b=batch_size, s=num_samples)
        all_normal_dirs = rearrange(all_normal_dirs, "b (r s) c -> b r s c", b=batch_size, s=num_samples)
        normal_dirs = self.accumulate_normals(all_normal_dirs, normal_density, normal_rgb, depth_normal, rendering_options)

        return rgb_final, depth_final, normal_dirs, weights.sum(2), normal_coords

    def get_normals(self, planes, decoder, coords, rendering_options):
        coords.requires_grad_(True)
        out_normals = self.run_model(planes, decoder, coords, None, rendering_options)
        densities_normals = F.softplus(out_normals["sigma"] - 5.0)
        normal_dirs =  - grad(outputs=densities_normals.sum(), inputs=coords, retain_graph=True)[0]
        normal_dirs = normal_dirs / (torch.norm(normal_dirs, dim=-1, keepdim=True) + 1e-8)
        return normal_dirs.detach(), out_normals

    def accumulate_normals(self, normal_dirs, normal_density, normal_rgb, normal_depth, rendering_options):
        _, _, weights_normal = self.ray_marcher(normal_rgb, normal_density, normal_depth, rendering_options)
        normal_dirs = (normal_dirs[:, :, :-1] + normal_dirs[:, :, 1:])
        normal_dirs = normal_dirs / (torch.norm(normal_dirs, dim=-1, keepdim=True) + 1e-8)
        normal_dirs = torch.sum(weights_normal * normal_dirs, -2) / (weights_normal.sum(2) + 1e-8)
        normal_dirs = normal_dirs / (torch.norm(normal_dirs, dim=-1, keepdim=True) + 1e-8)
        return normal_dirs.detach()

    def run_model(self, planes, decoder, sample_coordinates, sample_directions, options):
        sampled_features = sample_from_planes(self.plane_axes, planes, sample_coordinates, padding_mode="zeros", bbox_min=options["bbox_min"], bbox_max=options["bbox_max"])

        out = decoder(sampled_features, sample_directions)
        if options.get("density_noise", 0) > 0:
            out["sigma"] += torch.randn_like(out["sigma"]) * options["density_noise"]

        bbox_min=options["bbox_min"]
        bbox_max=options["bbox_max"]
        bbox_min = torch.tensor(bbox_min, dtype=sample_coordinates.dtype, device=sample_coordinates.device)
        bbox_max = torch.tensor(bbox_max, dtype=sample_coordinates.dtype, device=sample_coordinates.device)
        normalized_coords = (sample_coordinates - bbox_min)/(bbox_max - bbox_min)
        normalized_coords = normalized_coords * 2.0 - 1.0

        x_sq = torch.square(normalized_coords[:, :, 0:1])
        y_sq = torch.square(normalized_coords[:, :, 1:2])
        z_sq = torch.square(normalized_coords[:, :, 2:3])
        out['rgb'] = torch.where((x_sq + y_sq + z_sq > 1).repeat(1, 1, 3), torch.tensor([0.0], device=out['rgb'].device, dtype=torch.float32), out['rgb'])
        out['sigma'] = torch.where((x_sq + y_sq + z_sq > 1), torch.tensor([-10.0], device=out['sigma'].device, dtype=torch.float32), out['sigma'])

        return out

    def sort_samples(self, all_depths, all_colors, all_densities):
        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))
        return all_depths, all_colors, all_densities

    def unify_samples(self, depths1, colors1, densities1, depths2, colors2, densities2):
        all_depths = torch.cat([depths1, depths2], dim=-2)
        all_colors = torch.cat([colors1, colors2], dim=-2)
        all_densities = torch.cat([densities1, densities2], dim=-2)

        _, indices = torch.sort(all_depths, dim=-2)
        all_depths = torch.gather(all_depths, -2, indices)
        all_colors = torch.gather(all_colors, -2, indices.expand(-1, -1, -1, all_colors.shape[-1]))
        all_densities = torch.gather(all_densities, -2, indices.expand(-1, -1, -1, 1))

        return all_depths, all_colors, all_densities

    def sample_stratified(self, ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False, near_plane=0):
        """
        Return depths of approximately uniformly spaced samples along rays.
        """
        N, M, _ = ray_origins.shape
        if disparity_space_sampling:
            depths_coarse = (
                torch.linspace(0, 1, depth_resolution, device=ray_origins.device)
                .reshape(1, 1, depth_resolution, 1)
                .repeat(N, M, 1, 1)
            )
            depth_delta = 1 / (depth_resolution - 1)
            depths_coarse += torch.rand_like(depths_coarse) * depth_delta
            depths_coarse = 1.0 / (1.0 / ray_start * (1.0 - depths_coarse) + 1.0 / ray_end * depths_coarse)
        else:
            if type(ray_start) == torch.Tensor:
                ray_start = torch.clip(ray_start, near_plane)
                depths_coarse = math_utils.linspace(ray_start, ray_end, depth_resolution).permute(1, 2, 0, 3)
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta[..., None]
            else:
                if (ray_start < 0):
                    ray_start = 0.0
                depths_coarse = (
                    torch.linspace(ray_start, ray_end, depth_resolution, device=ray_origins.device)
                    .reshape(1, 1, depth_resolution, 1)
                    .repeat(N, M, 1, 1)
                )
                depth_delta = (ray_end - ray_start) / (depth_resolution - 1)
                depths_coarse += torch.rand_like(depths_coarse) * depth_delta

        return depths_coarse

    def sample_importance(self, z_vals, weights, N_importance):
        """
        Return depths of importance sampled points along rays. See NeRF importance sampling for more.
        """
        with torch.no_grad():
            batch_size, num_rays, samples_per_ray, _ = z_vals.shape

            z_vals = z_vals.reshape(batch_size * num_rays, samples_per_ray)
            weights = weights.reshape(batch_size * num_rays, -1)  # -1 to account for loss of 1 sample in MipRayMarcher

            # smooth weights
            weights = torch.nn.functional.max_pool1d(weights.unsqueeze(1).float(), 2, 1, padding=1)
            weights = torch.nn.functional.avg_pool1d(weights, 2, 1).squeeze()
            weights = weights + 0.01

            z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])
            importance_z_vals = (
                self.sample_pdf(z_vals_mid, weights[:, 1:-1], N_importance)
                .detach()
                .reshape(batch_size, num_rays, N_importance, 1)
            )
        return importance_z_vals

    def sample_pdf(self, bins, weights, N_importance, det=False, eps=1e-5):
        """
        Sample @N_importance samples from @bins with distribution defined by @weights.
        Inputs:
            bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
            weights: (N_rays, N_samples_)
            N_importance: the number of samples to draw from the distribution
            det: deterministic or not
            eps: a small number to prevent division by zero
        Outputs:
            samples: the sampled samples
        """
        N_rays, N_samples_ = weights.shape
        weights = weights + eps  # prevent division by zero (don't do inplace op!)
        pdf = weights / torch.sum(weights, -1, keepdim=True)  # (N_rays, N_samples_)
        cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
        cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
        # padded to 0~1 inclusive

        if det:
            u = torch.linspace(0, 1, N_importance, device=bins.device)
            u = u.expand(N_rays, N_importance)
        else:
            u = torch.rand(N_rays, N_importance, device=bins.device)
        u = u.contiguous()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp_min(inds - 1, 0)
        above = torch.clamp_max(inds, N_samples_)

        inds_sampled = torch.stack([below, above], -1).view(N_rays, 2 * N_importance)
        cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
        bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

        denom = cdf_g[..., 1] - cdf_g[..., 0]
        denom[denom < eps] = 1  # denom equals 0 means a bin has weight 0, in which case it will not be sampled
        # anyway, therefore any value for it is fine (set to 1 here)

        samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
        return samples
