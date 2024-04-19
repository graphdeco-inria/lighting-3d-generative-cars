import click
import json
import os
import numpy as np
import torch
import cv2 as cv
from torchvision.ops import box_iou
from torch.utils.data import DataLoader
import torch.nn.functional as F
from PIL import Image
from libs.model.egonet import EgoNet
from libs.arguments.parse import read_yaml_file
from grounded_sam import GroundedSAM
from copy import deepcopy

from cam2world import (
    generate_eg3d_cam2world,
    create_canonical_box,
    interp_coef,
    interp_dict,
)
from dataset import PathDataset


def flip_yaw(pose_matrix):
    flipped = deepcopy(pose_matrix)
    flipped[0, 1] *= -1
    flipped[0, 2] *= -1
    flipped[1, 0] *= -1
    flipped[2, 0] *= -1
    flipped[0, 3] *= -1
    return flipped

def collate_fn(batch):
    return [item for item in batch if item is not None]


def create_points_3d(l, h, w):
    corners_3d = create_canonical_box(l, h, w).T
    pidx, cidx = interp_dict["bbox12"]
    parents, children = corners_3d[:, pidx - 1], corners_3d[:, cidx - 1]
    lines = children - parents
    new_joints = [(parents + interp_coef[i] * lines) for i in range(len(interp_coef))]
    points_3d = np.hstack([corners_3d, np.hstack(new_joints)]).T
    return points_3d


def calc_square_pad(target_resolution, height, width):
    pad = (np.maximum(0, target_resolution - height), np.maximum(0, target_resolution - width))
    pad = (int(np.ceil(pad[1] / 2)), int(np.floor(pad[1] / 2)), int(np.ceil(pad[0] / 2)), int(np.floor(pad[0] / 2)))
    return pad


def prepare_for_estimation(image):
    _, h, w = image.shape
    pad = calc_square_pad(w, h, w)
    return F.interpolate(F.pad(image, pad)[None, ...], size=[256, 256]).cuda()


class DatasetAnotator:
    def __init__(self, cfg):
        self.cfg = read_yaml_file(cfg)

        # Intrinsics matrix
        focal_length = 1.5166258727104762  # fov=50 degrees
        norm_intrinsics = np.array([focal_length, 0.0, 0.5, 0.0, focal_length, 0.5, 0.0, 0.0, 1.0]).reshape(3, 3)
        intrinsics = norm_intrinsics * 255  # EgoNet was trained on 256x256
        intrinsics[2, 2] = 1.0
        self.intrinsics = intrinsics
        self.norm_intrinsics = norm_intrinsics.flatten().tolist()

        # cudnn related setting
        torch.backends.cudnn.benchmark = self.cfg["cudnn"]["benchmark"]
        torch.backends.cudnn.deterministic = self.cfg["cudnn"]["deterministic"]
        torch.backends.cudnn.enabled = self.cfg["cudnn"]["enabled"]

        # Initialize Ego-Net and load the pre-trained checkpoint
        print("Loading EgoNet...")
        self.model_3d = EgoNet(self.cfg, pre_trained=True).eval().cuda()

        # Canonical bounding box
        self.points_3d = create_points_3d(3 / 5, 1.5 / 5, 1.5 / 5)

        print("Loading TextSAM...")
        self.text_sam = GroundedSAM()

    def create_dataset(
        self, data, dest, max_images, batch_size, show_detection, iou_th, resolution, mirror, dest_seg=None
    ):
        dataloader = DataLoader(PathDataset(data), batch_size, num_workers=16, collate_fn=collate_fn)
        dataset_json = {"labels": []}
        count = 0
        total = 0
        done = False
        for images_raw in dataloader:
            if (max_images is not None) and (count >= max_images):
                done = True
            if done:
                break
            if resolution is not None:
                images_raw = [img.cuda() for img in images_raw if (img.shape[1] >= resolution and img.shape[2] >= resolution)]
            if len(images_raw) == 0:
                continue
            images = list(map(prepare_for_estimation, images_raw))
            detected_boxes = [[10, 25, 245, 225] for _ in range(len(images))]
            records = self.detection_3d(torch.cat(images), detected_boxes)

            for idx, record in enumerate(records):
                if (max_images is not None) and (count >= max_images):
                    done = True
                    break

                points_2d = record["kpts_2d_pred"][1:]  # ignore center point
                detected_box = detected_boxes[idx]
                iou = self.calc_iou(points_2d, detected_box)
                if iou < iou_th:
                    continue
                cam2world = generate_eg3d_cam2world(points_2d, self.points_3d, self.intrinsics)

                # Save results
                folder = str(count // 1000).zfill(5)
                img_name = f"img{str(count).zfill(8)}.png"
                os.makedirs(os.path.join(dest, folder), exist_ok=True)
                name = os.path.join(folder, img_name)
                filepath = os.path.join(dest, name)

                filepath_mask = None
                if dest_seg != None:
                    os.makedirs(os.path.join(dest_seg, folder), exist_ok=True)
                    filepath_mask = os.path.join(dest_seg, name)

                stat = self.save_image(images_raw[idx], records[idx], filepath, show_detection, resolution, mirror=True, mask_path=filepath_mask)

                if stat == 0:
                    label = cam2world.flatten().tolist() + self.norm_intrinsics
                    dataset_json["labels"].append([name, label])
                    
                    if mirror:
                        pose = np.array(label[:16]).reshape(4,4)
                        flipped_pose = flip_yaw(pose)
                        mirror_label = flipped_pose.flatten().tolist() + self.norm_intrinsics
                        mirror_name =  os.path.join(folder, f"img{str(count).zfill(8)}_mirror.png")
                        dataset_json["labels"].append([mirror_name, mirror_label])


                count += 1
            total += len(detected_boxes)
            print(f"Selecting {count} images from {total}")
        self.save_annotations(dest, dataset_json)

    def calc_iou(self, points_2d, detected_box):
        x_min = np.min(points_2d[:, 0]).astype(int)
        x_max = np.max(points_2d[:, 0]).astype(int)
        y_min = np.min(points_2d[:, 1]).astype(int)
        y_max = np.max(points_2d[:, 1]).astype(int)

        box_keypoints = torch.Tensor([[x_min, y_min, x_max, y_max]])
        box_detection = torch.Tensor(detected_box)[None, :]
        iou = box_iou(box_keypoints, box_detection).item()
        return iou

    def detection_3d(self, paths, detected_boxes):
        annots = {"image": [], "boxes": []}
        for i, detected_box in enumerate(detected_boxes):
            annots["image"].append(paths[i])
            annots["boxes"].append([detected_box])
        records = self.model_3d(annots)
        return records

    @staticmethod
    def save_annotations(dest, dataset_json):
        with open(os.path.join(dest, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=2)

    def save_image(self, img, record, filepath, show_detection, resolution, mirror=False, mask_path=None):

        img = (255 * img.cpu().numpy()).transpose(1, 2, 0).astype(np.uint8)
        img = np.ascontiguousarray(img)
        h, w = img.shape[0], img.shape[1]
        offset = calc_square_pad(w, h, w)[2] * 256 / w

        if show_detection:
            points = (w * (record["kpts_2d_pred"] - np.array([0.0, offset])) / 256).astype(int)
            # Draw lines
            lines = [
                (1, 2), (1, 3), (2, 4), (3, 4), # front 
                (5, 6), (5, 7), (8, 6), (8, 7), # back
                (3, 7), (4, 8), (1, 5), (2, 6)  # sides
            ]
            for line in lines:
                img = cv.line(img, points[line[0]], points[line[1]], thickness=2, color=(0, 255, 0))
            # Draw points
            for point in points:
                img = cv.circle(img, center=point, radius=int(2 * w / 256), thickness=-1, color=(255, 0, 0))

        p = calc_square_pad(img.shape[1], *img.shape[:2])
        img = np.pad(img, (p[::-1][:2], p[::-1][2:], (0,0)), mode='constant')

        text_promt = ['only the biggest car']
        masks, scores = self.text_sam.segment(img, text_promt)
        masks = masks[scores > 0.9]
        mask = masks[np.argmax(np.sum(masks, (1, 2)))]
        _, labels, stats, _ = cv.connectedComponentsWithStats(mask.astype(np.uint8))

        label = stats[1:, cv.CC_STAT_AREA].argmax()+1
        mask[labels != label] = 0
        
        img_final = Image.fromarray((img.astype(np.float32) * mask.astype(np.float32)[..., None]).astype(np.uint8)).resize([resolution, resolution], resample=Image.Resampling.LANCZOS)
        mask_res = Image.fromarray((255 * mask).astype(np.uint8)).resize([resolution, resolution], resample=Image.Resampling.LANCZOS)

        img_final.save(filepath)
        if mask_path != None:
            mask_res.save(mask_path)
        if mirror:
            img_name, ext = os.path.basename(filepath).split('.')
            filepath_mirror = os.path.join(os.path.dirname(filepath), f"{img_name}_mirror.{ext}")
            img_final.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(filepath_mirror)

            if mask_path != None:
                img_name, ext = os.path.basename(mask_path).split('.')
                mask_filepath_mirror = os.path.join(os.path.dirname(mask_path), f"{img_name}_mirror.{ext}")
                mask_res.transpose(Image.Transpose.FLIP_LEFT_RIGHT).save(mask_filepath_mirror)
        return 0


@click.command()
@click.option("--cfg", type=str, help="Configuration .yaml file", default="./config.yml")
@click.option("--data", type=str, help="Dataset folder path", required=True)
@click.option("--dest", type=str, help="Destination folder", required=True)
@click.option("--dest-seg", type=str, help="Destination folder for masks", default=None)
@click.option("--max-images", type=int, default=None)
@click.option("--batch-size", type=int, default=32)
@click.option("--show-detection", type=bool, default=False)
@click.option("--iou-th", type=float, default=0.18)
@click.option("--resolution", type=int, default=256)
@click.option("--mirror", type=bool, default=True)
def main(cfg, data, dest, dest_seg, max_images, batch_size, show_detection, iou_th, resolution, mirror):
    annotator = DatasetAnotator(cfg)
    annotator.create_dataset(data, dest, max_images, batch_size, show_detection, iou_th, resolution, mirror, dest_seg)


if __name__ == "__main__":
    main()
