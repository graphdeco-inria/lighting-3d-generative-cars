import numpy as np
from segment_anything import sam_model_registry, SamPredictor
from groundingdino.util.inference import Model as GroundedDINO
from groundingdino.config import GroundingDINO_SwinT_OGC
import urllib.request
import os

urls = {"dino_ckpt": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "sam_ckpt" : "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
        }

def download_from_url(url):
    filename = os.path.basename(url)
    filepath = os.path.join("checkpoints", filename)
    print(f"Downloading {filename}")
    urllib.request.urlretrieve(url, filepath)

class GroundedSAM:
    def __init__(self, device="cuda") -> None:

        GROUNDING_DINO_CONFIG_PATH = GroundingDINO_SwinT_OGC.__file__
        GROUNDING_DINO_CHECKPOINT_PATH = "./checkpoints/groundingdino_swint_ogc.pth"

        if not os.path.exists(GROUNDING_DINO_CHECKPOINT_PATH):
            download_from_url(urls["dino_ckpt"])

        self.grounding_dino_model = GroundedDINO(GROUNDING_DINO_CONFIG_PATH, GROUNDING_DINO_CHECKPOINT_PATH, device)

        self.box_thr = 0.35
        self.text_thr = 0.25

        SAM_CHECKPOINT = "./checkpoints/sam_vit_h_4b8939.pth"
        model_type = "vit_h"

        if not os.path.exists(SAM_CHECKPOINT):
            download_from_url(urls["sam_ckpt"])

        sam = sam_model_registry[model_type](checkpoint=SAM_CHECKPOINT)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)

    def segment(self, image: np.array, text_prompt: str):
        detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=text_prompt,
            box_threshold=self.box_thr,
            text_threshold=self.text_thr
        )
        self.sam_predictor.set_image(image)
        result_masks = []
        for box in detections.xyxy:
            masks, scores, _ = self.sam_predictor.predict(
                box=box,
                multimask_output=False,
            )
            index = np.argmax(scores)
            result_masks.append(masks[index])
        return np.array(result_masks), scores
