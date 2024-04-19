import imgui
from gui_utils import imgui_utils
import math
import cv2 as cv
import torch
from einops import rearrange
import os

# ----------------------------------------------------------------------------

class ShadingWidget:
    def __init__(self, viz):
        self.viz = viz
        self.shading_kwargs = {
            "metallic": 0.7,
            "roughness": 0.2,
            "use_clear_coat": False,
            "filter_normals": True,
            "filter_mask": True,
            "scale_specular": 0.25,
            "scale_env": 1.0,
            "hue_shift": 0.0,
        }

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text("Metallic")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, metallic = imgui.slider_float("##weight1", self.shading_kwargs["metallic"], 0, 1.0, format="%.2f")
                if changed:
                    self.shading_kwargs["metallic"] = metallic

            imgui.text("Scale Specular")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, scale_specular = imgui.slider_float("##weight3", self.shading_kwargs["scale_specular"], 0, 1.0, format="%.2f")
                if changed:
                    self.shading_kwargs["scale_specular"] = scale_specular

            imgui.text("Brightness")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, offset = imgui.slider_float("##weight4", self.shading_kwargs["scale_env"], 0, 5, format="%.2f")
                if changed:
                    self.shading_kwargs["scale_env"] = offset

            imgui.text("Hue")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, offset = imgui.slider_float("##weight5", self.shading_kwargs["hue_shift"], -0.5, 0.5, format="%.2f")
                if changed:
                    self.shading_kwargs["hue_shift"] = offset

            imgui.text('Filter Normals')
            imgui.same_line(viz.label_w + viz.spacing * 4)
            _clicked, filter_normals = imgui.checkbox('##normals', self.shading_kwargs["filter_normals"])
            self.shading_kwargs["filter_normals"] = filter_normals
            
            imgui.text('Filter Mask')
            imgui.same_line(viz.label_w + viz.spacing * 4)
            _clicked, filter_normals = imgui.checkbox('##mask', self.shading_kwargs["filter_mask"])
            self.shading_kwargs["filter_mask"] = filter_normals

            imgui.text('Clear Coat')
            imgui.same_line(viz.label_w + viz.spacing * 4)
            _clicked, use_clear_coat = imgui.checkbox('##cc', self.shading_kwargs["use_clear_coat"])
            self.shading_kwargs["use_clear_coat"] = use_clear_coat
            
            self.viz.args.shading_kwargs = self.shading_kwargs
