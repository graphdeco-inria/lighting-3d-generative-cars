import imgui
from gui_utils import imgui_utils
import math
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2 as cv
import torch
from einops import rearrange

# ----------------------------------------------------------------------------

def load_envmap(fname, device="cuda"):
    envmap = cv.cvtColor(cv.imread(fname, cv.IMREAD_ANYCOLOR | cv.IMREAD_ANYDEPTH), cv.COLOR_BGR2RGB)
    envmap = rearrange(torch.from_numpy(envmap), "h w c -> 1 c h w")
    return envmap.to(device)


def load_envmaps(envmaps_dir, device="cuda"):
    img_name = os.path.basename(envmaps_dir)
    specular_envmap = load_envmap(os.path.join((envmaps_dir), f"{img_name}_specular.hdr"), device)
    diffuse_envmap = load_envmap(os.path.join((envmaps_dir), f"{img_name}_diffuse.hdr"), device)
    return specular_envmap, diffuse_envmap


class EnvmapSelectWidget:
    def __init__(self, viz):
        self.viz = viz
        self.envmaps_dir = "./samples/envmap_sample"
        self.ground_kwargs = {
            "scale": 0.55,
            "offset": -0.15,
            "size_x":0.5,
            "size_y":0.75,
            "radius":0.3,
            "use_ground_shadow": True,
        }
        self.search_dirs    = []


        self.offset_azimuth = 0
        self.offset_elevation = 0
        self.specular_envmap, self.diffuse_envmap = load_envmaps(self.envmaps_dir)
    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            imgui.text("Envmap Path")
            imgui.same_line(viz.label_w)
            changed, envmap_path = imgui_utils.input_text(
                "##envmap_path",
                self.envmaps_dir,
                1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text="<ENVMAP_PATH>.png",
            )
            if changed:
                self.envmaps_dir = envmap_path
                self.specular_envmap, self.diffuse_envmap = load_envmaps(self.envmaps_dir)

            imgui.same_line()
            if imgui_utils.button('Browse...', enabled=len(self.search_dirs) > 0, width=-1):
                imgui.open_popup('browse_envmaps_popup')
                self.browse_cache.clear()
                self.browse_refocus = True
            
            imgui.text("Azimuth")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 20):
                changed, offset = imgui.slider_float("##weight1", self.offset_azimuth, 0, 360, format="%.2f")
                if changed:
                    self.offset_azimuth = offset

            # imgui.same_line(2 * viz.label_w)
            imgui.text("Elevation")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 20):
                changed, offset = imgui.slider_float("##weight2", self.offset_elevation, 0, 180, format="%.2f")
                if changed:
                    self.offset_elevation = offset
            
            imgui.text("Ground scale")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, offset = imgui.slider_float("##weight3", self.ground_kwargs["scale"], 0.1, 5.0, format="%.2f")
                if changed:
                    self.ground_kwargs["scale"] = offset

            imgui.text("Ground offset")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, offset = imgui.slider_float("##weight4", self.ground_kwargs["offset"], -1.0, 1.0, format="%.2f")
                if changed:
                    self.ground_kwargs["offset"] = offset

            imgui.text("Shadow size X")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, offset = imgui.slider_float("##weight5", self.ground_kwargs["size_x"], 0.0, 1.0, format="%.2f")
                if changed:
                    self.ground_kwargs["size_x"] = offset

            imgui.text("Shadow size Y")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, offset = imgui.slider_float("##weight6", self.ground_kwargs["size_y"], 0.0, 1.0, format="%.2f")
                if changed:
                    self.ground_kwargs["size_y"] = offset

            imgui.text("Shadow radius")
            imgui.same_line(viz.label_w)
            with imgui_utils.item_width(viz.font_size * 10):
                changed, offset = imgui.slider_float("##weight7", self.ground_kwargs["radius"], 0.0, 1.0, format="%.2f")
                if changed:
                    self.ground_kwargs["radius"] = offset

            if imgui.begin_popup('browse_envmaps_popup'):
                def recurse(parents):
                    key = tuple(parents)
                    items = self.browse_cache.get(key, None)
                    if items is None:
                        items = self.list_runs_and_pkls(parents)
                        self.browse_cache[key] = items
                    for item in items:
                        if item.type == 'run' and imgui.begin_menu(item.name):
                            recurse([item.path])
                            imgui.end_menu()
                        if item.type == 'exr':
                            clicked, _state = imgui.menu_item(item.name)
                            if clicked:
                                self.load(item.path, ignore_errors=True)
                    if len(items) == 0:
                        with imgui_utils.grayed_out():
                            imgui.menu_item('No results found')
                recurse(self.search_dirs)
                if self.browse_refocus:
                    imgui.set_scroll_here()
                    viz.skip_frame() # Focus will change on next frame.
                    self.browse_refocus = False
                imgui.end_popup()


            # Pass parameters to renderer
            self.viz.args.specular_envmap = self.specular_envmap
            self.viz.args.diffuse_envmap = self.diffuse_envmap
            self.viz.args.offset_azimuth = (self.offset_azimuth / 360.0)
            self.viz.args.offset_elevation = (self.offset_elevation / 180.0)
            self.viz.args.ground_kwargs = self.ground_kwargs
