import imgui
from gui_utils import imgui_utils

# ----------------------------------------------------------------------------


class GANSpaceWidget:
    def __init__(self, viz):
        self.viz = viz
        self.enables = []
        self.first_pca = -1
        self.last_pca = -1
        self.layers_pca = []
        self.offset_pca = 0
        self.enables = [False] * 16

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz
        if show:
            num_ws = viz.result.get("num_ws", 0)
            num_enables = 16
            pos2 = imgui.get_content_region_max()[0] - 1 - viz.button_w
            pos1 = pos2 - imgui.get_text_line_height() - viz.spacing
            pos0 = viz.label_w + viz.font_size * 12

            imgui.text("PCA components")
            imgui.same_line(viz.label_w * 1.5)
            imgui.text("Weight")
            imgui.same_line(viz.label_w * 3.18)
            imgui.text("Layers")
            with imgui_utils.item_width(viz.font_size * 5):
                changed, (first_pca, last_pca) = imgui.input_int2(
                    "##frac", self.first_pca, self.last_pca, flags=imgui.INPUT_TEXT_ENTER_RETURNS_TRUE
                )
                if changed:
                    self.first_pca = first_pca
                    self.last_pca = last_pca
            imgui.same_line(viz.label_w * 1.5)
            with imgui_utils.item_width(viz.font_size * 6):
                changed, offset_pca = imgui.slider_float("##weight", self.offset_pca, -10, 10, format="%.2f")
                if changed:
                    self.offset_pca = offset_pca

            imgui.push_style_var(imgui.STYLE_FRAME_PADDING, [0, 0])
            for idx in range(num_enables):
                imgui.same_line(round(pos0 + (pos1 - pos0) * (idx / (num_enables - 1))))
                if idx == 0:
                    imgui.set_cursor_pos_y(imgui.get_cursor_pos_y() + 3)
                with imgui_utils.grayed_out(num_ws == 0):
                    _clicked, self.enables[idx] = imgui.checkbox(f"##{idx}", self.enables[idx])
                if imgui.is_item_hovered():
                    imgui.set_tooltip(f"{idx}")
            imgui.pop_style_var(1)
            for idx in range(num_enables):
                imgui.same_line(round(pos0 + (pos1 - pos0) * (idx / (num_enables - 1))))
            imgui.new_line()

            # Pass parameters to renderer
            self.viz.args.first_pca = self.first_pca
            self.viz.args.last_pca = self.last_pca
            self.viz.args.layers_pca = self.enables
            self.viz.args.offset_pca = self.offset_pca
