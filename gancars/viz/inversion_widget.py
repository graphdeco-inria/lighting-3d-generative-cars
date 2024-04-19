import imgui
from gui_utils import imgui_utils

# ----------------------------------------------------------------------------


class InversionWidget:
    def __init__(self, viz):
        self.viz = viz
        self.w_path = ""

    @imgui_utils.scoped_by_object_id
    def __call__(self, show=True):
        viz = self.viz

        if show:
            imgui.text("W+Path")
            imgui.same_line(viz.label_w)
            changed, w_path = imgui_utils.input_text(
                "##inversion",
                self.w_path,
                1024,
                flags=(imgui.INPUT_TEXT_AUTO_SELECT_ALL | imgui.INPUT_TEXT_ENTER_RETURNS_TRUE),
                width=(-1 - viz.button_w * 2 - viz.spacing * 2),
                help_text="<W_PATH>.png",
            )
            if changed:
                self.w_path = w_path

        self.viz.args.w_path = self.w_path