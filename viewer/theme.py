"""Visual theme for the imgui UI."""

from imgui_bundle import imgui

# Scales every piece of UI text app-wide (imgui doesn't re-rasterize the
# font, so this just stretches glyphs -- fine up to ~1.5-2.0 before it looks
# blurry). Bump this here if the UI is too small.
TEXT_SCALE = 1.3


def apply_theme():
    """Sets a beautiful, modern slate-purple dark theme on ImGui."""
    style = imgui.get_style()

    # imgui 1.92's dynamic font system dropped io.font_global_scale in
    # favor of this per-style multiplier.
    style.font_scale_main = TEXT_SCALE

    style.window_rounding = 10.0
    style.frame_rounding = 6.0
    style.grab_rounding = 6.0
    style.scrollbar_rounding = 6.0
    style.window_border_size = 1.0

    theme_colors = {
        imgui.Col_.text: (0.90, 0.90, 0.95, 1.0),
        imgui.Col_.text_disabled: (0.50, 0.50, 0.55, 1.0),
        imgui.Col_.window_bg: (0.08, 0.08, 0.11, 0.95),
        imgui.Col_.child_bg: (0.08, 0.08, 0.11, 1.0),
        imgui.Col_.popup_bg: (0.12, 0.12, 0.16, 0.98),
        imgui.Col_.border: (0.25, 0.25, 0.35, 0.6),
        imgui.Col_.border_shadow: (0.00, 0.00, 0.00, 0.0),
        imgui.Col_.frame_bg: (0.14, 0.14, 0.20, 1.0),
        imgui.Col_.frame_bg_hovered: (0.20, 0.20, 0.28, 1.0),
        imgui.Col_.frame_bg_active: (0.28, 0.28, 0.38, 1.0),
        imgui.Col_.title_bg: (0.12, 0.12, 0.16, 1.0),
        imgui.Col_.title_bg_active: (0.18, 0.18, 0.24, 1.0),
        imgui.Col_.title_bg_collapsed: (0.08, 0.08, 0.11, 0.50),
        imgui.Col_.menu_bar_bg: (0.14, 0.14, 0.18, 1.0),
        imgui.Col_.scrollbar_bg: (0.10, 0.10, 0.14, 1.0),
        imgui.Col_.scrollbar_grab: (0.25, 0.25, 0.35, 1.0),
        imgui.Col_.scrollbar_grab_hovered: (0.35, 0.35, 0.45, 1.0),
        imgui.Col_.scrollbar_grab_active: (0.45, 0.45, 0.55, 1.0),
        imgui.Col_.check_mark: (0.50, 0.55, 0.90, 1.0),
        imgui.Col_.slider_grab: (0.40, 0.45, 0.80, 1.0),
        imgui.Col_.slider_grab_active: (0.50, 0.55, 0.90, 1.0),
        imgui.Col_.button: (0.20, 0.25, 0.45, 1.0),
        imgui.Col_.button_hovered: (0.28, 0.35, 0.60, 1.0),
        imgui.Col_.button_active: (0.35, 0.45, 0.75, 1.0),
        imgui.Col_.header: (0.20, 0.25, 0.45, 1.0),
        imgui.Col_.header_hovered: (0.28, 0.35, 0.60, 1.0),
        imgui.Col_.header_active: (0.35, 0.45, 0.75, 1.0),
    }

    for col_enum, col_val in theme_colors.items():
        style.set_color_(col_enum, imgui.ImVec4(*col_val))
