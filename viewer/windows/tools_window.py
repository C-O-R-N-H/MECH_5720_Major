"""Window 3: the extensible analysis tools window. Just a tab bar over the
tab registry in tabs/ — new tabs are added there, not here."""

from imgui_bundle import imgui

from tabs import TABS


def render(state):
    imgui.set_next_window_size((700, 500), imgui.Cond_.first_use_ever)
    imgui.set_next_window_pos((50, 670), imgui.Cond_.first_use_ever)

    imgui.begin("Analysis Tools")

    if imgui.begin_tab_bar("AnalysisTabs"):
        for tab in TABS:
            selected, _ = imgui.begin_tab_item(tab.name)
            if selected:
                tab.render(state)
                imgui.end_tab_item()
        imgui.end_tab_bar()

    imgui.end()
