"""Window 1: the live video feed, showing whichever of the processed MJPEG
stream or the polled raw stream is currently selected."""

from imgui_bundle import imgui


def render(state):
    imgui.set_next_window_size((800, 600), imgui.Cond_.first_use_ever)
    imgui.set_next_window_pos((50, 50), imgui.Cond_.first_use_ever)

    imgui.begin("Pi Camera Feed")

    if state.stream_mode == "raw":
        reader = state.raw_reader
        has_texture = state.has_raw_texture
        tex_id = state.raw_texture_id
        tex_w, tex_h = state.raw_tex_size
    else:
        reader = state.reader
        has_texture = state.has_processed_texture
        tex_id = state.processed_texture_id
        tex_w, tex_h = state.processed_tex_size

    connected = reader.get_status() if reader is not None else False

    if has_texture and connected:
        avail = imgui.get_content_region_avail()
        scale = min(avail.x / tex_w, avail.y / tex_h)
        draw_w = tex_w * scale
        draw_h = tex_h * scale

        cursor_pos = imgui.get_cursor_pos()
        cursor_x = cursor_pos.x + (avail.x - draw_w) / 2
        cursor_y = cursor_pos.y + (avail.y - draw_h) / 2
        imgui.set_cursor_pos((cursor_x, cursor_y))
        imgui.image(imgui.ImTextureRef(tex_id), imgui.ImVec2(draw_w, draw_h))
    else:
        avail = imgui.get_content_region_avail()
        imgui.set_cursor_pos((avail.x / 2 - 120, avail.y / 2 - 10))
        if connected:
            imgui.text("Decoding stream...")
        elif state.stream_mode == "raw":
            imgui.text("Waiting for raw frame (polling /capture_raw)...")
        else:
            imgui.text("Connecting to camera stream...")

    imgui.end()
