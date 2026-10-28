"""Window 2: server config, display processing, stream source, advanced
camera controls, and the save actions."""

import threading

from imgui_bundle import imgui

import capture_io
import raw_convert
import server_api


def render(state):
    imgui.set_next_window_size((380, 680), imgui.Cond_.first_use_ever)
    imgui.set_next_window_pos((900, 50), imgui.Cond_.first_use_ever)

    imgui.begin("Stream Settings")
    imgui.text("Camera Settings")
    imgui.separator()
    imgui.spacing()

    changed_res, new_res_idx = imgui.combo("Resolution", state.current_res_idx, state.resolutions)
    if changed_res:
        state.current_res_idx = new_res_idx
        w, h = map(int, state.resolutions[new_res_idx].split("x"))
        fmt_str = state.formats[state.current_fmt_idx]
        threading.Thread(target=server_api.change_config_thread, args=(state, w, h, fmt_str), daemon=True).start()

    changed_fmt, new_fmt_idx = imgui.combo("Format", state.current_fmt_idx, state.formats)
    if changed_fmt:
        state.current_fmt_idx = new_fmt_idx
        w, h = map(int, state.resolutions[state.current_res_idx].split("x"))
        fmt_str = state.formats[new_fmt_idx]
        threading.Thread(target=server_api.change_config_thread, args=(state, w, h, fmt_str), daemon=True).start()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.text("Stream Source")
    imgui.text_disabled("Raw/DNG are polled one frame at a time, not real streams")
    imgui.text_disabled("(DNG is slow: full raw file download + demosaic per frame)")

    for mode, label in (("processed", "Processed"), ("raw", "Raw"), ("dng", "DNG")):
        checked = state.stream_mode == mode
        changed, new_val = imgui.checkbox(label, checked)
        if changed and new_val:
            state.set_stream_mode(mode)
        if mode != "dng":
            imgui.same_line()

    if state.stream_mode == "raw" and state.raw_reader is not None:
        imgui.text(f"Raw poll rate: {state.raw_reader.fps:.1f} FPS")
    elif state.stream_mode == "dng" and state.dng_reader is not None:
        imgui.text(f"DNG poll rate: {state.dng_reader.fps:.1f} FPS")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.text("Display Processing")

    changed_color, new_color_idx = imgui.combo("Color Order", state.color_order_idx, state.color_order_options)
    if changed_color:
        state.color_order_idx = new_color_idx
        state.settings_dirty = True

    imgui.text(f"Rotation: {state.rotation_deg}°")
    if imgui.button("◀ Rotate Left"):
        state.rotation_deg = (state.rotation_deg - 90) % 360
        state.settings_dirty = True
    imgui.same_line()
    if imgui.button("Rotate Right ▶"):
        state.rotation_deg = (state.rotation_deg + 90) % 360
        state.settings_dirty = True
    imgui.same_line()
    if imgui.button("Reset"):
        state.rotation_deg = 0
        state.settings_dirty = True

    changed_bp, new_bp_idx = imgui.combo(
        "Bayer Pattern (raw only)",
        raw_convert.BAYER_PATTERNS.index(state.bayer_pattern),
        raw_convert.BAYER_PATTERNS,
    )
    if changed_bp:
        state.bayer_pattern = raw_convert.BAYER_PATTERNS[new_bp_idx]
        state.settings_dirty = True

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    if imgui.collapsing_header("Advanced Camera Controls"):
        if imgui.button("Refresh Controls"):
            threading.Thread(target=server_api.fetch_camera_controls, args=(state,), daemon=True).start()

        imgui.spacing()

        if state.camera_controls:
            imgui.begin_child("ControlsScroll", (0, 200))
            for name in sorted(state.camera_controls.keys()):
                ctrl = state.camera_controls[name]
                curr = ctrl.get("current")
                min_v = ctrl.get("min")
                max_v = ctrl.get("max")

                if isinstance(curr, bool):
                    changed, new_val = imgui.checkbox(name, curr)
                    if changed:
                        threading.Thread(
                            target=server_api.set_camera_control_thread, args=(state, name, new_val), daemon=True
                        ).start()
                elif isinstance(curr, (int, float)):
                    is_int = isinstance(curr, int)
                    has_simple_limits = min_v is not None and max_v is not None and (
                        (is_int and (max_v - min_v) <= 100) or (not is_int and (max_v - min_v) <= 20.0)
                    )

                    if has_simple_limits:
                        if is_int:
                            changed, ctrl["current"] = imgui.slider_int(name, curr, min_v, max_v)
                        else:
                            changed, ctrl["current"] = imgui.slider_float(name, curr, min_v, max_v)

                        if imgui.is_item_deactivated_after_edit():
                            threading.Thread(
                                target=server_api.set_camera_control_thread,
                                args=(state, name, ctrl["current"]),
                                daemon=True,
                            ).start()
                    else:
                        if is_int:
                            changed, ctrl["current"] = imgui.input_int(name, curr)
                        else:
                            changed, ctrl["current"] = imgui.input_float(name, curr)

                        if imgui.is_item_deactivated_after_edit():
                            threading.Thread(
                                target=server_api.set_camera_control_thread,
                                args=(state, name, ctrl["current"]),
                                daemon=True,
                            ).start()
                else:
                    imgui.text(f"{name}: {curr}")
            imgui.end_child()
        else:
            imgui.text("No controls available (click Refresh)")

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.text("Actions")

    imgui.set_next_item_width(-1)
    _, state.save_name_buf = imgui.input_text("##SaveName", state.save_name_buf)
    imgui.text_disabled("Save name (blank = auto timestamp); extension is added per action")

    if imgui.button("Capture Raw", (-1, 0)):
        threading.Thread(target=capture_io.capture_raw_thread, args=(state,), daemon=True).start()

    if imgui.button("Capture DNG", (-1, 0)):
        threading.Thread(target=capture_io.capture_dng_thread, args=(state,), daemon=True).start()

    if imgui.button("Save Current", (-1, 0)):
        threading.Thread(target=capture_io.save_current_thread, args=(state,), daemon=True).start()

    imgui.spacing()
    imgui.separator()
    imgui.spacing()

    imgui.text("Status: ")
    imgui.same_line()
    processed_connected = state.reader.get_status() if state.reader is not None else False
    if processed_connected:
        imgui.text_colored((0.2, 0.8, 0.2, 1.0), "Connected")
    else:
        imgui.text_colored((0.9, 0.2, 0.2, 1.0), "Disconnected")

    imgui.text("Activity: ")
    imgui.same_line()
    imgui.text_colored(state.status_color, state.status_message)

    imgui.spacing()
    imgui.separator()
    imgui.text(f"UI Frame Rate: {imgui.get_io().framerate:.1f} FPS")
    imgui.text(f"Processed Stream FPS: {state.reader.fps if state.reader else 0.0:.1f}")

    imgui.end()
