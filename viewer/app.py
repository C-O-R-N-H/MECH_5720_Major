"""
App: GLFW/imgui/GL setup and the main render loop. Owns the AppState and
the three background readers, and each frame just asks the three windows to
draw themselves from that shared state. See CLAUDE.md for the architecture.
"""

import os

# GLFW creates an X11/GLX context; without this, PyOpenGL auto-detects the
# Wayland session and defaults to its EGL backend, which can't see the GLX
# current-context state and raises "no valid context".
os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import OpenGL.GL as gl
from imgui_bundle import imgui, implot
from imgui_bundle.python_backends.glfw_backend import GlfwRenderer
import glfw

import raw_convert
import server_api
import theme
from state import AppState
from stream import DngStreamPoller, FrameReader, RawStreamPoller
from windows import settings_window, tools_window, video_window


class App:
    def __init__(self, host, port):
        self.base_url = f"http://{host}:{port}"
        self.state = AppState(self.base_url)
        self.window = None
        self.impl = None

    def _load_initial_config(self):
        state = self.state
        try:
            cfg = server_api.get_config(self.base_url)
            print("Connected to server. Current config:", cfg)
            server_api.fetch_camera_controls(state)
        except Exception:
            print("Warning: Could not connect to Pi camera server on start. Will retry in background.")
            return

        w, h, fmt = cfg.get("width"), cfg.get("height"), cfg.get("format")
        res_str = f"{w}x{h}"
        if res_str in state.resolutions:
            state.current_res_idx = state.resolutions.index(res_str)
        else:
            state.resolutions.append(res_str)
            state.current_res_idx = len(state.resolutions) - 1

        if fmt in state.formats:
            state.current_fmt_idx = state.formats.index(fmt)
        else:
            state.formats.append(fmt)
            state.current_fmt_idx = len(state.formats) - 1

    def _init_window(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        window = glfw.create_window(1280, 720, "Pi Camera Control Client", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")

        glfw.make_context_current(window)
        glfw.swap_interval(1)

        imgui.create_context()
        implot.create_context()
        self.impl = GlfwRenderer(window)
        theme.apply_theme()

        self.window = window

    def _init_textures(self):
        state = self.state
        state.processed_texture_id = gl.glGenTextures(1)
        state.raw_texture_id = gl.glGenTextures(1)
        state.dng_texture_id = gl.glGenTextures(1)
        for tex_id in (state.processed_texture_id, state.raw_texture_id, state.dng_texture_id):
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex_id)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 1, 1, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, b"\x00\x00\x00")

    def _refresh_display_frame(self, reader, bgr_source_attr, tex_id, tex_size_attr, display_frame_attr, has_tex_attr):
        state = self.state
        frame, is_new = reader.get_frame()
        if frame is None:
            return
        if is_new:
            setattr(state, bgr_source_attr, frame)

        if is_new or not getattr(state, has_tex_attr) or state.settings_dirty:
            source = getattr(state, bgr_source_attr)
            if source is None:
                return
            try:
                processed = raw_convert.process_frame_for_display(source, state.color_order_idx, state.rotation_deg)
                setattr(state, display_frame_attr, processed)
                w, h = raw_convert.upload_texture(tex_id, processed)
                setattr(state, tex_size_attr, (w, h))
                setattr(state, has_tex_attr, True)
                state.bump_frame_seq()
            except Exception as e:
                print(f"Texture upload failed ({display_frame_attr}):", e)

    def run(self):
        self._load_initial_config()
        self._init_window()
        self._init_textures()

        state = self.state
        state.reader = FrameReader(f"{self.base_url}/video_feed")
        state.raw_reader = RawStreamPoller(self.base_url, lambda: state.bayer_pattern)
        state.dng_reader = DngStreamPoller(self.base_url)

        while not glfw.window_should_close(self.window):
            glfw.poll_events()

            self._refresh_display_frame(
                state.reader, "last_processed_bgr_source", state.processed_texture_id,
                "processed_tex_size", "processed_display_frame", "has_processed_texture",
            )
            self._refresh_display_frame(
                state.raw_reader, "last_raw_bgr_source", state.raw_texture_id,
                "raw_tex_size", "raw_display_frame", "has_raw_texture",
            )
            self._refresh_display_frame(
                state.dng_reader, "last_dng_bgr_source", state.dng_texture_id,
                "dng_tex_size", "dng_display_frame", "has_dng_texture",
            )
            state.settings_dirty = False

            self.impl.process_inputs()
            imgui.new_frame()

            video_window.render(state)
            settings_window.render(state)
            tools_window.render(state)

            imgui.render()

            fb_w, fb_h = glfw.get_framebuffer_size(self.window)
            gl.glViewport(0, 0, fb_w, fb_h)
            gl.glClearColor(0.06, 0.06, 0.08, 1.0)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)

            self.impl.render(imgui.get_draw_data())
            glfw.swap_buffers(self.window)

        self._shutdown()

    def _shutdown(self):
        state = self.state
        if state.reader:
            state.reader.stop()
        if state.raw_reader:
            state.raw_reader.stop()
        if state.dng_reader:
            state.dng_reader.stop()
        self.impl.shutdown()
        implot.destroy_context()
        imgui.destroy_context()
        gl.glDeleteTextures([state.processed_texture_id, state.raw_texture_id, state.dng_texture_id])
        glfw.terminate()
