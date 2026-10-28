#!/home/will/micromamba/envs/MECH_5720/bin/python3 
"""
Entry point for the Pi camera control/viewer client. Run this on your PC /
laptop.

Requires: opencv-python, requests, glfw, pyopengl, imgui-bundle, rawpy
  pip install opencv-python requests glfw pyopengl imgui-bundle rawpy

See CLAUDE.md for the file layout, default state, and how to add a new
Analysis Tools tab.
"""

import argparse

from app import App


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=8089, type=int)
    return parser.parse_args()


def main():
    args = parse_args()
    App(args.host, args.port).run()


if __name__ == "__main__":
    main()
