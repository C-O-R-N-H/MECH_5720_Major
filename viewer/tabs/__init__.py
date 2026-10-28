"""
Tab registry for the Analysis Tools window.

To add a new tab: implement tabs/base.py's Tab shape (a `name` and a
`render(state)`) in its own module here, then add an instance to TABS below.
Nothing outside this file needs to change.
"""

from tabs.channels import ChannelsTab
from tabs.histogram import HistogramTab
from tabs.raw_bayer import RawBayerTab

TABS = [
    ChannelsTab(),
    HistogramTab(),
    RawBayerTab(),
]
