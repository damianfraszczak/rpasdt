"""Default values used for network rendering."""
import os
from pathlib import Path

NODE_SIZE = 300
NODE_COLOR = "#f0f8ff"
NODE_LABEL_COLOR = "#000000"
SOURCE_NODE_COLOR = "#ff0000"
RECOVERED_NODE_COLOR = "#7fffd4"
INFECTED_NODE_COLOR = "#cd5b45"
SUSCEPTIBLE_NODE_COLOR = NODE_COLOR
ESTIMATED_SOURCE_NODE_COLOR = "#FFD700"
RESOURCES_DIR = os.path.join(Path(__file__).parent.parent.absolute(), "resources")
APP_ICON_PATH = os.path.join(RESOURCES_DIR, "icon.png")
APP_NAME = "Rumor propagation and detection toolkit"
