"""
LM Studio Nodes for ComfyUI - Integration with local LLM models
"""

import importlib
import subprocess
import sys

def _ensure_lmstudio():
    """Install lmstudio if it is not already available in the current Python environment."""
    if importlib.util.find_spec("lmstudio") is None:
        print("[ComfyExpo LM Studio] lmstudio SDK not found — installing...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "lmstudio"],
            stdout=subprocess.DEVNULL,
        )
        print("[ComfyExpo LM Studio] lmstudio installed successfully.")

_ensure_lmstudio()

# Import all node classes from the main module file
from .expo_lmstudio_imagetotext import (
    ExpoLmstudioUnified,
    ExpoLmstudioImageToText,
    ExpoLmstudioTextGeneration,
    ExpoLmstudioStructuredOutput
)
from .random_list_picker import RandomListPicker
from .expo_lmstudio_multi_image import (
    NODE_CLASS_MAPPINGS as _MULTI_IMAGE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as _MULTI_IMAGE_DISPLAY_MAPPINGS,
)

# Define how ComfyUI maps the node name (used in backend) to the class
NODE_CLASS_MAPPINGS = {
    "Expo Lmstudio Unified": ExpoLmstudioUnified,
    "Expo Lmstudio Image To Text": ExpoLmstudioImageToText,
    "Expo Lmstudio Text Generation": ExpoLmstudioTextGeneration,
    "Expo Lmstudio Structured Output": ExpoLmstudioStructuredOutput,
    "Random List Picker": RandomListPicker
}
NODE_CLASS_MAPPINGS.update(_MULTI_IMAGE_CLASS_MAPPINGS)

# Define how ComfyUI maps the node name to its display name (shown in the UI)
NODE_DISPLAY_NAME_MAPPINGS = {
    "Expo Lmstudio Unified": "LM Studio (Unified)",
    "Expo Lmstudio Image To Text": "LM Studio (Image to Text)",
    "Expo Lmstudio Text Generation": "LM Studio (Text Gen)",
    "Expo Lmstudio Structured Output": "LM Studio (Structured Output)",
    "Random List Picker": "Random List Picker"
}
NODE_DISPLAY_NAME_MAPPINGS.update(_MULTI_IMAGE_DISPLAY_MAPPINGS)

# Standard dictionary telling ComfyUI what this package provides
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("--- ComfyExpo LM Studio Nodes Loaded ---")
