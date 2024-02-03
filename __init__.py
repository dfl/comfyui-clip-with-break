"""
@author: dfl
@title: Comfy Nodes
@nickname: dfl
@description: CLIP text encoder that does BREAK prompting like A1111
"""

from .nodes import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeWithBreak": CLIPTextEncodeWithBreak
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeWithBreak": "CLIPTextEncode with BREAK syntax"
}
