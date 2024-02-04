import re
import nodes as native
class CLIPTextEncodeWithBreak:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, text):
        prompts = re.split(r"\s*\bBREAK\b\s*", text) 
        # encode first prompt fragment
        prompt = prompts.pop(0)
        print(f"prompt: {prompt}")
        out = native.CLIPTextEncode.encode(self, clip, prompt)
        # encode and concatenate the rest of the prompt
        for prompt in prompts:
            cond_to = native.CLIPTextEncode.encode(self, clip, prompt)
            out = native.ConditioningConcat.concat(self, cond_to[0], out[0])
        return out


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeWithBreak": CLIPTextEncodeWithBreak
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeWithBreak": "CLIPTextEncode with BREAK syntax"
}
