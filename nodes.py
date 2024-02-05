import re
import nodes as native
from .adv_encode import advanced_encode #, advanced_encode_XL

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
        # print(f"prompt: {prompt}")
        out = native.CLIPTextEncode.encode(self, clip, prompt)
        # encode and concatenate the rest of the prompt
        for prompt in prompts:
            # print(f"prompt: {prompt}")
            cond_from = native.CLIPTextEncode.encode(self, clip, prompt)
            out = native.ConditioningConcat.concat(self, out[0], cond_from[0])
        return out

class AdvancedCLIPTextEncodeWithBreak:

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"multiline": True}),
            "clip": ("CLIP", ),
            "token_normalization": (["none", "mean", "length", "length+mean"],),
            "weight_interpretation": (["comfy", "A1111", "compel", "comfy++" ,"down_weight"],),
            #"affect_pooled": (["disable", "enable"],),
            }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning/advanced"

    def _encode(self, clip, text, token_normalization, weight_interpretation, affect_pooled='disable'):
        embeddings_final, pooled = advanced_encode(clip, text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=affect_pooled=='enable')
        return ([[embeddings_final, {"pooled_output": pooled}]], )

    def encode(self, clip, text, token_normalization, weight_interpretation, affect_pooled='disable'):
        prompts = re.split(r"\s*\bBREAK\b\s*", text) 
        # encode first prompt fragment
        prompt = prompts.pop(0)
        # print(f"prompt: {prompt}")
        out = self._encode(clip, prompt, token_normalization, weight_interpretation, affect_pooled)
        # encode and concatenate the rest of the prompt
        for prompt in prompts:
            # print(f"prompt: {prompt}")
            cond_to = self._encode(clip, prompt, token_normalization, weight_interpretation, affect_pooled)
            out = native.ConditioningConcat.concat(self, cond_to[0], out[0])
        return out



# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeWithBreak": CLIPTextEncodeWithBreak,
    "AdvancedCLIPTextEncodeWithBreak": AdvancedCLIPTextEncodeWithBreak
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeWithBreak": "CLIPTextEncode with BREAK syntax",
    "AdvancedCLIPTextEncodeWithBreak": "AdvancedCLIPTextEncode with BREAK syntax"
}
