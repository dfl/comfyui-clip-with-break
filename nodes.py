import re

class CLIPTextEncodeWithBreak:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), "clip": ("CLIP", )}}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    def encode(self, clip, text):
        prompts = re.split(r"\s*\bBREAK\b\s*", text) 
        prompt = prompts.pop(0)
        tokens = clip.tokenize(prompt)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        out = [[cond, {"pooled_output": pooled}]]

        for prompt in prompts:
            conditioning_from = out
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            conditioning_to = [[cond, {"pooled_output": pooled}]]

            # def concat(self, conditioning_to, conditioning_from):
            out = []

            if len(conditioning_from) > 1:
                print("Warning: ConditioningConcat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

            cond_from = conditioning_from[0][0]

            for i in range(len(conditioning_to)):
                t1 = conditioning_to[i][0]
                tw = torch.cat((t1, cond_from),1)
                n = [tw, conditioning_to[i][1].copy()]
                out.append(n)
        return (out, )


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "CLIPTextEncodeWithBreak": CLIPTextEncodeWithBreak
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLIPTextEncodeWithBreak": "CLIPTextEncode with BREAK syntax"
}
