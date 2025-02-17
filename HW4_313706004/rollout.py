import torch
import numpy as np

def rollout(attentions, discard_ratio, head_fusion):
    # Initialize the result as an identity matrix with the correct size
    result = torch.eye(attentions[0].size(-1), device=attentions[0].device)
    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError("Attention head fusion type not supported")

            # Drop the lowest attentions, but don't drop the class token
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
            flat[0, indices] = 0

            # Create an identity matrix with the same size as the attention_heads_fused
            I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
            a = (attention_heads_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)

    # Look at the total attention between the class token and the image patches
    mask = result[0, 0, 1:]
    # In case of 224x224 image, this brings us from 196 to 14x14
    width = int(mask.size(-1) ** 0.5)
    mask = mask.reshape(width, width).cpu().numpy()
    mask = mask / np.max(mask)
    return mask

class VITAttentionRollout:
    def __init__(self, model, attention_layer_name='attn', head_fusion="mean", discard_ratio=0.9):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio
        for name, module in self.model.named_modules():
            if attention_layer_name in name:
                module.register_forward_hook(self.get_attention)

        self.attentions = []

    def get_attention(self, module, input, output):
        print(f"Hook called on: {module}")  # Debug: Ensure hook is called
        self.attentions.append(output.cpu())

    def __call__(self, input_tensor):
        self.attentions = []
        with torch.no_grad():
            _ = self.model(input_tensor)

        return rollout(self.attentions, self.discard_ratio, self.head_fusion)
