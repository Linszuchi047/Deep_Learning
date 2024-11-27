import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from timm.models.vision_transformer import Attention
import cv2
import os
import argparse

class AttentionRollout:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        self.attentions = []  # list that should contain attention maps of each layer
        self.num_heads = 3
        self.transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406),  # ImageNet mean
                std=(0.229, 0.224, 0.225)    # ImageNet std
            )
        ])
        
        # Initialize weights for weighted average and adaptive fusion strategies
        self.head_weights = torch.nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
        self.residual_weights = torch.nn.Parameter(torch.tensor(0.8))  # Increase residual contribution slightly
        
        # Replace the 'forward' method of all 'Attention' modules 
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                # Bind the attention_forward method to the module instance
                module.forward = self.attention_forward.__get__(module, Attention)

                # Register the forward hook to extract attention weights
                module.register_forward_hook(self.get_attention)

    @staticmethod
    def attention_forward(self, x):
        # Query, Key, Value computation
        qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Save the attention map
        self.attn_weights = attn  # Save attention weights for later use

        # Return the output of the attention operation
        out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
        out = self.proj(out)
        return out
    
    def get_attention(self, module, input, output):
        self.attentions.append(module.attn_weights.detach().cpu())
        
    def clear_attentions(self):
        # clear the stored attention weights
        self.attentions = []
    
    def attention_rollout(self, discard_ratio=0.7, head_fusion="weighted", target_classes=None):
        """
        Perform attention rollout with enhanced focus on neglected regions and balanced coverage.
        """
        result = torch.eye(self.attentions[0].size(-1), device=self.attentions[0].device)

        with torch.no_grad():
            for attention in self.attentions:
                # Fuse attention heads adaptively
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(dim=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(dim=1)[0]
                elif head_fusion == "weighted":
                    # Use head_weights for weighted fusion
                    attention_heads_fused = (attention * self.head_weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
                else:
                    raise ValueError("Unsupported head fusion type. Choose from 'mean', 'max', or 'weighted'.")

                # Retain lower attention values more by reducing discard_ratio
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
                flat[0, indices] = 0

                # Add residual connection and normalize
                I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
                a = (attention_heads_fused + self.residual_weights * I) / (1 + self.residual_weights)
                a = a / a.sum(dim=-1, keepdim=True)

                result = torch.matmul(a, result)

        # Focus on multiple target classes or default to all tokens
        if target_classes is None:
            target_classes = list(range(1, result.size(1)))  # Default: all tokens except [CLS]

        combined_mask = None
        for class_idx in target_classes:
            mask = result[0, class_idx, 1:]  # Extract attention map excluding [CLS]
            width = int(mask.size(-1) ** 0.5)
            mask = mask.reshape(width, width).numpy()
            mask = mask / np.max(mask)  # Normalize to [0, 1]

            # Amplify neglected areas using lower thresholds and ensure coverage
            mask = np.clip(mask, 0.2, 1.0)  # Adjust threshold to retain low values

            if combined_mask is None:
                combined_mask = mask
            else:
                combined_mask = np.maximum(combined_mask, mask)

        return combined_mask

    def show_mask_on_image(self, img, mask):
        # Normalize the value of img to [0.0, 1.0]
        img = np.float32(img) / 255

        # Reshape the mask to 224x224 for later computation
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # Generate heatmap and normalize the value
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255

        # Add heatmap and original image together and normalize the value
        combination = heatmap + np.float32(img)
        combination = combination / np.max(combination)
        
        # Scale back the value to [0.0, 255.0]
        combination = np.uint8(255 * combination)

        return combination
    
    def run(self, image_path):
        # clean previous attention maps and result
        self.clear_attentions()
        
        # get the image name for saving output image
        image_name = os.path.basename(image_path)  # e.g., 'image.jpg'
        image_name, _ = os.path.splitext(image_name)  # ('image', '.jpg')
        
        # convert image to a tensor
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension

        # run the process of gathering attention flows
        # and put the mask on the input image
        with torch.no_grad():
            outputs = self.model(input_tensor)
            np_img = np.array(img)[:, :, ::-1]
            mask = self.attention_rollout()
            output_heatmap = self.show_mask_on_image(np_img, mask)
            output_filename = f"result_{image_name}.png"  
            cv2.imwrite(output_filename, output_heatmap)


if __name__ == '__main__':
    # arg parsing
    parser = argparse.ArgumentParser(description='Process an image for attention visualization.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()

    # Execution
    model = AttentionRollout()
    with torch.no_grad():
        outputs = model.run(args.image)


