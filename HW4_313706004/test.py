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
        self.layer_weights = None  # Initialized later after knowing the number of layers
        self.residual_weights = torch.nn.Parameter(torch.tensor(0.5))  # Weighted residual connection
        
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
    
    # def attention_rollout(self ,discard_ratio=0.9, head_fusion="max"):
    # # Set layer weights after collecting attentions if using adaptive fusion
    #     result = torch.eye(self.attentions[0].size(-1), device=self.attentions[0].device)
    #     with torch.no_grad():
    #         for attention in self.attentions:
    #             if head_fusion == "mean":
    #                 attention_heads_fused = attention.mean(dim=1)
    #             elif head_fusion == "max":
    #                 attention_heads_fused = attention.max(dim=1)[0]
    #             elif head_fusion == "min":
    #                 attention_heads_fused = attention.min(dim=1)[0]
    #             else:
    #                 raise ValueError("Attention head fusion type not supported")

    #             # Drop the lowest attentions, but don't drop the class token
    #             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
    #             _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
    #             flat[0, indices] = 0

    #             # Create an identity matrix with the same size as the attention_heads_fused
    #             I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
    #             a = (attention_heads_fused + I) / 2
    #             a = a / a.sum(dim=-1, keepdim=True)

    #             result = torch.matmul(a, result)

    #     mask = result[0, 0, 1:]  # Extract the attention flow for the [CLS] token
    #     width = int(mask.size(-1) ** 0.5)  # Assuming square attention
    #     mask = mask.reshape(width, width).numpy()
    #     mask = mask / np.max(mask)

    #     # Apply thresholding to focus only on main object regions
    #     _, mask = cv2.threshold(mask, 0.6, 1.0, cv2.THRESH_BINARY)
    #     return mask

    # def attention_rollout(self, discard_ratio=0.9, head_fusion="max", target_classes=None):
    #     """
    #     Perform attention rollout for one or more target classes.
    #     If target_classes is None, use all tokens except [CLS].
    #     """
    #     result = torch.eye(self.attentions[0].size(-1), device=self.attentions[0].device)

    #     with torch.no_grad():
    #         for attention in self.attentions:
    #             # Fuse attention heads
    #             if head_fusion == "mean":
    #                 attention_heads_fused = attention.mean(dim=1)
    #             elif head_fusion == "max":
    #                 attention_heads_fused = attention.max(dim=1)[0]
    #             else:
    #                 raise ValueError("Unsupported head fusion type.")

    #             # Drop the lowest attentions
    #             flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
    #             _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
    #             flat[0, indices] = 0

    #             # Add residual connection and normalize
    #             I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
    #             a = (attention_heads_fused + I) / 2
    #             a = a / a.sum(dim=-1, keepdim=True)

    #             result = torch.matmul(a, result)

    #     # Generate mask for all target classes or default to all tokens
    #     if target_classes is None:
    #         target_classes = list(range(1, result.size(1)))  # Default: all tokens except [CLS]

    #     combined_mask = None
    #     for class_idx in target_classes:
    #         mask = result[0, class_idx, 1:]  # Exclude the [CLS] token
    #         width = int(mask.size(-1) ** 0.5)  # Assuming square attention
    #         mask = mask.reshape(width, width).numpy()
    #         mask = mask / np.max(mask)  # Normalize to [0, 1]

    #         # Threshold to focus on main regions
    #         _, mask = cv2.threshold(mask, 0.6, 1.0, cv2.THRESH_BINARY)

    #         if combined_mask is None:
    #             combined_mask = mask
    #         else:
    #             combined_mask = np.maximum(combined_mask, mask)

    #     return combined_mask

    def attention_rollout(self, discard_ratio=0.92, head_fusion="max", target_classes=None):
        """
        Perform attention rollout with smooth mask generation and multiple head fusion strategies.
        :param discard_ratio: Percentage of lowest attention weights to discard.
        :param head_fusion: Strategy for fusing attention heads ('mean', 'max', 'min').
        :param target_classes: A list of target tokens to focus on (default: all tokens except [CLS]).
        """
        result = torch.eye(self.attentions[0].size(-1), device=self.attentions[0].device)

        with torch.no_grad():
            for attention in self.attentions:
                # Fuse attention heads
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(dim=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(dim=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(dim=1)[0]
                else:
                    raise ValueError("Unsupported head fusion type. Choose from 'mean', 'max', or 'min'.")

                # Drop the lowest attentions (optional)
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
                flat[0, indices] = 0

                # Add residual connection and normalize
                I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
                a = (attention_heads_fused + I) / 2
                a = a / a.sum(dim=-1, keepdim=True)

                result = torch.matmul(a, result)

        # Use the result without binarization
        if target_classes is None:
            target_classes = list(range(1, result.size(1)))  # Default: all tokens except [CLS]

        combined_mask = None
        for class_idx in target_classes:
            mask = result[0, class_idx, 1:]  # Extract attention map without [CLS] token
            width = int(mask.size(-1) ** 0.5)
            mask = mask.reshape(width, width).numpy()
            mask = mask / np.max(mask)  # Normalize to [0, 1]

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


# import torch
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np
# import cv2
# import os
# import argparse
# from timm.models.vision_transformer import Attention


# class AttentionRollout:
#     def __init__(self):
#         # Load a pre-trained DeiT model
#         self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
#         self.attentions = []  # Store attention maps for all layers
#         self.num_heads = 3
#         self.transform = T.Compose([
#             T.Resize(256),
#             T.CenterCrop(224),
#             T.ToTensor(),
#             T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
#         ])
        
#         # Residual scaling weights (learnable parameters)
#         self.residual_weights = torch.nn.Parameter(torch.tensor([0.5, 0.5]))  # alpha, beta
        
#         # Replace attention forward and register hooks
#         for name, module in self.model.named_modules():
#             if isinstance(module, Attention):
#                 module.forward = self.attention_forward.__get__(module, Attention)
#                 module.register_forward_hook(self.get_attention)

#     @staticmethod
#     def attention_forward(self, x):
#         # Compute query, key, value
#         qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         # Scaled dot-product attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)

#         # Save attention weights for visualization
#         self.attn_weights = attn

#         # Attention output
#         out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
#         out = self.proj(out)
#         return out

#     def get_attention(self, module, input, output):
#         self.attentions.append(module.attn_weights.detach().cpu())

#     def clear_attentions(self):
#         self.attentions = []

#     def attention_rollout(self, discard_ratio=0.9, head_fusion="mean"):
#         """
#         Perform attention rollout with residual scaling.
#         """
#         # Initialize result with identity matrix
#         result = torch.eye(self.attentions[0].size(-1), device=self.attentions[0].device)

#         with torch.no_grad():
#             for idx, attention in enumerate(self.attentions):
#                 # Fuse heads
#                 if head_fusion == "mean":
#                     attention_heads_fused = attention.mean(dim=1)
#                 elif head_fusion == "max":
#                     attention_heads_fused = attention.max(dim=1)[0]
#                 elif head_fusion == "min":
#                     attention_heads_fused = attention.min(dim=1)[0]
#                 else:
#                     raise ValueError(f"Head fusion type '{head_fusion}' not supported. Use 'mean', 'max', or 'min'.")

#                 # Apply residual scaling
#                 alpha, beta = self.residual_weights[0].item(), self.residual_weights[1].item()
#                 attention_heads_fused = alpha * attention_heads_fused + beta * torch.eye(
#                     attention_heads_fused.size(-1), device=attention_heads_fused.device
#                 )

#                 # Normalize attention
#                 attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)

#                 # Drop low attention values based on discard_ratio
#                 flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
#                 _, indices = flat.topk(int(flat.size(-1) * discard_ratio), -1, largest=False)
#                 flat[0, indices] = 0

#                 # Update result by multiplying with current attention matrix
#                 result = torch.matmul(attention_heads_fused, result)

#         # Extract the attention flow for the [CLS] token
#         mask = result[0, 0, 1:]  # Removing the [CLS] token
#         width = int(mask.size(-1) ** 0.5)  # Assuming square attention
#         mask = mask.reshape(width, width).numpy()
#         mask = mask / np.max(mask)  # Normalize to [0, 1]

#         # Apply thresholding to focus only on main object regions
#         _, mask = cv2.threshold(mask, 0.6, 1.0, cv2.THRESH_BINARY)

#         return mask

#     def show_mask_on_image(self, img, mask):
#         """
#         Overlay the attention mask on the input image.
#         """
#         img = np.float32(img) / 255
#         mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

#         heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#         heatmap = np.float32(heatmap) / 255

#         combination = heatmap + np.float32(img)
#         combination = combination / np.max(combination)

#         return np.uint8(255 * combination)

#     def run(self, image_path):
#         """
#         Process an input image and visualize attention.
#         """
#         self.clear_attentions()

#         # Extract image name for saving results
#         image_name = os.path.basename(image_path)
#         image_name, _ = os.path.splitext(image_name)

#         # Transform the image
#         img = Image.open(image_path).convert('RGB')
#         input_tensor = self.transform(img).unsqueeze(0)

#         # Run the model and compute attention rollout
#         with torch.no_grad():
#             outputs = self.model(input_tensor)
#             np_img = np.array(img)[:, :, ::-1]
#             mask = self.attention_rollout()
#             output_heatmap = self.show_mask_on_image(np_img, mask)
#             output_filename = f"result_{image_name}.png"
#             cv2.imwrite(output_filename, output_heatmap)


# if __name__ == '__main__':
#     # Argument parsing
#     parser = argparse.ArgumentParser(description='Process an image for attention visualization.')
#     parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
#     args = parser.parse_args()

#     # Execution
#     model = AttentionRollout()
#     model.run(args.image)
