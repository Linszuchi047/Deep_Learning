# import torch
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np
# from timm.models.vision_transformer import Attention
# import cv2
# import os
# import argparse


# class AttentionRollout:
#     def __init__(self):
#         self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
#         self.attentions = []  # list that should contain attention maps of each layer
#         self.num_heads = 3
#         self.transform = T.Compose([
#             T.Resize(256),
#             T.CenterCrop(224),
#             T.ToTensor(),
#             T.Normalize(
#                 mean=(0.485, 0.456, 0.406),  # ImageNet mean
#                 std=(0.229, 0.224, 0.225)    # ImageNet std
#             )
#         ])
        
#         # Initialize weights for weighted average and adaptive fusion strategies
#         self.head_weights = torch.nn.Parameter(torch.ones(self.num_heads) / self.num_heads)
#         self.layer_weights = None  # Initialized later after knowing the number of layers
        
#         # Replace the 'forward' method of all 'Attention' modules 
#         for name, module in self.model.named_modules():
#             if isinstance(module, Attention):
#                 # Bind the attention_forward method to the module instance
#                 module.forward = self.attention_forward.__get__(module, Attention)

#                 # Register the forward hook to extract attention weights
#                 module.register_forward_hook(self.get_attention)

#     @staticmethod
#     def attention_forward(self, x):
#         # Query, Key, Value computation
#         qkv = self.qkv(x).reshape(x.shape[0], x.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]

#         # Compute scaled dot-product attention
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = attn.softmax(dim=-1)

#         # Save the attention map
#         self.attn_weights = attn  # Save attention weights for later use

#         # Return the output of the attention operation
#         out = (attn @ v).transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
#         out = self.proj(out)
#         return out
    
#     def get_attention(self, module, input, output):
#         self.attentions.append(module.attn_weights.detach().cpu())
        
#     def clear_attentions(self):
#         # clear the stored attention weights
#         self.attentions = []
    
#     def attention_rollout(self, mode='mean'):
#         # Set layer weights after collecting attentions if using adaptive fusion
#         if mode == 'adaptive' and self.layer_weights is None:
#             self.layer_weights = torch.nn.Parameter(torch.ones(len(self.attentions)) / len(self.attentions))
        
#         result = torch.eye(self.attentions[0].size(-1))  # Initialize the result with an identity matrix
#         with torch.no_grad():
#             for idx, attention in enumerate(self.attentions):
#                 if mode == 'weighted':
#                     # Use head_weights to compute a weighted average across heads
#                     weighted_attention = (attention * self.head_weights.view(1, -1, 1, 1)).sum(dim=1)
#                     attention_fused = weighted_attention
#                 elif mode == 'adaptive':
#                     # Use layer_weights to weigh each layer's attention map
#                     layer_weight = self.layer_weights[idx]
#                     attention_fused = attention.mean(dim=1) * layer_weight
#                 elif mode == 'mean':
#                     attention_fused = attention.mean(dim=1)  # Mean across heads
#                 elif mode == 'min':
#                     attention_fused = attention.min(dim=1).values  # Minimum across heads
#                 elif mode == 'max':
#                     attention_fused = attention.max(dim=1).values  # Maximum across heads
#                 else:
#                     raise ValueError(f"Invalid mode: {mode}. Choose from 'weighted', 'adaptive', 'mean', 'min', or 'max'.")

#                 # Add skip connection (identity matrix)
#                 attention_with_skip = attention_fused + torch.eye(attention_fused.size(-1)).to(attention_fused.device)

#                 # Normalize the attention map row-wise
#                 attention_with_skip = attention_with_skip / attention_with_skip.sum(dim=-1, keepdim=True)

#                 # Multiply with the cumulative result from the previous layer
#                 result = attention_with_skip @ result

#         mask = result[0, 0, 1:]  # Extract the attention flow for the [CLS] token
#         width = int(mask.size(-1) ** 0.5)  # Assuming square attention
#         mask = mask.reshape(width, width).numpy()
#         mask = mask / np.max(mask)
#         return mask
    
#     def show_mask_on_image(self, img, mask):
#         # Normalize the value of img to [0.0, 1.0]
#         img = np.float32(img) / 255

#         # Reshape the mask to 224x224 for later computation
#         mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

#         # Generate heatmap and normalize the value
#         heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
#         heatmap = np.float32(heatmap) / 255

#         # Add heatmap and original image together and normalize the value
#         combination = heatmap + np.float32(img)
#         combination = combination / np.max(combination)
        
#         # Scale back the value to [0.0, 255.0]
#         combination = np.uint8(255 * combination)

#         return combination
    
#     def run(self, image_path):
#         # clean previous attention maps and result
#         self.clear_attentions()
        
#         # get the image name for saving output image
#         image_name = os.path.basename(image_path)  # e.g., 'image.jpg'
#         image_name, _ = os.path.splitext(image_name)  # ('image', '.jpg')
        
#         # convert image to a tensor
#         img = Image.open(image_path).convert('RGB')
#         input_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension

#         # run the process of gathering attention flows
#         # and put the mask on the input image
#         with torch.no_grad():
#             outputs = self.model(input_tensor)
#             np_img = np.array(img)[:, :, ::-1]
#             mask = self.attention_rollout()
#             output_heatmap = self.show_mask_on_image(np_img, mask)
#             output_filename = f"result_{image_name}.png"  
#             cv2.imwrite(output_filename, output_heatmap)

# if __name__ == '__main__':
#     # arg parsing
#     parser = argparse.ArgumentParser(description='Process an image for attention visualization.')
#     parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
#     args = parser.parse_args()

#     # Execution
#     model = AttentionRollout()
#     with torch.no_grad():
#         outputs = model.run(args.image)

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
    
    def attention_rollout(self, mode='mean', normalization='layer_norm'):
        # Set layer weights after collecting attentions if using adaptive fusion
        if mode == 'adaptive' and self.layer_weights is None:
            self.layer_weights = torch.nn.Parameter(torch.ones(len(self.attentions)) / len(self.attentions))
        
        result = torch.eye(self.attentions[0].size(-1))  # Initialize the result with an identity matrix
        with torch.no_grad():
            for idx, attention in enumerate(self.attentions):
                if mode == 'weighted':
                    # Use head_weights to compute a weighted average across heads
                    weighted_attention = (attention * self.head_weights.view(1, -1, 1, 1)).sum(dim=1)
                    attention_fused = weighted_attention
                elif mode == 'adaptive':
                    # Use layer_weights to weigh each layer's attention map
                    layer_weight = self.layer_weights[idx]
                    attention_fused = attention.mean(dim=1) * layer_weight
                elif mode == 'mean':
                    attention_fused = attention.mean(dim=1)  # Mean across heads
                elif mode == 'min':
                    attention_fused = attention.min(dim=1).values  # Minimum across heads
                elif mode == 'max':
                    attention_fused = attention.max(dim=1).values  # Maximum across heads
                else:
                    raise ValueError(f"Invalid mode: {mode}. Choose from 'weighted', 'adaptive', 'mean', 'min', or 'max'.")

                # Add weighted skip connection (identity matrix)
                attention_with_skip = attention_fused + self.residual_weights * torch.eye(attention_fused.size(-1)).to(attention_fused.device)

                # Normalize the attention map row-wise
                if normalization == 'layer_norm':
                    attention_with_skip = torch.nn.functional.layer_norm(attention_with_skip, attention_with_skip.size())
                elif normalization == 'batch_norm':
                    attention_with_skip = torch.nn.functional.batch_norm(attention_with_skip, running_mean=None, running_var=None, training=True)
                elif normalization == 'relu':
                    attention_with_skip = torch.nn.functional.relu(attention_with_skip)
                else:
                    attention_with_skip = attention_with_skip / attention_with_skip.sum(dim=-1, keepdim=True)

                # Multiply with the cumulative result from the previous layer
                result = attention_with_skip @ result

        mask = result[0, 0, 1:]  # Extract the attention flow for the [CLS] token
        width = int(mask.size(-1) ** 0.5)  # Assuming square attention
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)

        # Apply thresholding to focus only on main object regions
        _, mask = cv2.threshold(mask, 0.6, 1.0, cv2.THRESH_BINARY)
        return mask
    
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
