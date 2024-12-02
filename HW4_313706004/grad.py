# import torch
# import torchvision.transforms as T
# from PIL import Image
# import numpy as np
# from timm.models.vision_transformer import Attention
# import cv2
# import os
# import argparse

# class GradientRollout:
#     def __init__(self):
#         self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
#         for param in self.model.parameters():
#             param.requires_grad = True
#         self.attentions = []
#         self.attention_grads = []
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
#         # Replace the 'forward' method of all 'Attention' modules 
#         for name, module in self.model.named_modules():
#             if isinstance(module, Attention):
#                 # Bind the attention_forward method to the module instance
#                 module.forward = self.attention_forward.__get__(module, Attention)

#                 # Register the forward hook to extract attention weights
#                 module.register_forward_hook(self.get_attention)

#     def attention_forward(self, x):
#         # Extract attention weights
#         batch_size, seq_len, _ = x.size()
#         qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv[0], qkv[1], qkv[2]
#         attn = (q @ k.transpose(-2, -1)) * (self.scale)
#         attn = attn.softmax(dim=-1)
#         self.attn_weights = attn  # Save the attention map
#         return (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)

#     # Define a hook function to extract attention weights
#     def get_attention(self, module, input, output):
#         # Append the attention weights
#         self.attentions.append(module.attn_weights.detach().cpu())

#         # Retain gradients on attention weights
#         module.attn_weights.retain_grad()

#         # Register a hook on attn_weights to save the gradients during backward pass
#         def save_attn_grad(grad):
#             self.attention_grads.append(grad.cpu())
#         module.attn_weights.register_hook(save_attn_grad)

#     def clear_attentions(self):
#         # clear the stored attention weights
#         self.attentions = []
#         self.attention_grads = []
    
    
#     def gradient_rollout(self, head_fusion='weighted', weights=None):
#         result = torch.eye(self.attentions[0].size(-1), device=self.attentions[0].device)

#         with torch.no_grad():
#             for attention in self.attentions:
#                 # Fuse attention heads into a single map
#                 if head_fusion == "mean":
#                     attention_heads_fused = attention.mean(dim=1)
#                 elif head_fusion == "max":
#                     attention_heads_fused = attention.max(dim=1)[0]
#                 elif head_fusion == "min":
#                     attention_heads_fused = attention.min(dim=1)[0]
#                 elif head_fusion == "weighted":
#                     if weights is None or len(weights) != attention.size(1):
#                         raise ValueError("For 'weighted' fusion, weights must match the number of heads.")
#                     weights = torch.tensor(weights, device=attention.device, dtype=torch.float32)
#                     weights = weights / weights.sum()
#                     attention_heads_fused = (attention * weights.view(1, -1, 1, 1)).sum(dim=1)
#                 else:
#                     raise ValueError("Unsupported head fusion type. Use 'mean', 'max', 'min', or 'weighted'.")

#                 # Normalize the fused attention map
#                 attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)

#                 # Add skip connection
#                 attention_heads_fused += torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
#                 attention_heads_fused /= attention_heads_fused.sum(dim=-1, keepdim=True)

#                 # Perform cumulative attention rollout
#                 result = attention_heads_fused @ result

#         # Extract the attention mask for the [CLS] token
#         mask = result[0, 0, 1:]
#         width = int(mask.size(-1) ** 0.5)
#         mask = mask.reshape(width, width).cpu().numpy()
#         mask = mask / np.max(mask)  # Normalize to [0, 1]
        
#         return mask


#     def perform_backpropagation(self, input_tensor, target_category):
#         # 1. Inference the model
#         output = self.model(input_tensor)

#         # 2. Define a simple loss function that focuses on the target category
#         loss = output[0, target_category]

#         # 3. Perform backpropagation with respect to the loss
#         self.model.zero_grad()
#         loss.backward()
        
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
    
#     def run(self, image_path, target_category):
#         # clean previous attention maps and result
#         self.clear_attentions()
        
#         # get the image name for saving output image
#         image_name = os.path.basename(image_path)  # e.g., 'image.jpg'
#         image_name, _ = os.path.splitext(image_name)  # ('image', '.jpg')
        
#         # convert image to a tensor
#         img = Image.open(image_path).convert('RGB')
#         input_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
#         input_tensor.requires_grad_(True)

#         # do backpropagation manually during inference to produce gradients
#         self.perform_backpropagation(input_tensor, target_category)

#         np_img = np.array(img)[:, :, ::-1]
#         mask = self.gradient_rollout()
#         output_heatmap = self.show_mask_on_image(np_img, mask)
#         output_filename = f"gradient_result_{image_name}.png"  
#         cv2.imwrite(output_filename, output_heatmap)

# if __name__ == '__main__':
#     # arg parsing
#     parser = argparse.ArgumentParser(description='Process an image for attention visualization with respect to target category.')
#     parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
#     parser.add_argument('--category', type=int, required=True, help="target category of attention flows")
#     args = parser.parse_args()

#     # Execution
#     model = GradientRollout()
#     outputs = model.run(args.image, args.category)

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
from timm.models.vision_transformer import Attention
import cv2
import os
import argparse

class GradientRollout:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/deit:main', 'deit_tiny_patch16_224', pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = True
        self.attentions = []
        self.attention_grads = []
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
        # Replace the 'forward' method of all 'Attention' modules 
        for name, module in self.model.named_modules():
            if isinstance(module, Attention):
                # Bind the attention_forward method to the module instance
                module.forward = self.attention_forward.__get__(module, Attention)

                # Register the forward hook to extract attention weights
                module.register_forward_hook(self.get_attention)

    def attention_forward(self, x):
        # Extract attention weights
        batch_size, seq_len, _ = x.size()
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.scale)
        attn = attn.softmax(dim=-1)
        self.attn_weights = attn  # Save the attention map
        return (attn @ v).transpose(1, 2).reshape(batch_size, seq_len, -1)

    # Define a hook function to extract attention weights
    def get_attention(self, module, input, output):
        # Append the attention weights
        self.attentions.append(module.attn_weights.detach().cpu())

        # Retain gradients on attention weights
        module.attn_weights.retain_grad()

        # Register a hook on attn_weights to save the gradients during backward pass
        def save_attn_grad(grad):
            self.attention_grads.append(grad.cpu())
        module.attn_weights.register_hook(save_attn_grad)

    def clear_attentions(self):
        # clear the stored attention weights
        self.attentions = []
        self.attention_grads = []
   

    

    def gradient_rollout(self, discard_ratio = 0.9):
        result = torch.eye(self.attentions[0].size(-1), device=self.attentions[0].device)

        with torch.no_grad():
            for attention, grad in zip(self.attentions, self.attention_grads):
                # 使用梯度作为注意力的权重
                weights = grad
                attention_heads_fused = (attention * weights).mean(dim=1)

                # 将小于0的注意力值设为0（因为负值可能没有意义）
                attention_heads_fused[attention_heads_fused < 0] = 0

                # 丢弃最小的注意力值，但保留 [CLS] token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1) * discard_ratio), dim=-1, largest=False)
                flat[0, indices] = 0

                # 添加单位矩阵用于保持原始的自注意力特性
                I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
                a = (attention_heads_fused + I) / 2
                a = a / a.sum(dim=-1, keepdim=True)

                # 逐层累积注意力
                result = torch.matmul(a, result)

        # 获取最终的注意力图，排除 [CLS] token
        mask = result[0, 0, 1:]
        width = int(mask.size(-1) ** 0.5)
        mask = mask.reshape(width, width).cpu().numpy()
        mask = mask / np.max(mask)  # 归一化到 [0, 1]

        # inverted_mask = 1 - mask  # 取反操作
        # inverted_mask = inverted_mask / np.max(mask)  # 再次归一化到 [0, 1]

        return mask




    def perform_backpropagation(self, input_tensor, target_category):
        self.model.zero_grad()
        
        # 前向传播并计算输出
        output = self.model(input_tensor)
        
         # 创建类别掩码，确保仅对目标类别进行反向传播
        category_mask = torch.zeros_like(output)
        category_mask[:, target_category] = 1
        
        # 计算损失，保留目标类别的损失
        loss = (output * category_mask).sum()
        
        # 反向传播以计算梯度
        loss.backward(retain_graph=True)
        
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
    
    def run(self, image_path, target_category):
        # clean previous attention maps and result
        self.clear_attentions()
        
        # get the image name for saving output image
        image_name = os.path.basename(image_path)  # e.g., 'image.jpg'
        image_name, _ = os.path.splitext(image_name)  # ('image', '.jpg')
        
        # convert image to a tensor
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0)  # Add batch dimension
        input_tensor.requires_grad_(True)

        # do backpropagation manually during inference to produce gradients
        self.perform_backpropagation(input_tensor, target_category)

        np_img = np.array(img)[:, :, ::-1]
        mask = self.gradient_rollout()
        output_heatmap = self.show_mask_on_image(np_img, mask)
        output_filename = f"gradient_result_{image_name}.png"  
        cv2.imwrite(output_filename, output_heatmap)

if __name__ == '__main__':
    # arg parsing
    parser = argparse.ArgumentParser(description='Process an image for attention visualization with respect to target category.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--category', type=int, required=True, help="target category of attention flows")
    args = parser.parse_args()

    # Execution
    model = GradientRollout()
    outputs = model.run(args.image, args.category)
