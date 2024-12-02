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
        self.attentions = [] # list that should contain attention maps of each layer
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

    @staticmethod
    def attention_forward(self, x):
        """

        TODO:
            Implement the attention computation and store the attention maps
            You need to save the attention map into variable "self.attn_weights"
            
            Note: Due to @staticmethod, "self" here refers to the "Attention" module instance, not the class itself.

        """
        # write your code here
        self.attn_weights = None  # save the attention map into this variable
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
    

    def attention_rollout(self, head_fusion = 'mean', discard_ratio = 0.6, weights=[0.2, 0.6, 0.1]):
        """

        TODO:
        Define the attention rollout function that accumulate the final attention flows.
        You need to return parameter "mask", which should be the final attention flows.  
        You can follow the below procedures:

        For each attention layers:
            1. filter the attention by mean/min/max fiter with respect to 3 attention heads
            2. (optional)normalize the attention map of current layer
            3. perform matrix multiplication on current attention map and previous results
        
        (4. and 5. are already done for you, you just need to get the variable "result" correctly.)
        4. Obtain the mask: Gather the attention flows that use [CLS] token as query and the rest tokens as keys
        5. Normalize the values inside the mask to [0.0, 1.0]

        """
        result = torch.eye(self.attentions[0].size(-1))
        with torch.no_grad():
            for attention in self.attentions:
                # Write your code(Attention Rollout) here to update "result" matrix
   
                if head_fusion == "mean":
                    attention_heads_fused = attention.mean(axis=1)
                elif head_fusion == "max":
                    attention_heads_fused = attention.max(axis=1)[0]
                elif head_fusion == "min":
                    attention_heads_fused = attention.min(axis=1)[0]
                elif head_fusion == "weighted":
                    weights = torch.tensor(weights, device=attention.device, dtype=attention.dtype)
                    weights = weights / weights.sum()  # Normalize weights to sum to 1
                    attention_heads_fused = (attention * weights.view(1, -1, 1, 1)).sum(dim=1)  # Weighted sum
                else:
                    raise ValueError("Unsupported head fusion type. Choose from 'mean', 'min', 'max', or 'weighted'.")

                # Drop the lowest attentions, but
                # don't drop the class token
                flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
                _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)
                indices = indices[indices != 0]
                flat[0, indices] = 0

                I = torch.eye(attention_heads_fused.size(-1))
                a = (attention_heads_fused + 1.0*I)/2
                a = a / a.sum(dim=-1)

                result = torch.matmul(a, result)
                # Normalize the attention matrix to ensure values sum to 1
                # attention_heads_fused = attention_heads_fused / attention_heads_fused.sum(dim=-1, keepdim=True)
            
                # # Add skip connection (identity matrix)
                # I = torch.eye(attention_heads_fused.size(-1), device=attention_heads_fused.device)
                # attention_with_skip = attention_heads_fused + I

                # # Normalize again to ensure rows sum to 1
                # attention_with_skip = attention_with_skip / attention_with_skip.sum(dim=-1, keepdim=True)

                # # Apply cumulative attention rollout: R_l = A_l * R_(l-1)
                # result = torch.matmul(attention_with_skip, result)

        # Optionally focus on specific target classes
        # if target_classes is None:
        #     target_classes = list(range(1, result.size(1)))  # Default: all tokens except [CLS]

        # combined_mask = None
        # for class_idx in target_classes:
        #     # Extract the attention map for the target class
        #     mask = result[0, class_idx, 1:]  # Exclude [CLS] token
        #     width = int(mask.size(-1) ** 0.5)
        #     mask = mask.reshape(width, width).numpy()
        #     mask = mask / np.max(mask)  # Normalize to [0, 1]

        #     # Apply further enhancements for background suppression and target emphasis
        #     mask = np.where(mask < 0.1, mask * 0.05, mask)  # Suppress low attention values
        #     mask = np.clip(mask, 0.1, 1.0)  # Clip values for better contrast

        #     # Double non-linear transformation for stronger contrast
        #     mask = np.power(mask, 2.0)
        #     mask = np.power(mask, 1.5)

        #     # Normalize again for consistent visualization
        #     mask = mask / np.max(mask)

        #     if combined_mask is None:
        #         combined_mask = mask
        #     else:
        #         combined_mask = np.maximum(combined_mask, mask)

        # return combined_mask




        """
        if you have variable "result" in correct shape(shape of attenion map), then this part should work properly.
        """
        # Look at the total attention between the class token,
        # and the image patches
        mask = result[0, 0 , 1 :] # (0 -> batch idx, 0 -> [CLS] token, 1: -> rest tokens)
        
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        return mask
        
    
    def show_mask_on_image(self, img, mask):

        """Do not modify this part"""

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

        """Do not modify this part"""

        # clean previous attention maps and result
        self.clear_attentions()
        
        # get the image name for saving output image2
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

    """Do not modify this part"""

    # arg parsing
    parser = argparse.ArgumentParser(description='Process an image for attention visualization.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    args = parser.parse_args()

    # Execution
    model = AttentionRollout()
    with torch.no_grad():
        outputs = model.run(args.image)