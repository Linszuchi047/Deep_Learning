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

    @staticmethod
    def attention_forward(self, x):
        """

        TODO:
            Implement the attention computation and store the attention maps.
            You need to save the attention map into variable "self.attn_weights"

            Note: Due to @staticmethod, "self" here refers to the "Attention" module instance, not the class itself.

        """

        # write your code here
        self.attn_weights = None # save the attention map into this variable
        pass

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
    
    
    def gradient_rollout(self):
        """

        TODO:
            Define the attention rollout function that accumulate the final attention flows.
            You need to return parameter "mask", which should be the final attention flows.  
            You can follow the below procedures:

            For each attention layers:
                1. perform matrix mutiplication on current attention map and gradient
                2. filter the attention by mean/min/max fiter with respect to 3 attention heads
                3. wipe out all negative values to 0
                4. (optional) normalize the attention map of current layer
                5. perform matrix multiplication on current attention map and previous results

            (6. and 7. are already done for you, you just need to get the variable "result" correctly.)
            6. Obtain the mask: Gather the attention flows that use [CLS] token as query and the rest tokens as keys
            7. Normalize the values inside the mask to [0.0, 1.0]

        """
        result = torch.eye(self.attentions[0].size(-1))
        for attention, grad in zip(self.attentions, self.attention_grads): 
            # Write your code(Gradient Rollout) here to update "result" matrix
            pass
        



        """
        if you have variable "result" in correct shape(shape of attenion map), then this part should work properly.
        """
        # Look at the total attention between the class token,
        # and the image patches
        mask = result[0, 0 , 1 :]
        # In case of 224x224 image, this brings us from 196 to 14
        width = int(mask.size(-1)**0.5)
        mask = mask.reshape(width, width).numpy()
        mask = mask / np.max(mask)
        return mask


    def perform_backpropagation(self, input_tensor, target_category):
        """

        TODO:
            In order to get gradients, you need to do backpropagation mannually.
            You can follow the below procedures:

            1.Inference the model
            2.define a loss simple function that focus on the target category
            2-1. The loss function can be sum of the logits of target category or
            2-2. cross entropy of target category
            3.perform backpropagation with respect to the loss.

            This function will be invoked in "run" function.

        """

        # write your code here to perform backpropagation
        pass
        
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
    
    def run(self, image_path, target_category):

        """Do not modify this part"""

        # clean previous attention maps and result
        self.clear_attentions()
        
        # get the image name for saving output image2
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

    """Do not modify this part"""

    # arg parsing
    parser = argparse.ArgumentParser(description='Process an image for attention visualization with respect to target category.')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image.')
    parser.add_argument('--category', type=int, required=True, help="target category of attention flows")
    args = parser.parse_args()

    # Execution
    model = GradientRollout()
    outputs = model.run(args.image, args.category)