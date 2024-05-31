import os
import cv2
import torch
import warnings
from captum.attr import Saliency, LayerGradCam

from matplotlib import pyplot as plt
from model.RMTNET import RMTNET
from albumentations import (
    HorizontalFlip, VerticalFlip, ShiftScaleRotate, RandomBrightnessContrast, Compose
)

warnings.filterwarnings("ignore")

def get_transforms():
    return Compose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),
        RandomBrightnessContrast(p=0.5)
    ])

transforms = get_transforms()

def get_last_conv_layer(model):
    """
    Recursively find the last convolutional layer in the model.
    """
    # Get the model's children list
    children = list(model.children())

    # If the model has no children, this is a leaf node
    if len(children) == 0:
        if isinstance(model, torch.nn.Conv2d):
            return model
        else:
            return None

    # If the model has children, find the last convolutional layer in the reversed children list
    for child in reversed(children):
        last_conv_layer = get_last_conv_layer(child)
        if last_conv_layer is not None:
            return last_conv_layer

    # If no convolutional layer is found, return None
    return None

if __name__ == "__main__":
    
    model = RMTNET(num_classes=4)
    checkpoint = torch.load('./weights/resnet_18_minor_channels_0.8670255584879595.pth')
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    target_class = 1
    saliency_output = "./saliency_" + str(target_class)
    if target_class == 1:
        slice_paths_root = "../abdomen/dataset/images/batch5/1.2.840.113619.2.359.3.2831208971.107.1572999609.106/1.2.840.113619.2.359.3.2831208971.107.1572999609.204.4202496"
    elif target_class == 2:
        slice_paths_root = "../abdomen/dataset/images/batch5/1.2.840.113619.2.359.3.2831208971.107.1572999609.106/1.2.840.113619.2.359.3.2831208971.107.1572999609.204.4198400"
    os.makedirs(saliency_output, exist_ok=True)
    
    slice_paths = [f"{slice_paths_root}/{slice}" for slice in os.listdir(slice_paths_root)]
    slice_paths = sorted(slice_paths, key=lambda x: int(x.split('.')[-2]))
    slice_paths = slice_paths[:20]
    
    # Get the last convolutional layer in the model
    last_conv_layer = get_last_conv_layer(model)
    
    for slice_path in slice_paths:
        slice = cv2.imread(slice_path)
        slice = cv2.resize(slice, (256,256), interpolation=cv2.INTER_AREA)
        slice = transforms(image=slice)['image']
        slice = slice / 255.0
        
        slice = torch.from_numpy(slice.transpose(2, 0, 1)).float().unsqueeze(0)
        slice.requires_grad_()
        
        grad_cam = LayerGradCam(model, last_conv_layer)
        cam = grad_cam.attribute(slice, target=target_class)
        cam = cam.squeeze().detach().numpy()
        cam = cv2.resize(cam, (256,256))
        
        plt.imshow(cam, cmap='jet')
        plt.axis('off') 
        plt.savefig(f"{saliency_output}/{slice_path.split('/')[-1]}", bbox_inches='tight', pad_inches=0)
        plt.close()