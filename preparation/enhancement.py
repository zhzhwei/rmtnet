import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

affine_transform = transforms.RandomAffine(
    degrees=(-10, 10),         
    translate=(0.1, 0.1),      
    scale=(0.8, 1.2),           
    shear=(-10, 10)             
)

color_jitter = transforms.ColorJitter(
    brightness=0.5,             
    contrast=0.3               
)

mirror_transform = transforms.RandomHorizontalFlip(p=1.0) 

position_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),          
    transforms.RandomAffine(degrees=0,            
                            translate=(0.1, 0.1)) 
])

data_augmentation = transforms.Compose([
    affine_transform,
    color_jitter,
    mirror_transform,
    position_transform,
    transforms.ToTensor()
])

image_path = '1.jpg' 
image = Image.open(image_path)

augmented_image = data_augmentation(image)

augmented_image_pil = transforms.ToPILImage()(augmented_image)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Augmented Image")
plt.imshow(augmented_image_pil)

plt.show()
