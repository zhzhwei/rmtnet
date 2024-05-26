import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the affine transformation
affine_transform = transforms.RandomAffine(
    degrees=(-10, 10),          
    translate=(0.1, 0.1),       
    scale=(0.8, 1.2),          
    shear=(-10, 10)            
)

# Define the color jitter transformation
color_jitter = transforms.ColorJitter(
    brightness=0.5,            
    contrast=0.3               
)

data_augmentation = transforms.Compose([
    affine_transform,
    color_jitter,
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
