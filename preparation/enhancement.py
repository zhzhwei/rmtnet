import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the affine transformation (from article 11)
affine_transform = transforms.RandomAffine(
    degrees=(-10, 10),          # Rotation between -10 and 10 degrees
    translate=(0.1, 0.1),       # Translation up to 10% in both dimensions
    scale=(0.8, 1.2),           # Scaling between 80% and 120%
    shear=(-10, 10)             # Shearing between -10 and 10 degrees
)

# Define the color jitter transformation (from article 11)
color_jitter = transforms.ColorJitter(
    brightness=0.5,             # Brightness adjustment up to 50%
    contrast=0.3               # Contrast adjustment up to 30%
)

# Define the image mirroring (from article 12)
mirror_transform = transforms.RandomHorizontalFlip(p=1.0)  # 100% probability of flipping

# Define the position transformation (from article 13)
position_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),          # Random rotation up to 15 degrees
    transforms.RandomAffine(degrees=0,              # No additional rotation here
                            translate=(0.1, 0.1))  # Random shift up to 10% of the image size
])

# Compose all the transformations into one data augmentation pipeline
data_augmentation = transforms.Compose([
    affine_transform,
    color_jitter,
    mirror_transform,
    position_transform,
    transforms.ToTensor()
])

# Load an example image
image_path = 'example_image.jpg'  # Provide a path to an example image
image = Image.open(image_path)

# Apply the data augmentation
augmented_image = data_augmentation(image)

# Convert the tensor back to an image for visualization
augmented_image_pil = transforms.ToPILImage()(augmented_image)

# Display the original and augmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Augmented Image")
plt.imshow(augmented_image_pil)

plt.show()
