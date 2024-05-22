import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the affine transformation
affine_transform = transforms.RandomAffine(
    degrees=(-10, 10),          # Rotation between -10 and 10 degrees
    translate=(0.1, 0.1),       # Translation up to 10% in both dimensions
    scale=(0.8, 1.2),           # Scaling between 80% and 120%
    shear=(-10, 10)             # Shearing between -10 and 10 degrees
)

# Define the color jitter transformation
color_jitter = transforms.ColorJitter(
    brightness=0.5,             # Brightness adjustment up to 50%
    contrast=0.3               # Contrast adjustment up to 30%
)

# Compose the transformations
data_augmentation = transforms.Compose([
    affine_transform,
    color_jitter,
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
