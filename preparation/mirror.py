import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the image mirroring (flipping) transformation
mirror_transform = transforms.RandomHorizontalFlip(p=1.0)  # 100% probability of flipping

# Compose the transformations (if you have other transformations, you can add them here)
data_augmentation = transforms.Compose([
    mirror_transform,
    transforms.ToTensor()
])

# Load an example image
image_path = 'example_image.jpg'  # Provide a path to an example image
image = Image.open(image_path)

# Apply the data augmentation (image mirroring)
augmented_image = data_augmentation(image)

# Convert the tensor back to an image for visualization
augmented_image_pil = transforms.ToPILImage()(augmented_image)

# Display the original and augmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Mirrored Image")
plt.imshow(augmented_image_pil)

plt.show()
