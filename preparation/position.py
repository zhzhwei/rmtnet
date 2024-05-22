import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define the position transformation
position_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),          # Random rotation up to 15 degrees
    transforms.RandomAffine(degrees=0,              # No additional rotation here
                            translate=(0.1, 0.1))  # Random shift up to 10% of the image size
])

# Compose the transformations (if you have other transformations, you can add them here)
data_augmentation = transforms.Compose([
    position_transform,
    transforms.ToTensor()
])

# Load an example image
image_path = 'example_image.jpg'  # Provide a path to an example image
image = Image.open(image_path)

# Apply the data augmentation (position transformations)
augmented_image = data_augmentation(image)

# Convert the tensor back to an image for visualization
augmented_image_pil = transforms.ToPILImage()(augmented_image)

# Display the original and augmented images
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)

plt.subplot(1, 2, 2)
plt.title("Transformed Image")
plt.imshow(augmented_image_pil)

plt.show()
