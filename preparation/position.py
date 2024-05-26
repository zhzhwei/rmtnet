import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

position_transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),        
    transforms.RandomAffine(degrees=0,             
                            translate=(0.1, 0.1)) 
])

data_augmentation = transforms.Compose([
    position_transform,
    transforms.ToTensor()
])

image_path = 'example_image.jpg'
image = Image.open(image_path)

augmented_image = data_augmentation(image)

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
