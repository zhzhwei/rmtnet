import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

mirror_transform = transforms.RandomHorizontalFlip(p=1.0)

data_augmentation = transforms.Compose([
    mirror_transform,
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
plt.title("Mirrored Image")
plt.imshow(augmented_image_pil)

plt.show()
