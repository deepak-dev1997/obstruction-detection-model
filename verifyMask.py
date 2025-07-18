import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

def visualize_masks(image_folder, mask_folder, num_samples=30):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.png')])

    for img_file, mask_file in zip(image_files, mask_files):
        if num_samples <= 0:
            break

        img_path = os.path.join(image_folder, img_file)
        mask_path = os.path.join(mask_folder, mask_file)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Error loading {img_file} or {mask_file}. Skipping.")
            continue

        # Convert BGR to RGB for display
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Display
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='jet', vmin=0, vmax=max(np.unique(mask)))
        plt.title('Mask')
        plt.axis('off')

        plt.show()

        num_samples -= 1

if __name__ == "__main__":
    image_folder = './trainimages/'
    mask_folder = './maskedimages/'
    visualize_masks(image_folder, mask_folder, num_samples=3)