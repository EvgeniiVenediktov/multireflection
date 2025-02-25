import cv2
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from queue import Queue


def apply_circular_mask(image_array):
    """Applies a circular mask that blackens everything outside a centered circle."""
    h, w = image_array.shape[:2]
    radius = w // 2  # Radius is half of the width
    center = (w // 2, h // 2)

    # Create a black mask with a white circle in the center
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.circle(mask, center, radius, 255, -1)

    result = np.zeros_like(image_array)
    result[mask == 255] = image_array[mask == 255]

    return result


def process_single_image(image_name, image_array):
    """Applies the mask and saves the image."""
    # TODO: Optionally resize image
    masked_img = apply_circular_mask(image_array)
    return image_name, masked_img


def __load_images(input_folder):
    """Loads all images into RAM."""
    images = {}
    for file in os.listdir(input_folder):
        # print(file)
        if file.endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                images[file] = img
    return images


def data_from_folder(folder: str) -> dict[str, np.ndarray]:

    # Load all pictures in memory
    images = __load_images(folder)

    # Dict for processed imgs
    processed_images = {}

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_image, name, img): name for name, img in images.items()}
        for future in futures:
            name, processed_img = future.result()
            processed_images[name] = processed_img  # Store in-memory

    return processed_images


if __name__ == "__main__":
    data = data_from_folder("./data/test")

