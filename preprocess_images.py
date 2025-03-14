import cv2
import numpy as np
from numpy.typing import ArrayLike
import os
import time


def process_image_from_webcam(
    img: ArrayLike, target_size: tuple[int] = (125, 125)
) -> ArrayLike:
    # TODO
    # To grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cut off sides to turn into square image
    h, w = img.shape
    m = w // 2
    new_left_idx = m - h // 2
    new_right_idx = m + h // 2
    img = img[:, new_left_idx:new_right_idx]

    # Shrink image
    img = cv2.resize(img, target_size)

    return img


def apply_circular_mask(image_array):
    """Applies a circular mask that blackens everything outside a centered circle."""
    h, w = image_array.shape[:2]
    radius = min(w // 2, h // 2)  # Radius is half of the short side
    center = (w // 2, h // 2)

    # Create a black mask with a white circle in the center
    mask = np.zeros((h, w), dtype=np.uint8)
    mask = cv2.circle(mask, center, radius, 255, -1)

    result = np.zeros_like(image_array)
    result[mask == 255] = image_array[mask == 255]

    return result


def process_single_image(
    image_name, image_array, target_square_size=125, use_circular_mask=True
):
    """Applies the mask and saves the image."""
    # Resize image

    # Apply circular mask
    masked_img = apply_circular_mask(image_array)
    return image_name, masked_img


def __load_images(input_folder, grayscale=False):
    """Loads all images into RAM."""
    images = {}
    for file in os.listdir(input_folder):
        if file.endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(input_folder, file)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if not grayscale:
                img = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if img is not None:
                images[file] = img
    return images


def data_from_folder(
    folder: str, grayscale=True, verbose=False
) -> dict[str, np.ndarray]:
    t_start = time.time()

    # Load all pictures in memory
    images = __load_images(folder, grayscale=grayscale)

    # Dict for processed imgs
    processed_images = {}

    # Process sequentially
    for name, img in images.items():
        processed_images[name] = process_single_image(name, img)[1]

    if verbose:
        print(
            f"Processing completed. Loaded #{len(processed_images)} images. Elapsed time: {time.time()-t_start}"
        )

    return processed_images


if __name__ == "__main__":
    data = data_from_folder("./data/125x125_laser_x4_y6", grayscale=True, verbose=True)

    for _, v in data.items():
        print(v.shape)
        break
