import cv2
import numpy as np
from numpy.typing import ArrayLike
import os
import time


def to_square(img, t_x, t_y, b_x, b_y):
    h, w = img.shape[:2]
    m_x = w // 2 + b_x
    m_y = h // 2 + b_y
    return img[m_y - t_y // 2 : m_y + t_y // 2, m_x - t_x // 2 : m_x + t_x // 2]


def process_for_stopping_criteria(
    img: ArrayLike, 
    ht:int = 120,    
) -> ArrayLike:
    
    # High-pass filter
    img[img<ht] = 0

    return img


def process_image_from_webcam(
    img: ArrayLike, target_size: tuple[int] = (512, 512)
) -> ArrayLike:
    # TODO
    # To grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Cut off sides to turn into square image
    r = min(*img.shape)
    img = to_square(img, r, r, 0, 0)

    # Shrink image
    img = cv2.resize(img, target_size)

    # Apply circular mask
    img = apply_circular_mask(img)

    # Blur
    img = cv2.GaussianBlur(img, (7, 7), 0)

    return img

def process_image_from_webcam_color(
        img: ArrayLike, target_size: tuple[int] = (256, 256)
) -> ArrayLike:
    assert(len(img.shape) == 3)
    
    # Cut off sides to turn into square image
    r = min(*img.shape[:2])
    img = to_square(img, r, r, 0, 0)

    # Shrink image
    img = cv2.resize(img, target_size)

    # Apply circular mask
    img = apply_circular_mask(img)

    # Blur
    img = cv2.GaussianBlur(img, (7, 7), 0)

    return img

def apply_circular_mask(image_array) -> ArrayLike:
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
    image_name: str,
    image_array: ArrayLike,
    target_size=(125, 125),
    use_circular_mask=True,
):
    """
    Applies the mask and saves the image.

    Returns image_name:str , masked_img:ArrayLike
    """
    # Resize image
    image_array = cv2.resize(image_array, target_size)
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
    folder: str, grayscale=True, verbose=False, target_size=(125, 125)
) -> dict[str, np.ndarray]:
    t_start = time.time()

    # Load all pictures in memory
    images = __load_images(folder, grayscale=grayscale)

    # Dict for processed imgs
    processed_images = {}

    # Process sequentially
    for name, img in images.items():
        processed_images[name] = process_single_image(name, img, target_size)[1]

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
