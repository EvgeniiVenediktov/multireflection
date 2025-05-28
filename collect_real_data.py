from mf_control.controller import MFController
import numpy as np
from numpy.typing import ArrayLike
import cv2
from preprocess_images import process_image_from_webcam
import time
from datetime import datetime
from config import *
import logging
from tqdm import tqdm

logging.basicConfig(filename="data_collection_log.txt",
                    filemode='a',
                    format='%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.DEBUG)


def save_image(img: ArrayLike, x: float, y: float, folder: str = "data/main_light") -> None:
    cv2.imwrite(folder + f"/x{x:.2f}_y{y:.2f}.jpg", img)


def whole_process(controller: MFController, x:float, y:float, pbar:tqdm, i, i_m, time_delay=0.1, log=True) -> None:
    # Turn frame
    controller.set_tilt_y(y)

    # Wait for system to set
    time.sleep(time_delay)

    # Get image
    img = controller.capture_image()

    # Process
    img = process_image_from_webcam(img, target_size=(512, 512))

    # Save
    save_image(img, x, y)

    # Pbar
    pbar.set_description(f"X:{i}/{i_m}, Y:{y}")

    # Log
    if not log:
        return

    logging.info(controller.get_frame_position())


if __name__ == "__main__":
    controller = MFController(image_size=(1920, 1440))

    # Set coordinate matrix
    x_coord_start = X_TILT_START
    x_coord_stop = X_TILT_STOP

    y_coord_start = Y_TILT_START
    y_coord_stop = Y_TILT_STOP



    x_tilt = np.arange(x_coord_start, x_coord_stop, REAL_DATA_COLLECTION_STEP)  # horizontal tilt. Negative - looking right
    y_tilt = np.arange(y_coord_start, y_coord_stop, REAL_DATA_COLLECTION_STEP)  # vertical tilt. Negative - looking up

    if len(x_tilt) % 2 == 1:
        x_coord_stop += REAL_DATA_COLLECTION_STEP
        x_tilt = np.arange(x_coord_start, x_coord_stop, REAL_DATA_COLLECTION_STEP)
    

    # Go through coordinate matrix
    controller.start()
    i = 0
    i_m = len(x_tilt)
    try:
        while i < i_m:
            x = x_tilt[i]
            controller.set_tilt_x(x)
            # y: 1 -> 2 -> 3
            pbar = tqdm(y_tilt, leave=False)
            for y in pbar:
                whole_process(controller, x, y, pbar, i, i_m, time_delay=REAL_DATA_COLLECTION_DELAY)
            i += 1

            x = x_tilt[i]
            controller.set_tilt_x(x)
            # y: 3 -> 2 -> 1
            pbar = tqdm(y_tilt[::-1], leave=False)
            for y in pbar:
                whole_process(controller, x, y, pbar, i, i_m, time_delay=REAL_DATA_COLLECTION_DELAY)
            i += 1
    finally:
        
        # Return frame to (0, 0)
        controller.set_tilt_x(0)
        controller.set_tilt_y(0)
        print(controller.get_frame_position())
        controller.close()
