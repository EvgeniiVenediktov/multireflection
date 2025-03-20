from mf_control.controller import MFController
import numpy as np
from numpy.typing import ArrayLike
import cv2 
from preprocess_images import process_image_from_webcam
import time

from config import *

def save_image(img: ArrayLike, x:float, y:float, folder:str="data/real") -> None:
    cv2.imwrite(folder+f"/x{x:.2f}_y{y:.2f}.jpg", img)


def whole_process(controller: MFController, x:float, y:float, time_delay=0.3, verbose=True) -> None:
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

    # Log
    if not verbose:
        return
    print(controller.get_frame_position())


if __name__ == "__main__":
    controller = MFController(image_size=(1920, 1440))

    # Set coordinate matrix

    # X_START = X_TILT_START
    # X_STOP = X_TILT_STOP

    # Y_START = Y_TILT_START
    # Y_STOP = Y_TILT_STOP

    x_coord_start = -0.04
    x_coord_stop = 0.04

    y_coord_start = 0
    y_coord_stop = 0.06


    x_tilt = np.arange(x_coord_start, x_coord_stop, REAL_DATA_COLLECTION_STEP)  # horizontal tilt. Negative - looking right
    y_tilt = np.arange(y_coord_start, y_coord_stop, REAL_DATA_COLLECTION_STEP)  # vertical tilt. Negative - looking up

    if len(x_tilt) % 2 == 1:
        x_coord_stop += REAL_DATA_COLLECTION_STEP
        x_tilt = np.arange(x_coord_start, x_coord_stop, REAL_DATA_COLLECTION_STEP)
    

    # Go through coordinate matrix
    i = 0
    while i in range(len(x_tilt)):
        x = x_tilt[i]
        controller.set_tilt_x(x)
        # y: 1 -> 2 -> 3
        for y in y_tilt:
            whole_process(controller, x, y, REAL_DATA_COLLECTION_DELAY)
        i += 1

        x = x_tilt[i]
        controller.set_tilt_x(x)
        # y: 3 -> 2 -> 1
        for y in y_tilt[::-1]:
            whole_process(controller, x, y, REAL_DATA_COLLECTION_DELAY)
        i += 1
    
    # Return frame to (0, 0)
    controller.set_tilt_x(0)
    controller.set_tilt_y(0)
    print(controller.get_frame_position())