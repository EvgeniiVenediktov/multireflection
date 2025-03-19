from mf_control.controller import MFController
import numpy as np
from numpy.typing import ArrayLike
import cv2 
from preprocess_images import process_image_from_webcam
import time

def save_image(img: ArrayLike, x:float, y:float, folder:str="data/real") -> None:
    cv2.imwrite(folder+f"/x{x:.2f}_y{y:.2f}.jpg", img)


def whole_process(controller, x, y):
    # Turn frame
    controller.set_tilt_y(y)

    # Get image
    img = controller.capture_image()

    # Process
    img = process_image_from_webcam(img, target_size=(512, 512))

    # Save
    save_image(img, x, y)
    time.sleep(0.5)
    print(controller.get_frame_position())


if __name__ == "__main__":
    controller = MFController(image_size=(1920, 1440))

    # Set coordinate matrix
    STEP = 0.02

    # X_START = -2
    # X_STOP = 3.7

    # Y_START = -2
    # Y_STOP = 2

    X_START = -0.04
    X_STOP = 0.04

    Y_START = 0
    Y_STOP = 0.06


    x_tilt = np.arange(X_START, X_STOP, STEP)  # vertical tilt. Negative - looking down
    y_tilt = np.arange(Y_START, Y_STOP, STEP)  # horizontal tilt. Negative - looking right

    print("len(x_tilt):", len(x_tilt))
    

    # Go through coordinate matrix
    i = 0



    while i in range(len(x_tilt)):
        x = x_tilt[i]
        controller.set_tilt_x(x)
        for y in y_tilt:
            whole_process(controller, x, y)

        i += 1
        x = x_tilt[i]
        controller.set_tilt_x(x)
        for y in y_tilt[::-1]:
            whole_process(controller, x, y)
        i += 1
    
    controller.set_tilt_x(0)
    controller.set_tilt_y(0)
    print(controller.get_frame_position())