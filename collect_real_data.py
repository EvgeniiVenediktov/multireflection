from mf_control.controller import MFController
import numpy as np
from numpy.typing import ArrayLike
import cv2 
from preprocess_images import process_image_from_webcam

def save_image(img: ArrayLike, x:float, y:float, folder:str="data/real") -> None:
    cv2.imwrite(folder+f"/x{x:.2f}_y{y:.2f}.jpg", img)


if __name__ == "__main__":
    controller = MFController()

    # Set coordinate matrix
    STEP = 0.1

    X_START = -5
    X_STOP = 1.6

    Y_START = -3.5
    Y_STOP = 3

    x_tilt = np.arange(X_START, X_STOP, STEP)  # vertical tilt. Negative - looking down
    y_tilt = np.arange(Y_START, Y_STOP, STEP)  # horizontal tilt. Negative - looking right

    # Go through coordinate matrix
    for x in x_tilt:
        controller.tilt_x(x)
        for y in y_tilt:
            # Turn frame
            controller.tilt_y(y)

            # Get image
            img = controller.capture_image()

            # Process
            img = process_image_from_webcam(img)

            # Save
            save_image(img, x, y)