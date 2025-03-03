from mf_control.controller import MFController
import numpy as np
from numpy.typing import ArrayLike


def process_image(img: ArrayLike) -> ArrayLike:
    # TODO 
    return NotImplementedError

controller = MFController()

# Set coordinate matrix
STEP = 0.1

X_START = -5
X_STOP = 1.6

Y_START = -3.5
Y_STOP = 3

x_tilt = np.arange(-5, 1.6, STEP)  # vertical tilt. Negative - looking down
y_tilt = np.arange(-3.5, 3, STEP)  # horizontal tilt. Negative - looking right

x_pos = 0
y_pos = 0

# Bring frame to the starting point
controller.tilt_x(X_START)
controller.tilt_y(Y_START)

# Go through coordinate matrix
for _ in x_tilt:
    for _ in y_tilt:
        
        y_pos += STEP
    x_pos += STEP