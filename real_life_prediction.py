from preprocess_images import process_image_from_webcam
from inference import TiltPredictor, evaluate_position
from mf_control.controller import MFController
import cv2
import numpy as np
from config import *
from color_output import *

def clip(v, minv, maxv):
    v = max(v, minv)
    v = min(v, maxv)
    return v

# Load optimal state images
optimal_states = []
for path in OPTIMUM_IMAGE_PATH_LIST:
    optimal_states.append(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

# Connect Controller
controller = MFController(image_size=(1920, 1440))

# Load model
model = TiltPredictor(INFERENCE_MODEL_FILE_NAME, INFERENCE_MODEL_TYPE)
print(f"Loaded {INFERENCE_MODEL_FILE_NAME}")

command = "y"
while command == "y":
    # Get image from webcam
    img = controller.capture_image()

    # Display
    # cv2.imshow("Raw", img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("Raw")
    # Preprocess image
    img = process_image_from_webcam(img, target_size=(512, 512))
    
    # Display
    cv2.imshow("Processed", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Processed")

    # Evaluate position
    sim_index = evaluate_position(img, optimums=optimal_states)
    cprint("Similarity index:"+str(round(sim_index, 2)), MAGENTA)

    # Make prediction
    prediction = model.predict(np.array([[img]]))

    # Output
    cprint("prediction:"+str(prediction[0]), GRAY)

    # Clip prediction
    x, y = prediction[0]
    x = -clip(x, X_TILT_START, X_TILT_STOP)
    y = -clip(y, Y_TILT_START, Y_TILT_STOP)
    x = round(controller.get_x_tilt() + x, 2)
    y = round(controller.get_y_tilt() + y, 2)
    cprint("Proposed position:"+str((x, y)), GREEN)

    if input("Bring system to proposed positon? (y/n) ").lower() == "y":
        controller.set_tilt_x(x)
        controller.set_tilt_y(y)

    # Repeat
    command = input("Make another prediction? (y/n) ").lower()


# Bring frame to zero
controller.set_tilt_x(0)
controller.set_tilt_y(0)
controller.close()
