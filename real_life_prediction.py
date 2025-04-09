from preprocess_images import process_image_from_webcam
from inference import TiltPredictor
from mf_control.controller import MFController
import cv2
import numpy as np
from config import X_TILT_START, X_TILT_STOP, Y_TILT_START, Y_TILT_STOP, INFERENCE_MODEL_FILE_NAME, INFERENCE_MODEL_TYPE
from color_output import *

def clip(v, minv, maxv):
    v = max(v, minv)
    v = min(v, maxv)
    return v

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

    # Make prediction
    prediction = model.predict(np.array([[img]]))

    # Output
    print("prediction:", prediction[0])

    # Clip prediction
    x, y = prediction[0]
    x = -round(clip(x, X_TILT_START, X_TILT_STOP), 2)
    y = -round(clip(y, Y_TILT_START, Y_TILT_STOP), 2)
    cprint("Proposed turn:"+str((x, y)), GREEN)

    if input("Turn mirror by predicted angles? (y/n) ").lower() == "y":
        controller.set_tilt_x(x)
        controller.set_tilt_y(y)

    # Repeat
    command = input("Make another prediction? (y/n) ").lower()


# Bring frame to zero
controller.set_tilt_x(0)
controller.set_tilt_y(0)
controller.close()
