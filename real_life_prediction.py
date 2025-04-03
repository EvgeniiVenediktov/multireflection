from preprocess_images import process_image_from_webcam
from inference import TiltPredictor
from mf_control.controller import MFController
import cv2
from config import X_TILT_START, X_TILT_STOP, Y_TILT_START, Y_TILT_STOP, INFERENCE_MODEL_FILE_NAME, INFERENCE_MODEL_TYPE


def clip(v, minv, maxv):
    v = max(v, minv)
    v = min(v, maxv)
    return v

# Connect Controller
controller = MFController()

# Load model
model = TiltPredictor(INFERENCE_MODEL_FILE_NAME, INFERENCE_MODEL_TYPE)

command = "y"
while command == "y":
    # Get image from webcam
    img = controller.capture_image()

    # Display
    cv2.imshow("Raw", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Raw")
    # Preprocess image
    img = process_image_from_webcam(img, target_size=(512, 512))

    # Display
    cv2.imshow("Processed", img)
    cv2.waitKey(0)
    cv2.destroyWindow("Processed")

    # Make prediction
    prediction = model.predict([img])

    # Output
    print("original_prediction:", prediction)

    # Clip prediction
    x, y = prediction
    x = clip(x, X_TILT_START, X_TILT_STOP)
    y = clip(y, Y_TILT_START, Y_TILT_STOP)
    print("clipped prediciton", (x, y))

    if input("Turn mirror by predicted angles? (y/n) ").lower() == "y":
        controller.set_tilt_x(x)
        controller.set_tilt_y(y)

    # Repeat
    command = input("Make another prediction? (y/n) ").lower()


# Bring frame to zero
controller.set_tilt_x(0)
controller.set_tilt_y(0)