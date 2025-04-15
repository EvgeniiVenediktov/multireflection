from preprocess_images import process_image_from_webcam
from inference import TiltPredictor
from mf_control.controller import MFController
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np
from config import *
from color_output import *
import logging

logging.basicConfig(filename='autoalignment.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='a')

def clip(v, minv, maxv):
    v = max(v, minv)
    v = min(v, maxv)
    return v

# Load optimal state image
optimum_image = cv2.imread(OPTIMUM_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)

# Connect Controller
controller = MFController(image_size=(1920, 1440))

# Load model
model = TiltPredictor(INFERENCE_MODEL_FILE_NAME, INFERENCE_MODEL_TYPE)
print(f"Loaded {INFERENCE_MODEL_FILE_NAME}")

try:
    while True:
        # Get image from webcam
        img = controller.capture_image()

        # Display
        img = process_image_from_webcam(img, target_size=(512, 512))
        cv2.imshow("Processed", img)
        cv2.waitKey(2000)
        cv2.destroyWindow("Processed")

        # Evaluate position
        sim_index = round(ssim(optimum_image, img), 2)
        cprint("Similarity index:"+str(sim_index), MAGENTA)
        if sim_index >= SIMILARITY_INDEX_THRESHOLD:
            cprint("Alignment finished", GREEN)
            break

        # Make prediction
        prediction = model.predict(np.array([[img]]))

        # Clip prediction
        x, y = prediction[0]
        x = -clip(x, X_TILT_START, X_TILT_STOP)
        y = -clip(y, Y_TILT_START, Y_TILT_STOP)
        x = round(controller.get_x_tilt() + x, 2)
        y = round(controller.get_y_tilt() + y, 2)
        cprint("New position:"+str((x, y)), YELLOW)

        # Bring system to proposed positon
        controller.set_tilt_x(x)
        controller.set_tilt_y(y)

        # Log position
        logging.info(f"x: {controller.get_x_tilt()}, y: {controller.get_y_tilt()}")

        # TESTING
        #key = input("Type x for exit: ")
        #if key=='x':
        #    break
except BaseException as e:
    logging.error(e)
    print(e)
    print("Returning to origin")
    controller.set_tilt_x(0)
    controller.set_tilt_y(0)
    controller.close()
    exit()

input("Press enter to return system to origin and exit")

# Bring frame to zero
controller.set_tilt_x(0)
controller.set_tilt_y(0)
controller.close()
