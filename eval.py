from preprocess_images import process_image_from_webcam
from inference import TiltPredictor, evaluate_position
from mf_control.controller import MFController
import cv2
import numpy as np
from config import *
from color_output import *
import logging
from tqdm import tqdm


logging.basicConfig(filename='eval.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s', 
                    filemode='a')

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

# Generate start position grid
x_vals = list(np.arange(X_TILT_START, X_TILT_STOP+EVAL_GRID_STEP, EVAL_GRID_STEP))
y_vals = list(np.arange(Y_TILT_START, Y_TILT_STOP+EVAL_GRID_STEP, EVAL_GRID_STEP))
print("x_vals:", x_vals)
print("y_vals:", y_vals)
grid = []
for x in x_vals:
    for y in y_vals:
        grid.append((x,y))

for origin_x, origin_y in tqdm(grid, leave=False):
    try:
        # Start at a grid node
        controller.set_tilt_x(origin_x)
        controller.set_tilt_y(origin_y)
        adj_n = 0
        # Autoalignment
        while True:
            if adj_n >= EVAL_MAX_ADJ_NUMBER:
                break

            # Get image from webcam
            img = controller.capture_image()
            img = process_image_from_webcam(img, target_size=(512, 512))

            # Evaluate position
            sim_index = evaluate_position(img, optimums=optimal_states)
            cprint("Similarity index:"+str(sim_index), MAGENTA)
            if sim_index >= SIMILARITY_INDEX_THRESHOLD:
                cprint("Alignment finished", GREEN)
                break

            # Make prediction
            prediction = model.predict(np.array([[img]]))

            # Clip prediction
            x, y = prediction[0]
            x_pred = -clip(x, X_TILT_START, X_TILT_STOP)
            y_pred = -clip(y, Y_TILT_START, Y_TILT_STOP)
            x = round(controller.get_x_tilt() + x_pred, 2)
            y = round(controller.get_y_tilt() + y_pred, 2)
            # cprint("New position:"+str((x, y)), YELLOW)

            # Bring system to proposed positon
            controller.set_tilt_x(x)
            controller.set_tilt_y(y)
            adj_n += 1
            # Log position
            status = f"origin_x:{origin_x},origin_y:{origin_y},adj_n:{adj_n},\
                    pred_x:{x_pred},pred_y:{y_pred},pos_x:{x},pos_y:{y},sim_index:{sim_index}"
            logging.info(status)


    except BaseException as e:
        logging.error(e)
        print(e)
        print("Returning to origin")
        controller.set_tilt_x(0)
        controller.set_tilt_y(0)
        controller.close()
        exit()


# Bring frame to zero
controller.set_tilt_x(0)
controller.set_tilt_y(0)
controller.close()
