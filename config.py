# Frame settings
X_TILT_START = -2
X_TILT_STOP = 3.7

Y_TILT_START = -2
Y_TILT_STOP = 2

REAL_DATA_COLLECTION_STEP = 0.01
REAL_DATA_COLLECTION_DELAY = 0.1

# X/Y_TILT_START/STOP above define the system tilt range - the physical actuation limits
# used for inference de-normalization and the closed-loop alignment/eval code.
# The data-collection range below defaults to the system range and can be overridden
# to sweep a different range when collecting a dataset.
DATA_COLLECTION_X_TILT_STOP = X_TILT_STOP
# DATA_COLLECTION_X_TILT_STOP = 2
DATA_COLLECTION_Y_TILT_STOP = Y_TILT_STOP

DATA_COLLECTION_CVT_TO_GRAYSCALE = True 
DATA_COLLECTION_FOLDER = "data/dark512"

DATA_COLLECTION_FINAL_RESOLUTION = (512, 512)



# Model parameters
# INFERENCE_MODEL_FILE_NAME = "real/001step_SimpleFC_DarkOnly512_lmdb_360bs_0001lr_aug+_best_model.pth"
INFERENCE_MODEL_FILE_NAME = "real/resnet18_001step_BS_avid-sweep-4406_DarkOnly512_lmdb_50bs_0001lr_aug+_best_model.pth"
INFERENCE_MODEL_TYPE = "SimpleFC"

TRAINING_IMAGE_RESOLUTION = (512, 512)

# Square model input size in px for TiltPredictor (ResNet18). None: parsed from a checkpoint
# named r<res>_... (train/TRAINING_AND_EVALS.md), else TRAINING_IMAGE_RESOLUTION. The webcam
# frame is still processed to DATA_COLLECTION_FINAL_RESOLUTION for the SSIM test; the
# predictor area-resizes it to this size before the model.
INFERENCE_INPUT_RESOLUTION = None

SIMILARITY_INDEX_THRESHOLD = 0.95
OPTIMUM_IMAGE_PATH_LIST = [#"/home/raspberry/projects/multireflection/data/color_dark_004/x0.00_y0.00.jpg", 
                           #"/home/raspberry/projects/multireflection/data/color_mainlight_004/x0.00_y0.00.jpg", 
                           #"/home/raspberry/projects/multireflection/data/color/x0.00_y0.00.jpg", 
                           "/home/raspberry/projects/multireflection/data/dark512/x0.00_y0.00.jpg", 


                        #    "/home/raspberry/projects/multireflection/data/real/x0.00_y0.00.jpg", 
                        #    "/home/raspberry/projects/multireflection/data/light/x0.00_y0.00.jpg", 
                        #    "/home/raspberry/projects/multireflection/data/new/x0.00_y0.00.jpg"
                           ]

# EVAL
EVAL_GRID_STEP = 0.5
EVAL_MAX_ADJ_NUMBER = 10
