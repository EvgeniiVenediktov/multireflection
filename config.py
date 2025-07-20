# Frame settings
#X_TILT_START = -2
X_TILT_START = 2.8
X_TILT_STOP = 3.7

Y_TILT_START = -2
Y_TILT_STOP = 2

REAL_DATA_COLLECTION_STEP = 0.01
REAL_DATA_COLLECTION_DELAY = 0.1
DATA_COLLECTION_CVT_TO_GRAYSCALE = True 
DATA_COLLECTION_FOLDER = "data/dark512"
DATA_COLLECTION_FINAL_RESOLUTION = (512, 512)



# Model parameters
INFERENCE_MODEL_FILE_NAME = "001step_SimpleFC_DarkOnly512_lmdb_360bs_0001lr_aug+_best_model.pth"
INFERENCE_MODEL_TYPE = "SimpleFC"

TRAINING_IMAGE_RESOLUTION = (512, 512)

LMDB_USE_COMPRESSION = False

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
