# Frame settings
X_TILT_START = -2
X_TILT_STOP = 3.7

Y_TILT_START = -2
Y_TILT_STOP = 2

REAL_DATA_COLLECTION_STEP = 0.01
REAL_DATA_COLLECTION_DELAY = 0.1


# Model parameters
INFERENCE_MODEL_FILE_NAME = "trasnfer_002step_clahegradsimple_mix_512bs_0001lr_aug+_best_model.pth"
INFERENCE_MODEL_TYPE = "CLAHEGradSimpleFC"

TRAINING_IMAGE_RESOLUTION = (512, 512)

LMDB_USE_COMPRESSION = False

SIMILARITY_INDEX_THRESHOLD = 0.94
OPTIMUM_IMAGE_PATH_LIST = ["/home/raspberry/projects/multireflection/data/real/x0.00_y0.00.jpg", "/home/raspberry/projects/multireflection/data/light/x0.00_y0.00.jpg", "/home/raspberry/projects/multireflection/data/new/x0.00_y0.00.jpg"]

# EVAL
EVAL_GRID_STEP = 0.5
EVAL_MAX_ADJ_NUMBER = 10
