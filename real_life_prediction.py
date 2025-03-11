from preprocess_images import process_image_from_webcam
from inference import TiltPredictor
from mf_control.controller import MFController

# Connect Controller
controller = MFController()

# Load model
model = TiltPredictor("fc_4layers_1024batch_500epochs_50cosinescheduler_best_model.pth")

command = "y"
while command == "y":
    # Get image from webcam
    img = controller.capture_image()

    # Preprocess image
    img = process_image_from_webcam(img)

    # Make prediction
    prediction = model.predict([img])

    # Output
    print("prediction:",prediction)

    # Repeat
    command = input("Make another prediction? (y)").lower()
