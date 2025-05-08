# from picamera import PiCamera
from ultralytics import YOLO
import cv2

# Initialize the camera
# camera = PiCamera()

# Capture an image
# camera.capture('image.jpg')

# Close the camera
# camera.close()

# Load a pretrained model
model = YOLO('yolov8n.pt')

# Load the captured image
image = cv2.imread('image.png')

# Perform object detection
results = model(image)

# Display the results
#for result in results:
#    result.show()