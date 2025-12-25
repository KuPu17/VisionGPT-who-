import os

# Base Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
CAPTURE_DIR = os.path.join(OUTPUT_DIR, 'captures')

# Create dirs if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CAPTURE_DIR, exist_ok=True)

# Model Settings
# Use 'yolov8n.pt' for faster CPU performance, 'yolov8l.pt' for accuracy
YOLO_MODEL_NAME = 'yolov8l.pt' 
QWEN_MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"

# Device Settings
DEVICE = 'cpu' # Change to 'cuda' if you eventually get a GPU

# Context Builder Settings
ON_THRESHOLD = 0.3
NEAR_THRESHOLD = 150
ALIGNMENT_THRESHOLD = 50