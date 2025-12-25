from ultralytics import YOLO
from config import settings

class ObjectDetector:
    def __init__(self):
        print(f"Loading Object Detector ({settings.YOLO_MODEL_NAME})...")
        self.model = YOLO(settings.YOLO_MODEL_NAME)
    
    def detect(self, frame):
        """
        Runs inference and returns a clean list of dictionaries.
        """
        # Run inference (force device from settings)
        results = self.model.predict(frame, verbose=False, device=settings.DEVICE)
        
        detections = []
        for box in results[0].boxes:
            bbox = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            name = self.model.names[cls_id]
            
            detections.append({
                'class_name': name,
                'confidence': conf,
                'bbox': bbox
            })
            
        return detections, results[0] # Return raw result for plotting