import cv2
import sys
import threading
from datetime import datetime

# Local imports
from config import settings
from core.detector import ObjectDetector
from core.vision_utils import ContextBuilder
from core.vlm import VLMInference

def main():
    # 1. Initialize Modules
    try:
        detector = ObjectDetector()
        context_builder = ContextBuilder()
        # Lazy load VLM? Or load immediately? 
        # Loading immediately for stability, though it consumes RAM.
        vlm = VLMInference() 
    except Exception as e:
        print(f"Startup Failed: {e}")
        return

    # 2. Setup Camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Webcam not found.")
        return

    print("\n" + "="*50)
    print(f"VisionGPT Active on {settings.DEVICE.upper()}")
    print("Controls: 'q' to Quit | 's' for Scene Desc | 'c' for Count")
    print("="*50)

    frame_count = 0
    current_context = None
    current_frame = None

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        # A. Detection Phase
        detections, plot_result = detector.detect(frame)
        
        # B. Logic Phase (Spatial Reasoning)
        frame_data = {'detections': detections, 'frame_id': frame_count}
        current_context = context_builder.process_frame(frame_data)
        
        # C. Visualization Phase
        annotated_frame = plot_result.plot()
        cv2.putText(annotated_frame, f"Objects: {len(detections)}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('VisionGPT', annotated_frame)
        
        # Update shared state for the VLM trigger
        current_frame = frame.copy()

        # D. Input Handling
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('s') or key == ord('c'):
            # Determine Prompt
            base_prompt = current_context['vlm_prompt']
            user_q = "Describe the scene." if key == ord('s') else "Count objects and check spatial layout."
            full_prompt = f"{base_prompt}\nQuestion: {user_q}\nAnswer:"
            
            # Save debug capture
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{settings.CAPTURE_DIR}/capture_{timestamp}.jpg"
            cv2.imwrite(save_path, current_frame)
            print(f"\n[Captured frame to {save_path}]")
            
            # Trigger VLM (Blocking call on CPU to prevent crash)
            vlm.ask(current_frame, full_prompt)

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()