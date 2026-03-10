import cv2
import time
from ultralytics import YOLO
import os

class DistractionDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.distraction_classes = ['Closed Eye', 'Cigarette', 'Phone']
        self.focus_classes = ['Open Eye', 'Seatbelt']
        
        # State tracking
        self.start_time = None
        self.alert_triggered = False

    def process_frame(self, frame):
        results = self.model(frame, verbose=False)
        result = results[0]
        
        # Get detected classes
        detected_names = [result.names[int(c)] for c in result.boxes.cls]
        
        # Check if ANY distraction is present
        active_distractions = [name for name in detected_names if name in self.distraction_classes]
        distraction_present = len(active_distractions) > 0
        
        current_time = time.time()
        self.alert_triggered = False
        
        if distraction_present:
            if self.start_time is None:
                self.start_time = current_time
            
            duration = current_time - self.start_time
            if duration >= 1.5:
                self.alert_triggered = True
        else:
            self.start_time = None

        # Annotate frame
        annotated_frame = result.plot()
        
        # Add visual alert to frame
        if self.alert_triggered:
            # Draw a prominent red banner
            cv2.rectangle(annotated_frame, (0,0), (annotated_frame.shape[1], 60), (0,0,255), -1)
            cv2.putText(annotated_frame, "WARNING: STAY FOCUSED!", (annotated_frame.shape[1]//2 - 180, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

        return annotated_frame, self.alert_triggered

    @property
    def is_alerting(self):
        return getattr(self, 'alert_triggered', False)

# Singleton-like instance for the web stream
detector = None

def get_detector():
    global detector
    if detector is None:
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best.pt')
        detector = DistractionDetector(model_path)
    return detector
