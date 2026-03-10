from django.http import StreamingHttpResponse, JsonResponse
import cv2
from .utils import get_detector
from django.shortcuts import render 

def index(request):
    return render(request, 'detection/index.html')

def check_alert(request):
    detector = get_detector()
    return JsonResponse({'alerting': detector.is_alerting})

def gen_frames():
    detector = get_detector()
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process frame with YOLO and distraction logic
            annotated_frame, alert_message = detector.process_frame(frame)
            
            # Encode as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    cap.release()

def video_feed(request):
    return StreamingHttpResponse(gen_frames(), 
                                 content_type='multipart/x-mixed-replace; boundary=frame')
