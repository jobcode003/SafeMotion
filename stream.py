from flask import Flask, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load your trained model
model = YOLO("best.pt")

# Connect to rpicam stream
cap = cv2.VideoCapture("tcp://0.0.0.0:8888")
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
	_, buffer = cv2.imencode('.jpg', frame, encode_param)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


        # Run YOLO inference
        results = model(frame)

        # Draw detections
        annotated_frame = results[0].plot()

        # Compress (important for speed)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
        _, buffer = cv2.imencode('.jpg', annotated_frame, encode_param)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
>>>>>>> 371746031f53eba0d524eabe0990122c548db9ff

@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5000)
