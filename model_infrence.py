import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO inference
    results = model(frame)

    # Annotate frame
    annotated_frame = results[0].plot()

    # Show frame
    cv2.imshow("Driver Monitoring System", annotated_frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()