from ultralytics import YOLO
import cv2

model = YOLO("yolo26s-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, verbose=False)
    annotated = results[0].plot()

    cv2.imshow("Pose Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()