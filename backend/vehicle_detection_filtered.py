from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

vehicle_classes = ["car", "motorcycle", "bus", "truck"]

video_path = "../videos/traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error opening video.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]

            if class_name in vehicle_classes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                confidence = float(box.conf[0])

                cv2.rectangle(frame,
                              (x1, y1),
                              (x2, y2),
                              (0,255,0),
                              2)

                cv2.putText(
                    frame,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0,255,0),
                    2
                )

    cv2.imshow("Vehicle Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()