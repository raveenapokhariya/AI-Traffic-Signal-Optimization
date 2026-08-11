from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

vehicle_classes = {
    1: "motorcycle",
    2: "car",
    5: "bus",
    7: "truck"
}


video_path = "../videos/traffic.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

tracked_vehicles = set()

vehicle_counts = {
    "car": 0,
    "motorcycle": 0,
    "bus": 0,
    "truck": 0
}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        classes=list(vehicle_classes.keys()),
        verbose=False
    )

    if results[0].boxes.id is not None:

        boxes = results[0].boxes

        track_ids = boxes.id.int().cpu().tolist()
        class_ids = boxes.cls.int().cpu().tolist()

        for track_id, class_id in zip(track_ids, class_ids):

            vehicle_name = vehicle_classes[class_id]

            if track_id not in tracked_vehicles:

                tracked_vehicles.add(track_id)

                vehicle_counts[vehicle_name] += 1

    annotated_frame = results[0].plot()

    y = 40

    for vehicle, count in vehicle_counts.items():

        text = f"{vehicle}: {count}"

        cv2.putText(
            annotated_frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        y += 35

    total = sum(vehicle_counts.values())

    cv2.putText(
        annotated_frame,
        f"Total Vehicles: {total}",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.imshow("Vehicle Counter", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nFinal Vehicle Count:")
print("--------------------")

for vehicle, count in vehicle_counts.items():
    print(f"{vehicle}: {count}")

print(f"Total Vehicles: {sum(vehicle_counts.values())}")