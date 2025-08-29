from ultralytics import YOLO
import cv2

# Load trained model
model = YOLO("best_glove_model.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Class color mapping (0 = gloves, 1 = no-gloves) 
CLASS_COLORS = {
    "gloved_hand": (0, 255, 0),      # Green (BGR in OpenCV)
    "bare_hand": (0, 0, 255)    # Red
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame, conf=0.5, verbose=False)

    for result in results:
        for box in result.boxes:
            # Get bounding box coords
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Get class id and name
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            conf = float(box.conf[0])

            # Pick color based on label
            color = CLASS_COLORS.get(label, (255, 255, 255))  # default white

            # Draw rectangle + label
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame, f"{label} {conf:.2f}",
                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )

    # Show frame
    cv2.imshow("Glove Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
