import cv2
import os
import json
import argparse
from ultralytics import YOLO

# CLI arguments
def get_args():
    parser = argparse.ArgumentParser(description="Glove Detection Script")

    parser.add_argument("--input", type=str, default="input/",
                        help="Path to folder with input images")
    parser.add_argument("--output", type=str, default="output/",
                        help="Path to folder where annotated images will be saved")
    parser.add_argument("--logs", type=str, default="logs/",
                        help="Path to folder where detection logs will be saved")
    parser.add_argument("--confidence", type=float, default=0.5,
                        help="Confidence threshold for detections (0-1)")
    parser.add_argument("--model", type=str, default="runs/detect/train/weights/best.pt",
                        help="Path to trained YOLO model")

    return parser.parse_args()


def main():
    args = get_args()

    input_folder = args.input
    output_folder = args.output
    logs_folder = args.logs
    conf_threshold = args.confidence
    model_path = args.model

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(logs_folder, exist_ok=True)

    # Load trained model
    model = YOLO("best_glove_model.pt")

    for img_name in os.listdir(input_folder):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)

        results = model.predict(source=img_path, conf=conf_threshold, save=False)

        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Color coding: green = gloves, red = bare hand
                color = (0, 255, 0) if label == "gloved_hand" else (0, 0, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                detections.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2]
                })

        # Save annotated image
        out_path = os.path.join(output_folder, img_name)
        cv2.imwrite(out_path, img)

        # Save detections log
        log_data = {"filename": img_name, "detections": detections}
        log_path = os.path.join(logs_folder, img_name.replace(".jpg", ".json").replace(".png", ".json"))
        with open(log_path, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"Processed {img_name}")


if __name__ == "__main__":
    main()
