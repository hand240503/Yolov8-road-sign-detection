import random
import cv2
from ultralytics import YOLO

def load_class_names(path):
    with open(path, "r") as f:
        return f.read().strip().split("\n")

def generate_colors(num_colors):
    colors = []
    for _ in range(num_colors):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        colors.append((b, g, r))  # OpenCV dùng BGR
    return colors

def draw_detection_boxes(frame, boxes, class_list, colors):
    labels_detected = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            clsID = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]
            bb = box.xyxy.numpy()[0]

            # Vẽ khung
            cv2.rectangle(frame, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), colors[clsID], 3)

            # Vẽ text class + confidence (màu xanh lá)
            label_text = f"{class_list[clsID]} {conf*100:.2f}%"
            cv2.putText(frame, label_text, (int(bb[0]), int(bb[1]) - 10),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            labels_detected.append(class_list[clsID].lower())
    return labels_detected


def main():
    class_file_path = "D:/VKU/ky_6/Chuyen_De/yolov8_road_sign_recognition/road_signs.txt"
    model_path = "D:/VKU/ky_6/Chuyen_De/yolov8_road_sign_recognition/best.pt"

    class_list = load_class_names(class_file_path)
    detection_colors = generate_colors(len(class_list))
    model = YOLO(model_path)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera")
        return

    print("Nhấn 'q' để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được frame. Thoát.")
            break

        results = model.predict(source=[frame], conf=0.45, save=False)
        boxes = results[0].boxes

        labels_detected = draw_detection_boxes(frame, boxes, class_list, detection_colors)

        # Hiển thị cảnh báo nếu có biển STOP
        if "stop" in labels_detected:
            warning_text = "STOP!"
            cv2.putText(frame, warning_text, (50, 80), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 4)

        cv2.imshow("Road Sign Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
