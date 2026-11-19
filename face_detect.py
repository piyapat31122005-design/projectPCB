"""
Real-time face recognition ด้วย OpenCV (Haar Cascade + LBPH)

โครงสร้างโฟลเดอร์รูปสำหรับ train:

    dataset/
    ├── Alice/
    │   ├── 1.jpg
    │   └── 2.jpg
    └── Bob/
        └── 1.jpg

ชื่อโฟลเดอร์ย่อย = ชื่อที่จะแสดงบนหน้าจอ
"""

import argparse
import sys
from pathlib import Path

import cv2 as cv
import numpy as np

FACE_SIZE = (150, 150)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
WINDOW_NAME = "LBPH Face Recognition"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face recognition แบบ real-time ด้วย OpenCV LBPH"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("dataset"),
        help="โฟลเดอร์รูปสำหรับ train (default: dataset)",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="index ของกล้อง (default: 0)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=80.0,
        help="ค่า confidence สูงสุดที่ยอมรับว่าเป็นคนนั้น "
        "(ยิ่งต่ำยิ่งเข้มงวด, default: 80)",
    )
    parser.add_argument(
        "--cascade",
        type=Path,
        default=None,
        help="path ของไฟล์ Haar Cascade (ถ้าไม่ระบุจะใช้ของที่มากับ OpenCV)",
    )
    return parser.parse_args()


def load_face_detector(cascade_path=None):
    path = str(cascade_path) if cascade_path else (
        cv.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    detector = cv.CascadeClassifier(path)
    if detector.empty():
        sys.exit(f"โหลด cascade ไม่สำเร็จ: {path}")
    return detector


def detect_faces(detector, gray):
    return detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )


def preprocess_face(gray, box):
    x, y, w, h = box
    face = cv.resize(gray[y:y + h, x:x + w], FACE_SIZE)
    return cv.equalizeHist(face)  # ลดผลของแสงที่ต่างกัน


def load_training_data(dataset_dir, detector):
    """อ่านรูปจาก dataset/<ชื่อ>/*.jpg แล้วคืน (faces, labels, label_names)"""
    if not dataset_dir.is_dir():
        sys.exit(
            f"ไม่พบโฟลเดอร์ '{dataset_dir}' "
            "กรุณาสร้าง dataset/<ชื่อคน>/ แล้วใส่รูปก่อน (ดู README.md)"
        )

    faces, labels, label_names = [], [], {}
    person_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())

    for label_id, person_dir in enumerate(person_dirs):
        label_names[label_id] = person_dir.name
        count = 0
        for img_path in sorted(person_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            gray = cv.imread(str(img_path), cv.IMREAD_GRAYSCALE)
            if gray is None:
                print(f"[ข้าม] อ่านรูปไม่ได้: {img_path}")
                continue
            boxes = detect_faces(detector, gray)
            if len(boxes) == 0:
                print(f"[ข้าม] ไม่พบใบหน้า: {img_path}")
                continue
            # ใช้เฉพาะใบหน้าที่ใหญ่ที่สุดในรูป กันหน้าคนอื่นในพื้นหลังปนเข้า
            box = max(boxes, key=lambda b: b[2] * b[3])
            faces.append(preprocess_face(gray, box))
            labels.append(label_id)
            count += 1
        print(f"{person_dir.name}: ใช้ train {count} รูป")

    return faces, labels, label_names


def train_recognizer(faces, labels):
    if not hasattr(cv, "face"):
        sys.exit(
            "ไม่พบโมดูล cv.face กรุณาติดตั้ง opencv-contrib-python "
            "(pip install opencv-contrib-python)"
        )
    recognizer = cv.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    return recognizer


def run_camera(camera_index, detector, recognizer, label_names, threshold):
    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        sys.exit(f"เปิดกล้อง index {camera_index} ไม่ได้")

    print("กำลังเปิดกล้อง... กด 'q' เพื่อปิดโปรแกรม")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("อ่านภาพจากกล้องไม่ได้")
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            for box in detect_faces(detector, gray):
                x, y, w, h = box
                label_id, confidence = recognizer.predict(
                    preprocess_face(gray, box)
                )

                if confidence < threshold:
                    name, color = label_names[label_id], (0, 255, 0)
                else:
                    name, color = "Unknown", (0, 0, 255)

                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv.putText(
                    frame,
                    f"{name} ({confidence:.1f})",
                    (x, max(y - 10, 20)),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2,
                )

            cv.imshow(WINDOW_NAME, frame)
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv.destroyAllWindows()


def main():
    args = parse_args()
    detector = load_face_detector(args.cascade)

    faces, labels, label_names = load_training_data(args.dataset, detector)
    if not faces:
        sys.exit("ไม่พบใบหน้าในฐานข้อมูลสำหรับ train กรุณาตรวจสอบรูปภาพ")

    recognizer = train_recognizer(faces, labels)
    print("Train โมเดลสำเร็จ พร้อมใช้งาน!")

    run_camera(args.camera, detector, recognizer, label_names, args.threshold)


if __name__ == "__main__":
    main()
