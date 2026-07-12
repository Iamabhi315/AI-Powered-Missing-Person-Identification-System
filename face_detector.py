import cv2
from ultralytics import YOLO
import config

class FaceDetector:
    def __init__(self):
        print("[INFO] Loading YOLO Face Detector...")
        self.model = YOLO(config.YOLO_MODEL_PATH)

    def detect_faces(self, image):
        """
        Takes image and return list of detected faces.
        Bounding box(x1, y1, x2, y2) and cropped face is present in every item

        """
        results = self.model(image, conf=0.5, verbose=False)
        boxes = results[0].boxes
        
        detected_faces = []
        h, w, _ = image.shape

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Dynamic Padding (10%)
            box_w, box_h = x2 - x1, y2 - y1
            pad_x, pad_y = int(box_w * 0.10), int(box_h * 0.10)

            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

            face_crop = image[y1:y2, x1:x2]
            
            detected_faces.append({
                "box": (x1, y1, x2, y2),
                "crop": face_crop
            })
            
        return detected_faces