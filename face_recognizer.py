from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
import cv2

class FaceRecognizer:
    def __init__(self, threshold=0.8):
        self.mtcnn = MTCNN(keep_all=True)
        self.resnet = InceptionResnetV1(pretrained="vggface2").eval()
        self.threshold = threshold
        self.gallery = {}

    def load_gallery(self, folder_path: str):
        self.gallery = {}
        if not os.path.exists(folder_path):
            print(f"Directory {folder_path} doesn't exist.")
            return self.gallery

        for filename in os.listdir(folder_path):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            name = os.path.splitext(filename)[0]
            path = os.path.join(folder_path, filename)

            img = Image.open(path)
            faces = self.mtcnn(img)

            if faces is None:
                print(f"[WARN] No face detected in {filename}, skipping.")
                continue

            emb = self.resnet(faces[0].unsqueeze(0))
            self.gallery[name] = emb
            print(f"[OK] Loaded: {name}")

        return self.gallery

    def find_best_match(self, emb_target):
        best_name = None
        best_dist = float("inf")

        for name, emb in self.gallery.items():
            dist = (emb_target - emb).norm().item()
            if dist < best_dist:
                best_dist = dist
                best_name = name

        if best_dist < self.threshold:
            return best_name, best_dist
        return None, best_dist

    def process_frame(self, frame_rgb):
        img_frame = Image.fromarray(frame_rgb)
        boxes, _ = self.mtcnn.detect(img_frame)

        current_frame_names = set()
        names = []
        distances = []

        if boxes is not None:
            faces = self.mtcnn.extract(img_frame, boxes, None)
            if faces is not None:
                embeddings = self.resnet(faces)
                for emb in embeddings:
                    name, dist = self.find_best_match(emb.unsqueeze(0))
                    if name:
                        current_frame_names.add(name)
                    names.append(name)
                    distances.append(dist)

        return boxes, names, distances, current_frame_names
