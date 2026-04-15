from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import cv2

import session_manager
from mqtt_tracker import MQTTFaceTracker
from camera_viewer import CameraViewer

mtcnn = MTCNN(keep_all=True)  # keep_all=True to detect multiple faces per frame
resnet = InceptionResnetV1(pretrained="vggface2").eval()

session = session_manager.SessionManager()


def compare_image(path_source: str, path_target: str) -> float:
    """
    Compare two images using MTCNN + InceptionResnetV1
    :param path_source: Source image
    :param path_target: Target image
    :return: Distance between the two faces; lower is more similar
    """
    img_source = Image.open(path_source)
    img_target = Image.open(path_target)

    face_source = mtcnn(img_source)
    face_target = mtcnn(img_target)

    if face_source is None or face_target is None:
        print("No face detected in one of the images.")
        return float("inf")

    emb1 = resnet(face_source[0].unsqueeze(0))
    emb2 = resnet(face_target[0].unsqueeze(0))

    dist = (emb1 - emb2).norm().item()
    return dist


def load_gallery(folder_path: str) -> dict:
    """
    Pre-compute embeddings for all reference images in a folder.
    Filename (without extension) is used as the person's name.
    e.g. dataset/john.jpg → "john"
    """
    gallery = {}

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        name = os.path.splitext(filename)[0]
        path = os.path.join(folder_path, filename)

        img = Image.open(path)
        faces = mtcnn(img)

        if faces is None:
            print(f"[WARN] No face detected in {filename}, skipping.")
            continue

        emb = resnet(faces[0].unsqueeze(0))
        gallery[name] = emb
        print(f"[OK] Loaded: {name}")

    return gallery


def find_best_match(emb_target, gallery: dict, threshold: float = 0.8):
    """
    Compare a face embedding against all gallery embeddings.
    :return: (name, distance) of the closest match, or (None, distance) if no match
    """
    best_name = None
    best_dist = float("inf")

    for name, emb in gallery.items():
        dist = (emb_target - emb).norm().item()
        if dist < best_dist:
            best_dist = dist
            best_name = name

    if best_dist < threshold:
        return best_name, best_dist
    return None, best_dist


def compare_webcam_to_gallery(gallery_folder: str, threshold: float = 0.8):
    """
    Real-time webcam face recognition against a gallery of people.
    :param gallery_folder: Folder containing one reference image per person
    :param threshold: Distance threshold — below this is considered a match
    """
    print("Loading gallery...")
    gallery = load_gallery(gallery_folder)

    if not gallery:
        print("Gallery is empty. Add images to the dataset folder.")
        return

    print(f"Gallery ready: {len(gallery)} people loaded.\n")

    viewer = CameraViewer(window_name="preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
        print("Could not open webcam.")

    tracker_mqtt = MQTTFaceTracker()
    tracker_mqtt.start()

    frame_skip = 4
    frame_count = 0
    last_boxes, last_names, last_distances, last_current_names = None, [], [], set()

    while rval:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_count += 1

        # Exécute la reconnaissance faciale 1 frame sur 4 pour soulager le CPU
        if frame_count % frame_skip == 0:
            last_boxes, last_names, last_distances, last_current_names = recognizer.process_frame(frame_rgb)
            tracker_mqtt.update(last_current_names, session_manager)

        viewer.update_and_show(frame, last_boxes, last_names, last_distances, session_manager)

        rval, frame = vc.read()
        key = viewer.wait_key(20)
        if key == 27:  # exit on ESC
            break

    tracker_mqtt.stop()
    vc.release()
    viewer.close()
    cv2.destroyAllWindows()


def main():
    import os
    os.makedirs("dataset", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    images_file = [f for f in listdir("dataset") if isfile(join("dataset", f))]

    if not images_file:
        print("No images found in dataset/")
        return

    # --- Run webcam recognition against the full gallery ---
    compare_webcam_to_gallery("gallery", threshold=0.8)

    # --- Pairwise comparison (uncomment to use) ---
    # index = 0
    # for i in range(len(images_file)):
    #     for j in range(i + 1, len(images_file)):
    #         index += 1
    #         path_1 = join("dataset", images_file[i])
    #         path_2 = join("dataset", images_file[j])
    #         result = compare_image(path_1, path_2)
    #         print(f"[{index:03d}][{'✅' if result < 0.8 else '❌'}][{result:.4f}] Comparing {images_file[i]} and {images_file[j]}")


if __name__ == "__main__":
    main()