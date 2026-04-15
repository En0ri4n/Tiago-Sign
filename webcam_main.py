from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import cv2

mtcnn = MTCNN(keep_all=True)  # keep_all=True to detect multiple faces per frame
resnet = InceptionResnetV1(pretrained="vggface2").eval()


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

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    if vc.isOpened():
        rval, frame = vc.read()
    else:
        rval = False
        print("Could not open webcam.")

    while rval:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_frame = Image.fromarray(frame_rgb)

        boxes, _ = mtcnn.detect(img_frame)
        faces = mtcnn(img_frame)

        if faces is not None and boxes is not None:
            embeddings = resnet(faces)  # batch inference

            for i, emb in enumerate(embeddings):
                name, dist = find_best_match(emb.unsqueeze(0), gallery, threshold)

                x1, y1, x2, y2 = [int(b) for b in boxes[i]]
                color = (0, 255, 0) if name else (0, 0, 255)
                label = f"{name} ({dist:.2f})" if name else f"Unknown ({dist:.2f})"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cv2.putText(frame, "No face detected", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

        cv2.imshow("preview", frame)
        rval, frame = vc.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break

    vc.release()
    cv2.destroyAllWindows()


def main():
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