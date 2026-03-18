from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import os
from os import listdir
from os.path import isfile, join

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="vggface2").eval()


def compare_image(path_source: str, path_target: str) -> float:
    """
    Compare two dataset using MTCNN
    :param path_source: Source image
    :param path_target: Target image
    :return: Distance between the two dataset; lower is more similar
    """
    img_source = Image.open(path_source)
    img_target = Image.open(path_target)

    face_source = mtcnn(img_source).unsqueeze(0)
    face_target = mtcnn(img_target).unsqueeze(0)

    # Get bounding boxes
    # boxes_source, _ = mtcnn.detect(img_source)  # returns (boxes, probs)
    # boxes_target, _ = mtcnn.detect(img_target)
    #
    # # boxes_source is an array of [x1, y1, x2, y2]
    # for (x1, y1, x2, y2) in boxes_source:
    #     source_filename = os.path.basename(path_source)
    #     cropped_source = img_source.crop((x1, y1, x2, y2))
    #     cropped_source.save(os.path.join("output/", source_filename))

    emb1 = resnet(face_source)
    emb2 = resnet(face_target)

    dist = (emb1 - emb2).norm().item()
    return dist


def main():
    images_file = [f for f in listdir("dataset") if isfile(join("dataset", f))]

    index = 0

    os.path.exists("dataset") or os.mkdir("dataset")
    os.path.exists("output") or os.makedirs("output")

    for i in range(len(images_file)):
        for j in range(i + 1, len(images_file)):
            index += 1
            path_1 = join("dataset", images_file[i])
            path_2 = join("dataset", images_file[j])
            result = compare_image(path_1, path_2)
            print(f"[{index:03d}][{"✅" if result < 0.8 else "❌"}][{result:.4f}] Comparing {images_file[i]} and {images_file[j]}")


if __name__ == "__main__":
    main()
