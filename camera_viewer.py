import cv2
from PIL import Image

class CameraViewer:
    def __init__(self, window_name="preview", show_controls=False):
        self.window_name = window_name
        self.show_controls = show_controls
        cv2.namedWindow(self.window_name)

    def draw_faces(self, frame, boxes, names, distances, session_manager):
        """
        Dessine les rectangles et les labels sur l'image.
        Si la personne est déjà validée dans le `session_manager`, elle est en or (BGR: 0, 215, 255).
        Sinon (en attente), elle est en vert (BGR: 0, 255, 0).
        Si c'est un inconnu, c'est en rouge (BGR: 0, 0, 255).
        """
        for i, box in enumerate(boxes):
            name = names[i]
            dist = distances[i]

            x1, y1, x2, y2 = [int(b) for b in box]

            if name:
                is_published = session_manager.is_already_recognized(name)

                if is_published:
                    color = (0, 215, 255)  # Or (Gold) en BGR
                else:
                    color = (0, 255, 0)    # Vert en BGR
                label = f"{name} ({dist:.2f})"
            else:
                color = (0, 0, 255)        # Rouge en BGR
                label = f"Unknown ({dist:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    def show_no_face_warning(self, frame):
        cv2.putText(frame, "No face detected", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)

    def show_controls_info(self, frame):
        if self.show_controls:
            cv2.putText(frame, "Fleches: tete | c: centre | q: quitter",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    def update_and_show(self, frame, boxes, names, distances, session_manager):
        if boxes is not None:
            self.draw_faces(frame, boxes, names, distances, session_manager)
        else:
            self.show_no_face_warning(frame)
            
        self.show_controls_info(frame)
        cv2.imshow(self.window_name, frame)

    def wait_key(self, delay=1):
        return cv2.waitKey(delay) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)

