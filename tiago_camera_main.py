import paramiko
import cv2
import numpy as np
import sounddevice as sd
import threading
from time import sleep
from PIL import Image
from webcam_main import mtcnn, resnet, load_gallery, find_best_match

SSH_HOST = 'tiago-201c'
SSH_USER = 'pal'
SSH_PASS = 'pal'
GAIN = 5.0

# Limites articulaires de la tête (en radians)
# head_1_joint : gauche/droite  (~-1.3 à 1.3)
# head_2_joint : haut/bas       (~-1.0 à 0.3)
HEAD_STEP = 0.1  # radians par appui


# ── Script ROS déposé sur Tiago ──────────────────────────────────────────────

HEAD_SCRIPT = """
import rospy
import sys
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

rospy.init_node('head_keyboard_node', anonymous=True)
pub = rospy.Publisher('/head_controller/command', JointTrajectory, queue_size=1)
rospy.sleep(0.5)

pan  = 0.0   # head_1_joint (gauche/droite)
tilt = 0.0   # head_2_joint (haut/bas)

PAN_MIN,  PAN_MAX  = -1.3,  1.3
TILT_MIN, TILT_MAX = -1.0,  0.3
STEP = 0.1

def send(p, t):
    msg = JointTrajectory()
    msg.joint_names = ['head_1_joint', 'head_2_joint']
    pt = JointTrajectoryPoint()
    pt.positions = [p, t]
    pt.time_from_start = rospy.Duration(0.2)
    msg.points = [pt]
    pub.publish(msg)

sys.stdout.write('READY\\n')
sys.stdout.flush()

for line in sys.stdin:
    cmd = line.strip()
    if   cmd == 'LEFT':  pan  = max(PAN_MIN,  pan  - STEP)
    elif cmd == 'RIGHT': pan  = min(PAN_MAX,  pan  + STEP)
    elif cmd == 'UP':    tilt = min(TILT_MAX, tilt + STEP)
    elif cmd == 'DOWN':  tilt = max(TILT_MIN, tilt - STEP)
    elif cmd == 'CENTER': pan = 0.0; tilt = 0.0
    else: continue
    send(pan, tilt)
    sys.stdout.write(f'PAN={pan:.2f} TILT={tilt:.2f}\\n')
    sys.stdout.flush()
"""

# ── Fonctions SSH ─────────────────────────────────────────────────────────────

def start_head_controller(ssh):
    """Dépose le script et retourne le channel SSH ouvert."""
    ssh.exec_command(f"cat > /tmp/head_keyboard.py << 'PYEOF'\n{HEAD_SCRIPT}\nPYEOF")
    sleep(1)
    transport = ssh.get_transport()
    channel = transport.open_session()
    channel.get_pty()
    channel.exec_command('bash -lic "python3 /tmp/head_keyboard.py"')

    # Attendre le signal READY
    buf = ''
    while 'READY' not in buf:
        buf += channel.recv(256).decode(errors='ignore')
    print("Contrôle tête prêt.")
    return channel


def send_head_command(channel, cmd):
    channel.sendall((cmd + '\n').encode())


# ── Audio ─────────────────────────────────────────────────────────────────────

def stream_audio(ssh_audio):
    transport = ssh_audio.get_transport()
    channel = transport.open_session()
    channel.exec_command('arecord -f S16_LE -r 44100 -c 1 -t raw 2>/dev/null')

    stream = sd.RawOutputStream(samplerate=44100, channels=1, dtype='int16', blocksize=4096)
    stream.start()
    try:
        while True:
            data = channel.recv(4096 * 2)
            if not data:
                break
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            samples *= GAIN
            np.clip(samples, -32768, 32767, out=samples)
            stream.write(samples.astype(np.int16).tobytes())
    except Exception as e:
        print(f"Erreur audio: {e}")
    finally:
        stream.stop()
        stream.close()
        channel.close()


# ── Caméra + clavier ──────────────────────────────────────────────────────────

def stream_camera(ssh_cam, head_channel, gallery, threshold=0.8):
    script = """
import rospy
from sensor_msgs.msg import CompressedImage
import sys

def callback(msg):
    data = bytes(msg.data)
    sys.stdout.buffer.write(len(data).to_bytes(4, 'big'))
    sys.stdout.buffer.write(data)
    sys.stdout.buffer.flush()

rospy.init_node('stream_node', anonymous=True)
rospy.Subscriber('/xtion/rgb/image_raw/compressed', CompressedImage, callback)
rospy.spin()
"""
    ssh_cam.exec_command(f"cat > /tmp/stream_camera.py << 'PYEOF'\n{script}\nPYEOF")
    sleep(1)

    transport = ssh_cam.get_transport()
    channel = transport.open_session()
    channel.exec_command('bash -lic "python3 /tmp/stream_camera.py"')

    def read_exactly(n):
        buf = b''
        while len(buf) < n:
            chunk = channel.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("Channel caméra fermé")
            buf += chunk
        return buf

    # Touches → commandes
    KEY_MAP = {
        ord('d'): 'LEFT',   # flèche gauche
        ord('g'): 'RIGHT',  # flèche droite
        ord('r'): 'UP',     # flèche haut
        ord('f'): 'DOWN',   # flèche bas
        ord('c'): 'CENTER',
    }

    print("Stream démarré. Flèches = tête | 'c' = centrer | 'q' = quitter")

    while True:
        try:
            size = int.from_bytes(read_exactly(4), 'big')
            jpeg_data = read_exactly(size)
            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is not None:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_frame = Image.fromarray(frame_rgb)

                boxes, _ = mtcnn.detect(img_frame)
                faces = mtcnn(img_frame)

                if faces is not None and boxes is not None:
                    embeddings = resnet(faces)

                    for i, emb in enumerate(embeddings):
                        name, dist = find_best_match(emb.unsqueeze(0), gallery, threshold)

                        x1, y1, x2, y2 = [int(b) for b in boxes[i]]
                        color = (0, 255, 0) if name else (0, 0, 255)
                        label = f"{name} ({dist:.2f})" if name else f"Unknown ({dist:.2f})"

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.putText(frame, "Fleches: tete | c: centre | q: quitter",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Tiago", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key in KEY_MAP:
                send_head_command(head_channel, KEY_MAP[key])

        except Exception as e:
            print(f"Erreur vidéo: {e}")
            break

    cv2.destroyAllWindows()
    channel.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    gallery = load_gallery("gallery")
    if not gallery:
        print("Galerie vide ou introuvable.")
    else:
        print(f"{len(gallery)} personnes chargées dans la galerie.")

    ssh_cam   = paramiko.SSHClient()
    ssh_audio = paramiko.SSHClient()
    ssh_head  = paramiko.SSHClient()

    for ssh in (ssh_cam, ssh_audio, ssh_head):
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SSH_HOST, username=SSH_USER, password=SSH_PASS)

    head_channel = start_head_controller(ssh_head)

    threading.Thread(target=stream_audio, args=(ssh_audio,), daemon=True).start()

    stream_camera(ssh_cam, head_channel, gallery)

    for ssh in (ssh_cam, ssh_audio, ssh_head):
        ssh.close()


main()