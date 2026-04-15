import paramiko
import cv2
import numpy as np
from time import sleep
from mqtt_tracker import MQTTFaceTracker
from camera_viewer import CameraViewer
import face_recognizer
import session_manager as sm

SSH_HOST = 'tiago-201c'
SSH_USER = 'pal'
SSH_PASS = 'pal'

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

recognizer = face_recognizer.FaceRecognizer(threshold=0.8)
recognizer.load_gallery("gallery")

session_manager = sm.SessionManager()
viewer = CameraViewer(window_name="Tiago", show_controls=True)

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

# ── Caméra + clavier ──────────────────────────────────────────────────────────

def stream_camera(ssh_cam, head_channel):
    script = """
import rospy
from sensor_msgs.msg import CompressedImage
import sys

def callback(msg):
    data = bytes(msg.data)
    try:
        sys.stdout.buffer.write(len(data).to_bytes(4, 'big'))
        sys.stdout.buffer.write(data)
        sys.stdout.buffer.flush()
    except Exception as e:
        rospy.signal_shutdown("SSH tunnel closed")

rospy.init_node('stream_node', anonymous=True)
rospy.Subscriber('/xtion/rgb/image_raw/compressed', CompressedImage, callback)
rospy.spin()
"""
    ssh_cam.exec_command(f"cat > /tmp/stream_camera.py << 'PYEOF'\n{script}\nPYEOF")
    sleep(1)

    transport = ssh_cam.get_transport()
    transport.set_keepalive(10) # Pour éviter le socket timeout (10054)
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


    # Initialisation de notre tracker MQTT
    tracker_mqtt = MQTTFaceTracker()
    tracker_mqtt.start()

    frame_skip = 4
    frame_count = 0
    last_boxes, last_names, last_distances, last_current_names = None, [], [], set()

    while True:
        try:
            size_bytes = read_exactly(4)
            size = int.from_bytes(size_bytes, 'big')
            jpeg_data = read_exactly(size)
            arr = np.frombuffer(jpeg_data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            if frame is not None:
                frame_count += 1
                
                # Exécute la reconnaissance faciale 1 frame sur 4 pour soulager le CPU
                if frame_count % frame_skip == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_boxes, last_names, last_distances, last_current_names = recognizer.process_frame(frame_rgb)
                    
                    # Mise à jour du tracker et envoi MQTT 
                    tracker_mqtt.update(last_current_names, session_manager)

                # Affiche toujours la frame (fluide) mais avec les boites de la dernière détection
                viewer.update_and_show(frame, last_boxes, last_names, last_distances, session_manager)


            key = viewer.wait_key(1)
            if key == ord('q'):
                break
            if key in KEY_MAP:
                send_head_command(head_channel, KEY_MAP[key])

        except Exception as e:
            print(f"Erreur vidéo: {e}")
            break

    tracker_mqtt.stop()
    viewer.close()
    channel.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ssh_cam   = paramiko.SSHClient()
    # ssh_audio = paramiko.SSHClient()
    ssh_head  = paramiko.SSHClient()

    for ssh in (ssh_cam, '''ssh_audio, ssh_head'''):
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(SSH_HOST, username=SSH_USER, password=SSH_PASS)

    head_channel = start_head_controller(ssh_head)

    # threading.Thread(target=stream_audio, args=(ssh_audio,), daemon=True).start()

    stream_camera(ssh_cam, head_channel)

    for ssh in (ssh_cam, '''ssh_audio, ssh_head'''):
        ssh.close()


main()