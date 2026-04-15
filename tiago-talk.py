import paramiko
from time import sleep


def build_tts_goal(text: str, lang_id: str) -> str:
    """Build the inline YAML payload for rostopic pub."""
    # On utilise des single quotes YAML autour des strings
    # => pas de conflit avec les double quotes bash qui encadreront le tout
    safe_text = text.replace("'", "''")  # Ã©chapper les apostrophes YAML
    return f"{{goal:{{rawtext:{{text: '{safe_text}', lang_id: '{lang_id}'}}}}}}"


def send_tts_via_ssh(message: str):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect('tiago-201c', username='pal', password='pal')

    payload = build_tts_goal(message, 'fr_FR')

    # bash -li = login + interactive => charge ~/.bashrc et ~/.bash_profile
    cmd = f'bash -lic "rostopic pub -1 /tts/goal pal_interaction_msgs/TtsActionGoal \\"{payload}\\""'

    print(f"CMD: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    print(stdout.read().decode())
    print(stderr.read().decode())
    sleep(1)
    ssh.close()


if __name__ == "__main__":
    print("Entrez votre message (ou tapez 'q' pour quitter).")
    while True:
        message = input("Message : ")
        if message.strip().lower() in ['q', 'quit', 'exit']:
            break
        if message.strip():
            send_tts_via_ssh(message)
