import time
import json
import paho.mqtt.client as mqtt

class MQTTFaceTracker:
    def __init__(self, broker_ip="10.167.129.217", port=1883, topic="edusign/attend/tiago", presence_time=3.0, tolerance=1.0):
        self.broker_ip = broker_ip
        self.port = port
        self.topic = topic
        self.presence_time = presence_time
        self.tolerance = tolerance
        
        self.client = mqtt.Client()
        self.tracked_persons = {}

    def start(self):
        try:
            self.client.connect(self.broker_ip, self.port, 60)
            self.client.loop_start()
            print(f"Connecté au broker MQTT {self.broker_ip}:{self.port}")
        except Exception as e:
            print(f"Erreur connexion MQTT: {e}")

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()

    def update(self, current_frame_names, session_manager):
        current_time = time.time()
        
        # Mise à jour des personnes détectées
        for name in current_frame_names:
            if session_manager.is_already_recognized(name):
                continue  # Déjà pris en compte
            
            if name not in self.tracked_persons:
                self.tracked_persons[name] = {"first_seen": current_time, "published": False}
            
            self.tracked_persons[name]["last_seen"] = current_time

            # Si on l'a vu pendant le temps requis (ex: 3s) et pas encore publié
            if not self.tracked_persons[name]["published"] and (current_time - self.tracked_persons[name]["first_seen"] >= self.presence_time):
                payload = json.dumps({"name": name, "id": name})
                self.client.publish(self.topic, payload)
                print(f"[MQTT] Envoyé: {payload}")
                self.tracked_persons[name]["published"] = True
                session_manager.add_student(name)

        # Nettoyage des personnes qui ne sont plus dans le cadre (avec une tolérance)
        to_remove = []
        for name, data in self.tracked_persons.items():
            if current_time - data["last_seen"] > self.tolerance:
                to_remove.append(name)
                
        for name in to_remove:
            del self.tracked_persons[name]
