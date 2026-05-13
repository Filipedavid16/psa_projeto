import threading
import paho.mqtt.client as mqtt
import numpy as np
import cv2


class MQTTFrameReceiver:
    def __init__(self, broker="localhost", port=1883, topic="webots/camera"):
        self.broker = broker
        self.port = port
        self.topic = topic

        self.client = mqtt.Client()
        self.client.on_message = self._on_message

        self.lock = threading.Lock()
        self.latest_frame = None
        self.connected = False

    def _on_message(self, client, userdata, msg):
        try:
            np_arr = np.frombuffer(msg.payload, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if frame is not None:
                with self.lock:
                    self.latest_frame = frame
        except Exception as e:
            print(f"Erro ao descodificar frame MQTT: {e}")

    def start(self):
        self.client.connect(self.broker, self.port, 60)
        self.client.subscribe(self.topic)
        self.client.loop_start()
        self.connected = True
        print(f"Ligado ao broker MQTT em {self.broker}:{self.port}, tópico '{self.topic}'.")

    def read(self):
        with self.lock:
            if self.latest_frame is None:
                return False, None
            return True, self.latest_frame.copy()

    def stop(self):
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass