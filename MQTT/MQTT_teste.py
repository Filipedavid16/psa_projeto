import paho.mqtt.client as mqtt
import numpy as np
import cv2
import time
import math
import json

#Função chamada automaticamente quando chega mensagem em MQTT
def on_message(client, userdata, msg):
    #Converte os dados recebidos (bytes) em array
    np_arr = np.frombuffer(msg.payload, np.uint8)
    #Converte o array em imagem real
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    #Mostra a imagem numa janela
    cv2.imshow("Webots Camera", frame)
    #Atualiza a janela (sem isto janela não aparece)
    cv2.waitKey(1)

#Cria o cliente MQTT
client = mqtt.Client()
#Liga a função ao evento (“quando chega mensagem -> chama esta função”)
client.on_message = on_message

#Liga ao broker MQTT (Mosquitto)
client.connect("localhost", 1883, 60)
#Diz: “quero receber mensagens deste tópico”
client.subscribe("webots/camera")

client.loop_start()
t = 0

#Comando da câmara
while True:
    pan = 2.0 * math.sin(0.75 * t)
    tilt = 0.2 * math.sin(2*t)

    data = {
        "pan": pan,
        "tilt": tilt
    }

    client.publish("webots/pan_tilt", json.dumps(data))

    t += 0.1
    time.sleep(0.05)
