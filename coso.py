from djitellopy import Tello
import time

tello = Tello()

print("Modo SDK...")
respuesta = tello.send_command_with_return("command")
print("Respuesta:", respuesta)

time.sleep(2)

try:
    print("Batería:", tello.query_battery())
except Exception as e:
    print("ERROR:", e)