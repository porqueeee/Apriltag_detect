from djitellopy import Tello
import time

tello = Tello()

print("Conectando...")
tello.connect()

time.sleep(3)

print("Batería:", tello.get_battery())