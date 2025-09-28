import os
import cv2
import time

import numpy as np

from vehicle import Car, Driver
from controller import Display, Keyboard, Robot, Camera

def init_camera(camera: Camera, timestep: int):
    camera.enable(timestep)


def limit(value, max = 1, min = -1):
    if value < min:
        return min
    elif value > max:
        return max
    else:
        return round(value, 4)


def get_image(camera: Camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]



def main():
    # Inicializar instancias
    vehicle = Car()
    timestep = int(vehicle.getBasicTimeStep())
    front_camera = vehicle.getDevice("front_camera")

    init_camera(front_camera, timestep)

    model = cv2.dnn.readNetFromONNX('./nvidia_model.onnx')
    start_time = time.time()
    
    vehicle.setCruisingSpeed(25)

    while vehicle.step() != -1:
        if time.time() - start_time > 0.25:
            image = get_image(front_camera)
            blob = cv2.dnn.blobFromImage(image, 1. / 255., (200, 66), (0, 0, 0), swapRB=True, crop=False)
            model.setInput(blob)
            steering_angle = model.forward()
            steering_angle = round(steering_angle[0][0], 3)
            print(steering_angle)
            vehicle.setSteeringAngle(steering_angle)
            start_time = time.time()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e.with_traceback())
        exit(0)