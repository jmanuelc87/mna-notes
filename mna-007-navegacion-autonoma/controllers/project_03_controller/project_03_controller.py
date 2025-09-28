import os
import cv2
import csv
import time
import XInput
import pathlib
import logging
import threading
import traceback
import numpy as np
import pandas as pd

from vehicle import Car
from controller import Camera, Display



def init_camera(camera: Camera, display: Display, timestep: int):
    camera.enable(timestep)
    display.attachCamera(camera)

def get_image(camera: Camera):
    height = camera.getHeight()
    width = camera.getWidth()
    channels  = 4
    raw = np.frombuffer(camera.getImage(), dtype=np.uint8).reshape((height, width, channels))
    return cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)

def interpolation(value, x1=0, x2=1, y1=0, y2=1):
    return y1 + (( value - x1 ) * ( y2 - y1 )) / (x2 - x1)

def limit(value, min=-0.75, max=0.75):
    if value < min:
        return min
    elif value > max:
        return max
    else:
        return value


class DatasetCollector:

    def __init__(self, path, car: Car, camera: Camera, filename="dataset.csv", time_step=0.25) -> None:
        self.path = pathlib.Path(path).with_suffix("")
        self.car = car
        self.camera = camera
        self.time_step = time_step
        self.start_time = time.time()
        self.next = 0
        self.running = False
        self.filenames = ["image_name", "steering_angle"]
        
        filepath = self.path / filename
        
        if not os.path.exists(filepath):
            with open(filepath, 'w+', newline='') as file:
                csvWriter = csv.DictWriter(file, fieldnames=self.filenames)
                if os.path.getsize(filepath) == 0:
                    csvWriter.writeheader()
        else:
            frame = pd.read_csv(filepath)
            self.next = frame['image_name'].count()
            
        if not os.path.exists(self.path / "Train"):
            os.mkdir(self.path / "Train")

    def start(self):
        print("Starting recording...")
        self.running = True
        self.thread = threading.Thread(target=self.__process)
        self.thread.start()

    def stop(self):
        print("Stoping recording...")
        self.running = False

    def __process(self):
        while self.running:
            self.__gather()

    def __gather(self):
        if time.time() - self.start_time >= self.time_step:
            self.__collect()
            self.start_time = time.time()
            self.next += 1

    def __collect(self):
        with open(self.path / "dataset.csv", "a+", newline='') as file:
            image = get_image(self.camera)
            image_name = str(self.path / "Train" / f"{self.next:07d}.png")
            steering_angle = self.car.getSteeringAngle()
            cv2.imwrite(image_name, image)
            csvWriter = csv.DictWriter(file, fieldnames=self.filenames)
            csvWriter.writerow({"image_name": f"{self.next:07d}.png", "steering_angle": round(steering_angle, 5)})


class GamepadController(XInput.EventHandler):

    def __init__(self, *controllers, sdv=None, collector=None, filter=...):
        super().__init__(*controllers, filter=filter)
        self.sdv = sdv
        self.collector = collector
    
    def process_button_event(self, event):
        if event.type == XInput.EVENT_BUTTON_PRESSED:
            if event.button == "A":
                # start recording
                self.collector.start()
            elif event.button == "B":
                # stop recording
                self.collector.stop()

    def process_trigger_event(self, event):
        if event.trigger == XInput.LEFT:
            brake_intensity = round(event.value, 4)
            self.sdv.setBrakeIntensity(brake_intensity)
        elif event.trigger == XInput.RIGHT:
            speed = round(event.value * 40, 4)
            self.sdv.setCruisingSpeed(speed)

    def process_stick_event(self, event):
        if event.stick == XInput.LEFT:
            x, y = round(event.x, 6), round(event.y, 6)
            theta = -90
            c = np.cos(theta)
            s = np.sin(theta)
            
            rot = np.array([[c, -s], [s, c]])
            vec = np.array([x, y])
            
            real = rot @ vec
            
            angle = -int(np.arctan2(real[1], real[0]) * 180 / np.pi)
            steering_angle = round(interpolation(angle, 180, -180, 1, -1), 3)
            
            self.sdv.setSteeringAngle(limit(steering_angle))

    def process_connection_event(self, event):
        if event.type == XInput.EVENT_CONNECTED:
            print("Controller Connected")
        elif event.type == XInput.EVENT_DISCONNECTED:
            print("Controller Disconnected")


def main():
    sdv = Car()
    timestep = int(sdv.getBasicTimeStep())
    
    camera = sdv.getDevice('front_camera')
    display = sdv.getDevice('display')
    init_camera(camera, display, timestep)
    
    if not any(XInput.get_connected()):
        logging.info("No gamepads connected!")
        
    collector = DatasetCollector("D:\\webots_lanes", sdv, camera)

    gamepad = XInput.GamepadThread()
    handler = GamepadController(0, sdv=sdv, collector=collector)
    handler.set_filter(XInput.STICK_LEFT + XInput.BUTTON_A + XInput.BUTTON_B + XInput.BUTTON_X + XInput.TRIGGER_RIGHT + XInput.TRIGGER_LEFT)
    gamepad.add_event_handler(handler)


    while sdv.step() != -1:
        pass



if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(traceback.format_exc())