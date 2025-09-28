"""camera_pid controller with lane detection, manual control, and automatic steering adjustment."""

from controllers.project_02_controller.project_02_controller import Display, Keyboard, Robot, Camera
from vehicle import Car, Driver
import numpy as np
import cv2
from datetime import datetime
import os
import math

# Obtener la imagen de la cámara
def get_image(camera):
    raw_image = camera.getImage()
    image = np.frombuffer(raw_image, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )
    return image[:, :, :3]  # Convertir a formato BGR, eliminando el canal alfa

# Procesamiento de imagen para detectar carriles
def detect_lane_lines(image):
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Aplicar filtro Gaussiano para suavizar la imagen
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detectar bordes utilizando Canny
    edges = cv2.Canny(blur, 50, 150)

    # Definir una región de interés (ROI) en forma de triángulo
    height, width = edges.shape
    mask = np.zeros_like(gray)
    triangle =np.array([[(0, height),((width/2)*1, (height/4)*2.2), (width, height)]], dtype=np.int32)
    cv2.fillPoly(mask, triangle, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)

    # Utilizar la Transformada de Hough para detectar líneas
    #lines = cv2.HoughLinesP(cropped_edges, 1, np.pi / 180, 50, np.array([]), minLineLength=40, maxLineGap=5)
    lines = cv2.HoughLinesP(cropped_edges, 1.2, np.pi / 180, 50, np.array([]), minLineLength=50, maxLineGap=10)
    # Dibujar las líneas detectadas
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
    # Combinar las líneas con la imagen original
    combo_image = cv2.addWeighted(image, 1, line_image, 1, 1)
    return combo_image, lines

# Calcular el ángulo de dirección
def calculate_steering_angle(lines, width):
    if lines is None:
        return 0  # No hay líneas, dirección recta
    angles = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            angle = math.atan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)
    if len(angles) > 0:
        mean_angle = np.mean(angles)
        return -mean_angle * np.pi / 180  # Convertir a radianes para Webots y ajustar el signo si es necesario
    else:
        return -1.0 # No hay ángulos útiles, mantener dirección recta

# Mostrar imagen
def display_image(display, image):
    image_rgb = image[..., ::-1]  # Convertir BGR a RGB
    image_ref = display.imageNew(
        image_rgb.tobytes(),
        Display.RGB,
        width=image_rgb.shape[1],
        height=image_rgb.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)

def main():
    # Inicializar instancias
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    display_img = Display("display_image")
    keyboard = Keyboard()
    keyboard.enable(timestep)

    speed = 30  # Velocidad constante

    while robot.step() != -1:
        image = get_image(camera)
        lane_image, lines = detect_lane_lines(image)
        steering_angle = calculate_steering_angle(lines, image.shape[1])
        driver.setSteeringAngle(steering_angle)

        display_image(display_img, lane_image)  # Mostrar la imagen con líneas detectadas

        # Control manual
        key = keyboard.getKey()
        while key > 0:
            if key == keyboard.UP:
                speed += 5
            elif key == keyboard.DOWN:
                speed -= 5
            elif key == keyboard.RIGHT:
                driver.setSteeringAngle(driver.getSteeringAngle() + 0.1)
            elif key == keyboard.LEFT:
                driver.setSteeringAngle(driver.getSteeringAngle() - 0.1)
            elif key == ord('A'):
                # Guardar imagen
                current_datetime = str(datetime.now().strftime("%Y-%m-%d %H-%M-%S"))
                file_name = current_datetime + ".png"
                camera.saveImage(os.getcwd() + "/" + file_name, 1)
                print("Image saved:", file_name)
            key = keyboard.getKey()

        driver.setCruisingSpeed(speed)

if __name__ == "__main__":
    main()