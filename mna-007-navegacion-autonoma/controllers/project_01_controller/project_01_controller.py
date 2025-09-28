import cv2
import numpy as np

from controller import Display  # type: ignore
from vehicle import Car, Driver  # type: ignore


def display_image(image, display):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        image.tobytes(),
        Display.RGB,
        width=image.shape[1],
        height=image.shape[0],
    )
    display.imagePaste(image_ref, 0, 0, False)


def get_image(camera):
    raw_bytes = camera.getImage()
    buf_bytes = np.frombuffer(raw_bytes, np.uint8)
    image = buf_bytes.reshape((camera.getHeight(), camera.getWidth(), 4))  # type: ignore
    return image[:, :, :3]  # Convertir a formato BGR, eliminando el canal alfa


def segment_roads(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(3, 3))

    lower = np.array([110, 3, 15])
    upper = np.array([120, 23, 50])
    mask1 = cv2.inRange(hsv_image, lower, upper)

    lower = np.array([20, 100, 120])
    upper = np.array([30, 150, 220])
    mask2 = cv2.inRange(hsv_image, lower, upper)

    lower = np.array([110, 0, 180])
    upper = np.array([120, 15, 230])
    mask3 = cv2.inRange(hsv_image, lower, upper)

    mask3[:, 0 : mask3.shape[1] // 2] = 0

    mask4 = mask1 + mask2 + mask3

    full_mask = cv2.morphologyEx(mask4, cv2.MORPH_CLOSE, kernel)  # type: ignore

    result = cv2.bitwise_and(image, image, mask=full_mask)  # type: ignore

    return result, full_mask, mask2, mask3


def prepare_image(image, mask):
    segmented = cv2.bitwise_and(image, image, mask=mask)
    gray_image = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edge_image = cv2.Canny(binary, 50, 150)

    contours, _ = cv2.findContours(edge_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=lambda c: cv2.arcLength(c, True), reverse=True)

    return contours


def find_line(points, delta=0):
    if len(points) > 2:
        x = points[:, 0]
        y = points[:, 1]

        coefficients = np.polyfit(x, y, 2)
        x_fit = np.linspace(min(x) - delta, max(x) + delta, 100)
        y_fit = np.polyval(coefficients, x_fit)

    return x_fit, y_fit


def draw_fitted_line(image, x, y):
    copy = image.copy()
    for i in range(len(x) - 1):
        pt1 = (int(x[i]), int(y[i]))
        pt2 = (int(x[i + 1]), int(y[i + 1]))
        cv2.line(copy, pt1, pt2, (0, 255, 0), 2)
    return copy


def detect_lanes(image, mask1, mask2):
    contours1 = prepare_image(image, mask1)
    contours2 = prepare_image(image, mask2)

    points1 = contours1[0].reshape(-1, 2)

    points2 = np.zeros((0, 2))
    for contour in contours2[:3]:
        contour = contour.reshape(-1, 2)
        points2 = np.concat([points2, contour], axis=0)

    x1, y1 = find_line(points1)
    x2, y2 = find_line(points2, delta=25)

    return (x1, y1), (x2, y2)


def get_middle_line(line1, line2):
    y = (line1[1] + np.flip(line2[1])) / 2.0
    x = (line1[0] + np.flip(line2[0])) / 2.0
    return x, y


def get_stering_angle(line, L):
    x_tp, y_tp = line[0][-1], line[1][-1]
    ld = np.sqrt(np.pow(x_tp - line[0][0], 2) + np.pow(y_tp - line[1][-1], 2))

    alpha = np.atan2(y_tp, x_tp)

    stering_angle = np.atan2((2 * L * np.sin(alpha)) / (ld + 1e-5), 1.0)

    return np.clip(stering_angle, -np.pi / 4, np.pi / 4), x_tp, y_tp


def main():
    # Inicializar instancias
    robot = Car()
    driver = Driver()
    timestep = int(robot.getBasicTimeStep())
    camera1 = robot.getDevice("front_camera")
    camera1.enable(timestep)

    display = Display("display")

    speed = 5
    track_front = robot.getWheelbase()

    while robot.step() != -1:
        image = get_image(camera1)
        roads, mask1, mask2, mask3 = segment_roads(image)
        line1, line2 = detect_lanes(image, mask2, mask3)
        x, y = get_middle_line(line1, line2)
        delta, x_tp, y_tp = get_stering_angle(line=(x, y), L=track_front)

        cv2.putText(
            roads,
            f"{delta:.5f}",
            (10, 20),
            cv2.FONT_HERSHEY_PLAIN,
            1.0,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.circle(roads, (int(x_tp), int(y_tp)), 5, (255, 0, 0), -1)

        roads = draw_fitted_line(roads, x, y)

        display_image(roads, display)

        # driver.setSteeringAngle(delta)
        # driver.setCruisingSpeed(speed)


if __name__ == "__main__":
    main()
