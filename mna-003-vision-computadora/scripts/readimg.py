import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("path_image", help="path to the input image")


args = parser.parse_args()

img = cv2.imread(args.path_image)

cv2.imshow("loaded image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()