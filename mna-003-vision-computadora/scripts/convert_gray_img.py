import argparse
import cv2


parser = argparse.ArgumentParser()
parser.add_argument("path_image_input", help="path to the input image")
parser.add_argument("path_image_output", help="path to the output image")


args = parser.parse_args()

img = cv2.imread(args.path_image_input)
cv2.imwrite(filename=args.path_image_output, img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

cv2.imshow("loaded image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()