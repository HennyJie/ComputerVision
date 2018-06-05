import argparse
import imutils
import cv2
import sys
from panorama_stiching import panorama_stitching


def main(args):

    # load two images, resize them to width 400 pixels
    image_left = cv2.imread(args.first)
    image_right = cv2.imread(args.second)
    image_left = imutils.resize(image_left, width=400)
    image_right = imutils.resize(image_right, width=400)

    # panorama creating
    (result, vis) = panorama_stitching([image_left, image_right], show_matches=True)

    cv2.imshow("Image left", image_left)
    cv2.imshow("Image right", image_right)
    cv2.imshow("Key point Matches", vis)
    cv2.imshow("Panorama", result)
    print('Press any key to exit!')
    cv2.waitKey(0)


# construct the argument parse and parse the arguments
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--first", help="path to the first image", default='images/bryce_left.png')
    parser.add_argument("-s", "--second", help="path to the second image", default='images/bryce_right.png')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
