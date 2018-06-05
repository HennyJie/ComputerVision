import cv2
import argparse
import sys
from ScaleInvariantPointDetection import scale_invariant_point_detection


def main(args):

    img = cv2.imread(args.picture_path)

    # initial parameters
    sigma_init = args.sigma_init
    sigma_final = args.sigma_final
    s = args.s
    k = 2**(1/s)
    threshold = args.threshold

    # Detect Scale Invariant Points
    r, c, rad = scale_invariant_point_detection(img, sigma_init, k, sigma_final, threshold)

    print("Drawing blobs...")
    # Draw the circles to indicate the positions and characteristic scale
    for x, y, r in zip(r,c,rad):
        cv2.circle(img, (int(x), int(y)), int(r), (0, 0, 255), 1)

    cv2.imshow("Scale Invariant Blob Detection", img)
    print("Press any key to exit!")
    cv2.waitKey(0)


# construct the argument parse and parse the arguments
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--picture_path', type=str, help="Directory where to get the source picture.", default='butterfly.jpg')
    parser.add_argument('-i', '--sigma_init', type=float, help="The initial value of sigma.", default=1.6)
    parser.add_argument('-f', '--sigma_final', type=float, help="The largest scale to process.", default=15)
    parser.add_argument('-s', '--s', type=float, help="The number of scales for calculating the extreme value.", default=3)
    parser.add_argument('-t', '--threshold', type=int, help="Laplacian threshold.", default=500)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))