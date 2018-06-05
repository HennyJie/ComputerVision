import cv2
import numpy as np


def kps_and_features_detect(image):

    # detect and extract features of the image
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])

    return (kps, features)


def keypoints_match(kpsA, kpsB, featuresA, featuresB, ratio, maximum_pixelroom_for_RANSC):

    # compute the raw matches and initialize the list of true matches
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    raw_matches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in raw_matches:
        # ensure the distance is within a certain ratio of each other
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0].trainIdx, m[0].queryIdx))

    # use at least 4 matches to compute a homography
    if len(matches) > 4:
        # construct two sets of points
        ptsA = np.float32([kpsA[i] for (_, i) in matches])
        ptsB = np.float32([kpsB[i] for (i, _) in matches])

        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, maximum_pixelroom_for_RANSC)

        return (matches, H, status)

    return None


def matches_draw(imageA, imageB, kpsA, kpsB, matches, status):

    # initialize output image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the true matching points
    for ((trainIdx, queryIdx), s) in zip(matches, status):
        if s == 1:
            # draw the matches
            ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
            ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

    return vis


def panorama_stitching(images, ratio=0.75, maximum_pixelroom_for_RANSC =4.0, show_matches=False):

    # Detect key points and extract features of the image
    (imageB, imageA) = images
    (kpsA, featuresA) = kps_and_features_detect(imageA)
    (kpsB, featuresB) = kps_and_features_detect(imageB)

    # match features between the two images
    M = keypoints_match(kpsA, kpsB, featuresA, featuresB, ratio, maximum_pixelroom_for_RANSC)

    # situation that there aren't enough matched key points to create a panorama
    if M is None:
        return None

    # apply a warp to stitch the images together
    (matches, H, status) = M
    result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
    result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

    # key points matches be visualized or not
    if show_matches:
        vis = matches_draw(imageA, imageB, kpsA, kpsB, matches, status)
        return (result, vis)

    return result








