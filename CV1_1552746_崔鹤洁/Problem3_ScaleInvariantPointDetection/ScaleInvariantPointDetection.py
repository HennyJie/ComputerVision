import numpy as np
import cv2
import math
import scipy.ndimage as nd
import scipy

# Arguments:
#            im     - input image
#            sigma  - initial scale
#            k   - scale multiplication constant
#            sigma_final - largest scale to process
#            threshold - Laplacian threshold
# Returns:
#            r      - row coordinates of blob centers
#            c      - column coordinates of blob centers
#            rad    - radius of blobs


def filter2d(img, filter):
    pad_len = filter.shape[0]//2
    img_res = img.copy()
    img_res = cv2.copyMakeBorder(img_res, pad_len, pad_len, pad_len, pad_len, cv2.BORDER_DEFAULT)
    filtered = np.zeros(img.shape)
    for i in range(0, img.shape[0]):
        for j in range(0, img.shape[1]):
            filtered[i,j] = np.sum(filter*img_res[i: i+1+2*pad_len, j: j+1+2*pad_len])
    return filtered


def scale_invariant_point_detection(img, sigma, k, sigma_final, threshold):
    if img.shape[2] > 1:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # number of scale iterations
    n = math.ceil((math.log(sigma_final) - math.log(sigma))/math.log(k))

    # init scale space
    h = img.shape[0]
    w = img.shape[1]
    scale_space = np.zeros((h,w,n))

    # generate the Laplacian of Gaussian for the first scale level
    filt_size = 2 * math.ceil(3 * sigma) + 1
    log_filter = np.zeros((filt_size, filt_size))
    log_filter[filt_size // 2, filt_size // 2] = 1
    log_filter = nd.gaussian_laplace(log_filter, sigma)
    log_filter = sigma * sigma * log_filter

    # generate the Laplacian of Gaussian for the remaining levels
    img_res = img.copy()
    for i in range(n):
        im_filtered = filter2d(img_res, log_filter)
        im_filtered = np.power(im_filtered, 2)
        scale_space[:, :, i] = cv2.resize(im_filtered, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)

        if i != n-1:
            img_res = cv2.resize(img, (math.ceil(img.shape[1] / (k ** (i + 1))),
                                       math.ceil(img.shape[0] / (k ** (i + 1)))), interpolation=cv2.INTER_CUBIC)
            print('add layer...')

    # perform non-maximum suppression for each scale-space slice
    super_size = 5
    max_space = np.zeros((h, w, n))
    for i in range(n):
        max_space[:,:,i] = nd.generic_filter(scale_space[:,:,i], lambda x: np.max(x), size=(super_size, super_size))

    # perform non-maximum suppression between scales and threshold
    for i in range(n):
        max_space[:,:,i] = np.max(max_space[:,:,max(i-1,0):min(i+2,n-1)], axis=2)

    max_space = max_space * (max_space == scale_space)
    print("max: %f"%max_space.max())
    print("min: %f"%max_space.min())

    # record the positions and correspondence radius of scale invariant blobs
    r = []
    c = []
    rad = []
    for i in range(n):
        [rows, cols] = np.where(max_space[:,:,i] > threshold)
        num_blobs = len(rows)
        radii = sigma*(k**i)*(2**0.5)
        radii = [radii]*num_blobs
        r.extend(rows)
        c.extend(cols)
        rad.extend(radii)

    return c, r, rad










