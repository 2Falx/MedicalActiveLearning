import numpy as np
import cv2
import matplotlib.pyplot as plt

def canny(patch_image, prediction):

    # map 0 to 'non-vessel' and 1 to 'vessel'
    prediction = 'non-vessel' if prediction == 0 else 'vessel'

    norm_image = cv2.normalize(patch_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_image = norm_image.astype(np.uint8)
    gray_norm_image = cv2.cvtColor(norm_image, cv2.COLOR_BGR2GRAY)
    # Blur the image to reduce noise
    gray_norm_image = cv2.medianBlur(gray_norm_image, 5)
    edges = cv2.Canny(gray_norm_image, 150, 175)
    cnts = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[2:]
    for c in cnts:
        cv2.drawContours(gray_norm_image, [c], -1, (0, 0, 0), -1)

    # Threshold for vessels
    if prediction == 'vessel' and (np.mean(norm_image,axis=2).min() > 1):  # due to the dataset we classify borders of the eye as non vessels even if a small one is present
        # make it white and white and black
        darkest_pixel = gray_norm_image.min() # the ones where it's likely to have vessels
        # keep only black and switch it to white, switch all the rest to black
        # first check if too many pixels are classified as vessels (probably it would be a misclassification)
        th = 10
        if np.sum(gray_norm_image <= (darkest_pixel + th)) >= (0.25 * 32 * 32):  # 1/4 of the patch
            # in this case ignore the patche by masking it
            gray_norm_image[:, :] = 0  # mask it since it is no-vessel
            result_image_final = gray_norm_image.copy()
        else:
            result_image_final = gray_norm_image.copy()
            result_image_final[gray_norm_image <= darkest_pixel + th] = 255
            result_image_final[gray_norm_image > darkest_pixel + th] = 0
    else:
        result_image = gray_norm_image.copy()
        result_image[:,:] = 0  # mask it since it is no-vessel
        result_image_final = result_image.copy()

    return result_image_final