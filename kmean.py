import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

def main():
    img = cv.imread("01.png")
    # img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    vectorized = np.float32(img.reshape((-1, 3)))
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    
    K = 2
    attempts = 10
    ret, label, center = cv.kmeans(vectorized, K, None, criteria, attempts, cv.KMEANS_PP_CENTERS)
    print(center)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((img.shape))

    figure_size = 15
    plt.figure(figsize=(figure_size,figure_size))
    plt.subplot(2, 3, 1),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 2),plt.imshow(result_image)
    plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])
    plt.show()

if __name__ == "__main__":
    main()
    
