import numpy as np
import cv2
from skimage.morphology import skeletonize

img = np.zeros((100,100), dtype=np.uint8)
img[20:80, 45:55] = 255

skel = skeletonize(img > 0)
print("Pipeline OK")
