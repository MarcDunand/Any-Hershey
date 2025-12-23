import numpy as np
from skimage.morphology import skeletonize

bw = np.zeros((217, 564), dtype=bool)
bw[50:150, 200:260] = True  # simple filled block

print("before")
skel = skeletonize(bw)
print("after", skel.shape, skel.dtype)
