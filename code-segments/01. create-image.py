import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from random import randint

im = np.zeros((6, 8, 3), dtype=np.uint8)
im[2, 3] = [randint(0, 255), randint(0, 255), randint(0, 255)]
fig, ax = plt.subplots(1, 1, figsize=(6, 8))
im_display = ax.imshow(im)
ax.xaxis.set_ticks_position('top')
plt.show()